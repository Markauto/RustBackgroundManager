use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{Context as _, Result};
use chrono::Utc;
use rusqlite::{Transaction, params};
use serde::Serialize;
use walkdir::WalkDir;

use crate::{
    AppPaths,
    analysis::{
        ImageAnalysis, analyze_image, decode_image, probe_dimensions, within_import_bounds,
        write_thumbnail,
    },
    config::Config,
    db::{Database, ImageStatus, SourceRoot, path_bytes, path_from_bytes},
    filesystem::blake3_file,
};

const SCAN_WRITE_BATCH_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOptions {
    pub full: bool,
    pub no_ai: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ScanReport {
    pub discovered: usize,
    pub analyzed: usize,
    pub unchanged: usize,
    pub out_of_bounds: usize,
    pub missing: usize,
    pub failed: usize,
    pub ai_deferred: usize,
    pub failures: Vec<FileFailure>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FileFailure {
    pub path: PathBuf,
    pub error: String,
}

#[derive(Clone, Debug)]
pub enum ScanEvent {
    Started { files: usize },
    Processing { index: usize, path: PathBuf },
    Failed(FileFailure),
    Finished(ScanReport),
}

#[derive(Clone, Debug)]
struct Discovery {
    source_id: i64,
    path: PathBuf,
    size: u64,
    modified_ns: i64,
}

#[derive(Clone, Debug)]
struct Existing {
    id: i64,
    source_id: i64,
    size: u64,
    modified_ns: i64,
    hash: Option<String>,
    status: ImageStatus,
    analysis_key: Option<String>,
}

#[derive(Clone, Debug)]
enum ProcessOutcome {
    Complete(Option<CatalogMutation>),
    RecordedFailure {
        mutation: CatalogMutation,
        error: String,
    },
}

#[derive(Clone, Debug)]
enum CatalogMutation {
    Analysis {
        discovery: Discovery,
        hash: String,
        thumbnail: PathBuf,
        analysis: ImageAnalysis,
    },
    OutOfBounds {
        discovery: Discovery,
        dimensions: (u32, u32),
    },
    Status {
        discovery: Discovery,
        status: ImageStatus,
        error: String,
        analysis_key: String,
    },
    Touch {
        image_id: i64,
        discovery: Discovery,
    },
    RestoreReady {
        image_id: i64,
        discovery: Discovery,
    },
}

struct ProcessContext<'a> {
    paths: &'a AppPaths,
    config: &'a Config,
    options: ScanOptions,
    analysis_key: &'a str,
}

pub fn scan_catalog(
    database: &Database,
    paths: &AppPaths,
    config: &Config,
    options: ScanOptions,
) -> Result<ScanReport> {
    scan_catalog_with_progress(database, paths, config, options, |_| {})
}

pub fn scan_catalog_with_progress(
    database: &Database,
    paths: &AppPaths,
    config: &Config,
    options: ScanOptions,
    mut progress: impl FnMut(ScanEvent),
) -> Result<ScanReport> {
    let sources = database.list_sources()?;
    let (discoveries, traversal_failures, unreliable_sources) = discover(&sources);
    let mut existing = existing_images(database)?;
    progress(ScanEvent::Started {
        files: discoveries.len(),
    });
    let analysis_key = analysis_key(config)?;
    let mut report = ScanReport {
        discovered: discoveries.len(),
        ..ScanReport::default()
    };
    let context = ProcessContext {
        paths,
        config,
        options,
        analysis_key: &analysis_key,
    };
    let mut mutations = Vec::with_capacity(SCAN_WRITE_BATCH_SIZE);
    for failure in traversal_failures {
        record_failure(&mut report, &mut progress, failure);
    }

    for (index, discovery) in discoveries.into_iter().enumerate() {
        let previous = existing.remove(&discovery.path);
        progress(ScanEvent::Processing {
            index: index + 1,
            path: discovery.path.clone(),
        });
        let mutation = match process_file(&context, &discovery, previous.as_ref(), &mut report) {
            Ok(ProcessOutcome::Complete(mutation)) => mutation,
            Ok(ProcessOutcome::RecordedFailure { mutation, error }) => {
                record_failure(
                    &mut report,
                    &mut progress,
                    FileFailure {
                        path: discovery.path.clone(),
                        error,
                    },
                );
                Some(mutation)
            }
            Err(error) => {
                let message = format!("{error:#}");
                record_failure(
                    &mut report,
                    &mut progress,
                    FileFailure {
                        path: discovery.path.clone(),
                        error: message.clone(),
                    },
                );
                Some(CatalogMutation::Status {
                    discovery: discovery.clone(),
                    status: ImageStatus::Error,
                    error: message,
                    analysis_key: String::new(),
                })
            }
        };
        if let Some(mutation) = mutation {
            mutations.push(mutation);
        }
        if mutations.len() >= SCAN_WRITE_BATCH_SIZE {
            persist_mutations(database, &mutations, &analysis_key)?;
            mutations.clear();
        }
    }
    persist_mutations(database, &mutations, &analysis_key)?;
    report.missing = mark_missing(database, &sources, &unreliable_sources, &existing)?;
    progress(ScanEvent::Finished(report.clone()));
    Ok(report)
}

fn discover(sources: &[SourceRoot]) -> (Vec<Discovery>, Vec<FileFailure>, HashSet<i64>) {
    let mut by_path: HashMap<PathBuf, Discovery> = HashMap::new();
    let mut failures = Vec::new();
    let mut unreliable_sources = HashSet::new();
    let registered_roots = sources
        .iter()
        .map(|source| source.path.clone())
        .collect::<HashSet<_>>();
    for source in sources {
        // A registered nested root owns its subtree. Pruning it from the
        // parent's walk avoids reading the same directory tree twice.
        let entries = WalkDir::new(&source.path)
            .follow_links(false)
            .into_iter()
            .filter_entry(|entry| entry.depth() == 0 || !registered_roots.contains(entry.path()));
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    unreliable_sources.insert(source.id);
                    failures.push(FileFailure {
                        path: error
                            .path()
                            .map_or_else(|| source.path.clone(), Path::to_owned),
                        error: error.to_string(),
                    });
                    continue;
                }
            };
            let file_type = entry.file_type();
            if file_type.is_symlink() || !file_type.is_file() || !is_supported_image(entry.path()) {
                continue;
            }
            let metadata = match entry.metadata() {
                Ok(metadata) => metadata,
                Err(error) => {
                    unreliable_sources.insert(source.id);
                    failures.push(FileFailure {
                        path: entry.path().to_owned(),
                        error: error.to_string(),
                    });
                    continue;
                }
            };
            match metadata
                .modified()
                .map(|modified| (metadata.len(), modified))
            {
                Ok((size, modified)) => {
                    let modified_ns = modified.duration_since(UNIX_EPOCH).map_or(0, |duration| {
                        i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX)
                    });
                    by_path.insert(
                        entry.path().to_owned(),
                        Discovery {
                            source_id: source.id,
                            path: entry.path().to_owned(),
                            size,
                            modified_ns,
                        },
                    );
                }
                Err(error) => {
                    unreliable_sources.insert(source.id);
                    failures.push(FileFailure {
                        path: entry.path().to_owned(),
                        error: error.to_string(),
                    });
                }
            }
        }
    }
    let mut discoveries: Vec<_> = by_path.into_values().collect();
    discoveries.sort_by(|left, right| left.path.cmp(&right.path));
    (discoveries, failures, unreliable_sources)
}

fn process_file(
    context: &ProcessContext<'_>,
    discovery: &Discovery,
    existing: Option<&Existing>,
    report: &mut ScanReport,
) -> Result<ProcessOutcome> {
    if let Some(record) = existing
        && !context.options.full
        && record.size == discovery.size
        && record.modified_ns == discovery.modified_ns
        && record.analysis_key.as_deref() == Some(context.analysis_key)
    {
        match record.status {
            ImageStatus::Ready if record.hash.is_some() => {
                return reuse_ready_analysis(context, discovery, record, report);
            }
            ImageStatus::OutOfBounds => {
                report.out_of_bounds += 1;
                return Ok(ProcessOutcome::Complete(discovery_touch_if_changed(
                    record, discovery,
                )));
            }
            _ => {}
        }
    }

    let dimensions = match probe_dimensions(&discovery.path) {
        Ok(dimensions) => dimensions,
        Err(error) => {
            let message = format!("{error:#}");
            return Ok(ProcessOutcome::RecordedFailure {
                mutation: CatalogMutation::Status {
                    discovery: discovery.clone(),
                    status: ImageStatus::Corrupt,
                    error: message.clone(),
                    analysis_key: context.analysis_key.to_owned(),
                },
                error: message,
            });
        }
    };
    if !within_import_bounds(dimensions.0, dimensions.1, &context.config.import) {
        let mutation = CatalogMutation::OutOfBounds {
            discovery: discovery.clone(),
            dimensions,
        };
        report.out_of_bounds += 1;
        return Ok(ProcessOutcome::Complete(Some(mutation)));
    }

    let hash = blake3_file(&discovery.path)?;
    if let Some(record) = existing
        && !context.options.full
        && record.hash.as_deref() == Some(&hash)
        && matches!(record.status, ImageStatus::Ready | ImageStatus::Missing)
        && record.analysis_key.as_deref() == Some(context.analysis_key)
    {
        return reuse_ready_analysis(context, discovery, record, report);
    }

    let (analysis, decoded) = analyze_image(&discovery.path, &context.config.analysis)?;
    let thumbnail = context.paths.thumbnails_dir.join(format!("{hash}.jpg"));
    if context.options.full || !thumbnail.exists() {
        write_thumbnail(
            &decoded,
            &thumbnail,
            context.config.analysis.thumbnail_long_edge,
        )?;
    }
    let mutation = CatalogMutation::Analysis {
        discovery: discovery.clone(),
        hash,
        thumbnail,
        analysis,
    };
    report.analyzed += 1;
    if context.options.no_ai {
        report.ai_deferred += 1;
    }
    Ok(ProcessOutcome::Complete(Some(mutation)))
}

fn reuse_ready_analysis(
    context: &ProcessContext<'_>,
    discovery: &Discovery,
    existing: &Existing,
    report: &mut ScanReport,
) -> Result<ProcessOutcome> {
    let hash = existing
        .hash
        .as_deref()
        .context("ready image has no retained content hash")?;
    let thumbnail = context.paths.thumbnails_dir.join(format!("{hash}.jpg"));
    match std::fs::metadata(&thumbnail) {
        Ok(metadata) if metadata.is_file() => {}
        Ok(_) => anyhow::bail!(
            "thumbnail cache path is not a regular file: {}",
            thumbnail.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let decoded = decode_image(&discovery.path)?;
            write_thumbnail(
                &decoded,
                &thumbnail,
                context.config.analysis.thumbnail_long_edge,
            )?;
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to inspect thumbnail cache at {}",
                    thumbnail.display()
                )
            });
        }
    }
    report.unchanged += 1;
    if context.options.no_ai {
        report.ai_deferred += 1;
    }
    let mutation = if existing.status == ImageStatus::Missing {
        Some(CatalogMutation::RestoreReady {
            image_id: existing.id,
            discovery: discovery.clone(),
        })
    } else {
        discovery_touch_if_changed(existing, discovery)
    };
    Ok(ProcessOutcome::Complete(mutation))
}

fn discovery_touch_if_changed(
    existing: &Existing,
    discovery: &Discovery,
) -> Option<CatalogMutation> {
    if existing.source_id != discovery.source_id
        || existing.size != discovery.size
        || existing.modified_ns != discovery.modified_ns
    {
        return Some(CatalogMutation::Touch {
            image_id: existing.id,
            discovery: discovery.clone(),
        });
    }
    None
}

fn existing_images(database: &Database) -> Result<HashMap<PathBuf, Existing>> {
    database.with_connection(|connection| {
        let mut statement = connection.prepare(
            "SELECT id, source_id, path, size, modified_ns, blake3, status, analysis_key
             FROM images",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                path_from_bytes(row.get_ref(2)?.as_blob()?),
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(3)? as u64,
                row.get::<_, i64>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })?;
        rows.map(|row| {
            let (path, id, source_id, size, modified_ns, hash, status, analysis_key) = row?;
            Ok((
                path,
                Existing {
                    id,
                    source_id,
                    size,
                    modified_ns,
                    hash,
                    status: ImageStatus::parse(&status)?,
                    analysis_key,
                },
            ))
        })
        .collect()
    })
}

fn persist_mutations(
    database: &Database,
    mutations: &[CatalogMutation],
    analysis_key: &str,
) -> Result<()> {
    if mutations.is_empty() {
        return Ok(());
    }
    database.with_transaction(|transaction| {
        for mutation in mutations {
            let result = match mutation {
                CatalogMutation::Analysis {
                    discovery,
                    hash,
                    thumbnail,
                    analysis,
                } => store_analysis(
                    transaction,
                    discovery,
                    hash,
                    analysis_key,
                    thumbnail,
                    analysis,
                ),
                CatalogMutation::OutOfBounds {
                    discovery,
                    dimensions,
                } => store_out_of_bounds(transaction, discovery, *dimensions, analysis_key),
                CatalogMutation::Status {
                    discovery,
                    status,
                    error,
                    analysis_key,
                } => store_status(transaction, discovery, *status, error, analysis_key),
                CatalogMutation::Touch {
                    image_id,
                    discovery,
                } => touch_existing(transaction, *image_id, discovery),
                CatalogMutation::RestoreReady {
                    image_id,
                    discovery,
                } => restore_ready(transaction, *image_id, discovery),
            };
            result.with_context(|| {
                format!(
                    "failed to update the catalog for {}",
                    mutation.discovery().path.display()
                )
            })?;
        }
        Ok(())
    })
}

impl CatalogMutation {
    const fn discovery(&self) -> &Discovery {
        match self {
            Self::Analysis { discovery, .. }
            | Self::OutOfBounds { discovery, .. }
            | Self::Status { discovery, .. }
            | Self::Touch { discovery, .. }
            | Self::RestoreReady { discovery, .. } => discovery,
        }
    }
}

fn store_analysis(
    transaction: &Transaction<'_>,
    discovery: &Discovery,
    hash: &str,
    analysis_key: &str,
    thumbnail: &Path,
    analysis: &ImageAnalysis,
) -> Result<()> {
    let now = Utc::now().timestamp_millis();
    let mut upsert = transaction.prepare_cached(
        "INSERT INTO images(
                source_id, path, size, modified_ns, blake3, status, error, width, height,
                ratio, orientation, common_ratio, dominant_hex, dominant_name, luminance,
                saturation, contrast, light_dark, thumbnail_path, analysis_key,
                discovered_at, updated_at, missing_since
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, 'ready', NULL, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?19, NULL
             ) ON CONFLICT(path) DO UPDATE SET
                source_id=excluded.source_id, size=excluded.size, modified_ns=excluded.modified_ns,
                blake3=excluded.blake3, status='ready', error=NULL, width=excluded.width,
                height=excluded.height, ratio=excluded.ratio, orientation=excluded.orientation,
                common_ratio=excluded.common_ratio, dominant_hex=excluded.dominant_hex,
                dominant_name=excluded.dominant_name, luminance=excluded.luminance,
                saturation=excluded.saturation, contrast=excluded.contrast,
                light_dark=excluded.light_dark, thumbnail_path=excluded.thumbnail_path,
                analysis_key=excluded.analysis_key, updated_at=excluded.updated_at,
                missing_since=NULL
             RETURNING id",
    )?;
    let image_id = upsert.query_row(
        params![
            discovery.source_id,
            path_bytes(&discovery.path),
            discovery.size as i64,
            discovery.modified_ns,
            hash,
            analysis.width,
            analysis.height,
            analysis.ratio,
            analysis.orientation.as_str(),
            analysis.common_ratio,
            analysis.dominant_hex,
            analysis.dominant_name,
            analysis.luminance,
            analysis.saturation,
            analysis.contrast,
            analysis.light_dark.as_str(),
            path_bytes(thumbnail),
            analysis_key,
            now,
        ],
        |row| row.get(0),
    )?;
    clear_derived_data(transaction, image_id)?;
    let mut insert_palette = transaction.prepare_cached(
        "INSERT INTO image_palette(
            image_id, rank, oklab_l, oklab_a, oklab_b, proportion, hex, name
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;
    for (rank, colour) in analysis.palette.iter().enumerate() {
        insert_palette.execute(params![
            image_id,
            rank as i64,
            colour.oklab.l,
            colour.oklab.a,
            colour.oklab.b,
            colour.proportion,
            colour.hex,
            colour.name,
        ])?;
    }
    Ok(())
}

fn store_out_of_bounds(
    transaction: &Transaction<'_>,
    discovery: &Discovery,
    dimensions: (u32, u32),
    analysis_key: &str,
) -> Result<()> {
    let now = Utc::now().timestamp_millis();
    let mut upsert = transaction.prepare_cached(
        "INSERT INTO images(
                source_id, path, size, modified_ns, status, width, height, analysis_key,
                discovered_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, 'out_of_bounds', ?5, ?6, ?7, ?8, ?8)
             ON CONFLICT(path) DO UPDATE SET
                source_id=excluded.source_id, size=excluded.size, modified_ns=excluded.modified_ns,
                blake3=NULL, status='out_of_bounds', error=NULL, width=excluded.width,
                height=excluded.height, ratio=NULL, orientation=NULL, common_ratio=NULL,
                dominant_hex=NULL, dominant_name=NULL, luminance=NULL, saturation=NULL,
                contrast=NULL, light_dark=NULL, thumbnail_path=NULL,
                analysis_key=excluded.analysis_key, updated_at=excluded.updated_at,
                missing_since=NULL
             RETURNING id",
    )?;
    let image_id = upsert.query_row(
        params![
            discovery.source_id,
            path_bytes(&discovery.path),
            discovery.size as i64,
            discovery.modified_ns,
            dimensions.0,
            dimensions.1,
            analysis_key,
            now,
        ],
        |row| row.get(0),
    )?;
    clear_derived_data(transaction, image_id)
}

fn store_status(
    transaction: &Transaction<'_>,
    discovery: &Discovery,
    status: ImageStatus,
    error: &str,
    analysis_key: &str,
) -> Result<()> {
    let now = Utc::now().timestamp_millis();
    let mut upsert = transaction.prepare_cached(
        "INSERT INTO images(
                source_id, path, size, modified_ns, status, error, analysis_key,
                discovered_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)
             ON CONFLICT(path) DO UPDATE SET
                source_id=excluded.source_id, size=excluded.size, modified_ns=excluded.modified_ns,
                status=excluded.status, error=excluded.error, analysis_key=excluded.analysis_key,
                blake3=NULL, width=NULL, height=NULL, ratio=NULL, orientation=NULL,
                common_ratio=NULL, dominant_hex=NULL, dominant_name=NULL, luminance=NULL,
                saturation=NULL, contrast=NULL, light_dark=NULL, thumbnail_path=NULL,
                updated_at=excluded.updated_at, missing_since=NULL
             RETURNING id",
    )?;
    let image_id = upsert.query_row(
        params![
            discovery.source_id,
            path_bytes(&discovery.path),
            discovery.size as i64,
            discovery.modified_ns,
            status.as_str(),
            error,
            analysis_key,
            now,
        ],
        |row| row.get(0),
    )?;
    clear_derived_data(transaction, image_id)
}

fn clear_derived_data(transaction: &Transaction<'_>, image_id: i64) -> Result<()> {
    transaction
        .prepare_cached("DELETE FROM image_palette WHERE image_id=?1")?
        .execute([image_id])?;
    transaction
        .prepare_cached("DELETE FROM embeddings WHERE image_id=?1")?
        .execute([image_id])?;
    transaction
        .prepare_cached("DELETE FROM label_scores WHERE image_id=?1")?
        .execute([image_id])?;
    Ok(())
}

fn touch_existing(transaction: &Transaction<'_>, id: i64, discovery: &Discovery) -> Result<()> {
    transaction
        .prepare_cached(
            "UPDATE images SET source_id=?1, size=?2, modified_ns=?3, updated_at=?4,
             missing_since=NULL WHERE id=?5",
        )?
        .execute(params![
            discovery.source_id,
            discovery.size as i64,
            discovery.modified_ns,
            Utc::now().timestamp_millis(),
            id,
        ])?;
    Ok(())
}

fn restore_ready(transaction: &Transaction<'_>, id: i64, discovery: &Discovery) -> Result<()> {
    let changed = transaction
        .prepare_cached(
            "UPDATE images SET source_id=?1, size=?2, modified_ns=?3, status='ready',
             error=NULL, missing_since=NULL, updated_at=?4 WHERE id=?5",
        )?
        .execute(params![
            discovery.source_id,
            discovery.size as i64,
            discovery.modified_ns,
            Utc::now().timestamp_millis(),
            id,
        ])?;
    anyhow::ensure!(
        changed == 1,
        "missing image disappeared while being restored"
    );
    Ok(())
}

fn mark_missing(
    database: &Database,
    sources: &[SourceRoot],
    unreliable_sources: &HashSet<i64>,
    existing: &HashMap<PathBuf, Existing>,
) -> Result<usize> {
    database.with_transaction(|transaction| {
        let now = Utc::now().timestamp_millis();
        let mut changed = 0;
        for (path, record) in existing {
            let owner = sources
                .iter()
                .filter(|source| path.starts_with(&source.path))
                .max_by_key(|source| source.path.components().count());
            if owner.is_some_and(|source| !unreliable_sources.contains(&source.id))
                && record.status != ImageStatus::Missing
            {
                transaction.execute(
                    "UPDATE images SET status='missing', missing_since=?1, updated_at=?1 WHERE id=?2",
                    params![now, record.id],
                )?;
                changed += 1;
            }
        }
        Ok(changed)
    })
}

fn analysis_key(config: &Config) -> Result<String> {
    let data = serde_json::to_vec(&(&config.import, &config.analysis))?;
    Ok(blake3::hash(&data).to_hex().to_string())
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "gif" | "bmp" | "tif" | "tiff"
            )
        })
}

fn record_failure(
    report: &mut ScanReport,
    progress: &mut impl FnMut(ScanEvent),
    failure: FileFailure,
) {
    report.failed += 1;
    progress(ScanEvent::Failed(failure.clone()));
    report.failures.push(failure);
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    struct Fixture {
        _directory: tempfile::TempDir,
        paths: AppPaths,
        source: PathBuf,
        database: Database,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = tempfile::tempdir().expect("tempdir");
            let root = directory.path();
            let paths = AppPaths::from_xdg_roots(
                root.join("config"),
                root.join("data"),
                root.join("cache"),
                root.join("state"),
            );
            paths.ensure_owned_dirs().expect("paths");
            let source = root.join("images");
            std::fs::create_dir(&source).expect("source");
            let database = Database::open(&paths.database).expect("database");
            database.add_source(&source).expect("add source");
            Self {
                _directory: directory,
                paths,
                source,
                database,
            }
        }

        fn image(&self, name: &str, width: u32, height: u32) -> PathBuf {
            let path = self.source.join(name);
            RgbImage::from_pixel(width, height, Rgb([40, 80, 160]))
                .save(&path)
                .expect("save image");
            path
        }
    }

    #[test]
    fn incremental_scan_skips_unchanged_images() {
        let fixture = Fixture::new();
        fixture.image("one.png", 64, 32);
        let first = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("first scan");
        assert_eq!(first.analyzed, 1);
        let changes_before = fixture
            .database
            .with_connection(|connection| Ok(connection.total_changes()))
            .expect("change count");
        let second = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("second scan");
        assert_eq!(second.unchanged, 1);
        assert_eq!(second.analyzed, 0);
        let changes_after = fixture
            .database
            .with_connection(|connection| Ok(connection.total_changes()))
            .expect("change count");
        assert_eq!(changes_after, changes_before);
    }

    #[test]
    fn incremental_scan_repairs_a_missing_thumbnail_without_reanalysis() {
        let fixture = Fixture::new();
        let path = fixture.image("thumbnail.png", 64, 32);
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("first scan");
        let image_id = fixture
            .database
            .image_id_by_path(&path)
            .expect("lookup")
            .expect("image id");
        let thumbnail = fixture
            .database
            .get_image(image_id)
            .expect("load")
            .expect("image")
            .thumbnail_path
            .expect("thumbnail path");
        std::fs::remove_file(&thumbnail).expect("remove thumbnail");

        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("repair scan");

        assert_eq!(report.unchanged, 1);
        assert_eq!(report.analyzed, 0);
        assert!(thumbnail.is_file());
    }

    #[test]
    fn scan_persists_more_than_one_write_batch() {
        const IMAGE_COUNT: usize = 65;

        let fixture = Fixture::new();
        for index in 0..IMAGE_COUNT {
            fixture.image(&format!("image-{index:03}.png"), 8, 4);
        }

        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan");
        assert_eq!(report.analyzed, IMAGE_COUNT);
        let ready = fixture
            .database
            .with_connection(|connection| {
                connection
                    .query_row(
                        "SELECT COUNT(*) FROM images WHERE status='ready'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .map_err(Into::into)
            })
            .expect("ready count");
        assert_eq!(ready, IMAGE_COUNT as i64);
    }

    #[test]
    fn metadata_only_change_reuses_matching_content_analysis() {
        let fixture = Fixture::new();
        let path = fixture.image("touched.png", 64, 32);
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("first scan");
        let id = fixture
            .database
            .image_id_by_path(&path)
            .expect("lookup")
            .expect("image id");
        let original = fixture.database.get_image(id).expect("get").expect("image");

        let modified = std::fs::metadata(&path)
            .expect("metadata")
            .modified()
            .expect("modified time");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open image")
            .set_modified(modified + std::time::Duration::from_secs(1))
            .expect("touch image");

        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("rescan");
        let updated = fixture.database.get_image(id).expect("get").expect("image");
        assert_eq!(report.unchanged, 1);
        assert_eq!(report.analyzed, 0);
        assert_eq!(updated.hash, original.hash);
        assert_ne!(updated.modified_ns, original.modified_ns);
    }

    #[test]
    fn full_scan_reanalyses_metadata_stable_images() {
        let fixture = Fixture::new();
        fixture.image("full.png", 64, 32);
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("first scan");

        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions {
                full: true,
                no_ai: false,
            },
        )
        .expect("full scan");
        assert_eq!(report.analyzed, 1);
        assert_eq!(report.unchanged, 0);
    }

    #[test]
    fn analysis_setting_change_invalidates_metadata_fast_path() {
        let fixture = Fixture::new();
        let path = fixture.image("settings.png", 64, 32);
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("first scan");
        let id = fixture
            .database
            .image_id_by_path(&path)
            .expect("lookup")
            .expect("image id");
        let original_hash = fixture
            .database
            .get_image(id)
            .expect("get")
            .expect("image")
            .hash;
        let mut config = Config::default();
        config.analysis.dark_threshold = 0.0;

        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &config,
            ScanOptions::default(),
        )
        .expect("rescan");
        let updated = fixture.database.get_image(id).expect("get").expect("image");
        assert_eq!(report.analyzed, 1);
        assert_eq!(report.unchanged, 0);
        assert_eq!(updated.hash, original_hash);
        assert_eq!(updated.light_dark.as_deref(), Some("light"));
    }

    #[test]
    fn retains_corrupt_and_missing_records() {
        let fixture = Fixture::new();
        let path = fixture.source.join("broken.jpg");
        std::fs::write(&path, b"not an image").expect("broken image");
        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan");
        assert_eq!(report.failed, 1);
        let id = fixture
            .database
            .image_id_by_path(&path)
            .expect("lookup")
            .expect("image id");
        assert_eq!(
            fixture
                .database
                .get_image(id)
                .expect("get")
                .expect("image")
                .status,
            ImageStatus::Corrupt
        );

        std::fs::remove_file(&path).expect("remove");
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("rescan");
        assert_eq!(
            fixture
                .database
                .get_image(id)
                .expect("get")
                .expect("image")
                .status,
            ImageStatus::Missing
        );
    }

    #[test]
    fn byte_identical_reappearing_image_reuses_retained_analysis() {
        let fixture = Fixture::new();
        let path = fixture.image("returns.png", 64, 32);
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("initial scan");
        let id = fixture
            .database
            .image_id_by_path(&path)
            .expect("lookup")
            .expect("id");
        let original = fixture
            .database
            .get_image(id)
            .expect("load")
            .expect("image");
        let held = fixture._directory.path().join("held.png");
        std::fs::rename(&path, &held).expect("hide image");
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("missing scan");
        assert_eq!(
            fixture
                .database
                .get_image(id)
                .expect("load")
                .expect("image")
                .status,
            ImageStatus::Missing
        );
        std::fs::rename(&held, &path).expect("restore image");

        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("reappearance scan");
        let restored = fixture
            .database
            .get_image(id)
            .expect("load")
            .expect("image");
        assert_eq!(report.unchanged, 1);
        assert_eq!(report.analyzed, 0);
        assert_eq!(restored.status, ImageStatus::Ready);
        assert_eq!(restored.hash, original.hash);
        assert_eq!(restored.palette, original.palette);
        assert_eq!(restored.thumbnail_path, original.thumbnail_path);
    }

    #[test]
    fn changed_corrupt_file_drops_stale_analysis() {
        let fixture = Fixture::new();
        let path = fixture.image("becomes-broken.png", 24, 12);
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("initial scan");
        let id = fixture
            .database
            .image_id_by_path(&path)
            .expect("lookup")
            .expect("id");
        assert!(
            !fixture
                .database
                .get_image(id)
                .expect("load")
                .expect("image")
                .palette
                .is_empty()
        );

        std::fs::write(&path, b"not a png anymore").expect("corrupt");
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("corrupt scan");
        let record = fixture
            .database
            .get_image(id)
            .expect("load")
            .expect("image");
        assert_eq!(record.status, ImageStatus::Corrupt);
        assert!(record.hash.is_none());
        assert!(record.palette.is_empty());
        assert!(record.thumbnail_path.is_none());
    }

    #[test]
    fn bounds_keep_only_lightweight_record() {
        let fixture = Fixture::new();
        let path = fixture.image("small.png", 20, 20);
        let mut config = Config::default();
        config.import.min_width = Some(100);
        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &config,
            ScanOptions::default(),
        )
        .expect("scan");
        assert_eq!(report.out_of_bounds, 1);
        let id = fixture
            .database
            .image_id_by_path(&path)
            .expect("lookup")
            .expect("image id");
        let record = fixture.database.get_image(id).expect("get").expect("image");
        assert_eq!(record.status, ImageStatus::OutOfBounds);
        assert!(record.hash.is_none());
        assert!(record.thumbnail_path.is_none());

        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("rescan after bounds change");
        assert_eq!(report.analyzed, 1);
        let record = fixture.database.get_image(id).expect("get").expect("image");
        assert_eq!(record.status, ImageStatus::Ready);
        assert!(record.hash.is_some());
        assert_eq!(record.palette.len(), 1);
    }

    #[test]
    fn overlapping_roots_assign_the_most_specific_source() {
        let directory = tempfile::tempdir().expect("tempdir");
        let root = directory.path();
        let paths = AppPaths::from_xdg_roots(
            root.join("config"),
            root.join("data"),
            root.join("cache"),
            root.join("state"),
        );
        paths.ensure_owned_dirs().expect("paths");
        let outer = root.join("images");
        let inner = outer.join("featured");
        std::fs::create_dir_all(&inner).expect("roots");
        let image = inner.join("wall.png");
        RgbImage::from_pixel(12, 8, Rgb([10, 20, 30]))
            .save(&image)
            .expect("image");
        let database = Database::open(&paths.database).expect("database");
        database.add_source(&outer).expect("outer");
        let inner_source = database.add_source(&inner).expect("inner");
        scan_catalog(
            &database,
            &paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan");
        let id = database
            .image_id_by_path(&image)
            .expect("lookup")
            .expect("id");
        assert_eq!(
            database
                .get_image(id)
                .expect("load")
                .expect("image")
                .source_id,
            inner_source.id
        );
    }

    #[test]
    fn unavailable_nested_root_does_not_mark_outer_owned_rows_missing() {
        let directory = tempfile::tempdir().expect("tempdir");
        let root = directory.path();
        let paths = AppPaths::from_xdg_roots(
            root.join("config"),
            root.join("data"),
            root.join("cache"),
            root.join("state"),
        );
        paths.ensure_owned_dirs().expect("paths");
        let outer = root.join("images");
        let inner = outer.join("featured");
        std::fs::create_dir_all(&inner).expect("roots");
        let image = inner.join("wall.png");
        RgbImage::from_pixel(12, 8, Rgb([10, 20, 30]))
            .save(&image)
            .expect("image");
        let database = Database::open(&paths.database).expect("database");
        database.add_source(&outer).expect("outer");
        scan_catalog(
            &database,
            &paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("initial scan");
        let id = database
            .image_id_by_path(&image)
            .expect("lookup")
            .expect("id");
        database.add_source(&inner).expect("inner");

        let unavailable = root.join("temporarily-unavailable-featured");
        std::fs::rename(&inner, &unavailable).expect("hide inner root");
        let report = scan_catalog(
            &database,
            &paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan unavailable inner root");
        assert!(report.failed >= 1);
        assert_eq!(report.missing, 0);
        assert_eq!(
            database.get_image(id).expect("load").expect("image").status,
            ImageStatus::Ready
        );
        std::fs::rename(unavailable, &inner).expect("restore inner root");
    }

    #[test]
    fn traversal_failure_does_not_falsely_mark_every_image_missing() {
        let fixture = Fixture::new();
        let path = fixture.image("stays-ready.png", 32, 16);
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("initial scan");
        let id = fixture
            .database
            .image_id_by_path(&path)
            .expect("lookup")
            .expect("id");
        let unavailable = fixture.source.with_extension("temporarily-unavailable");
        std::fs::rename(&fixture.source, &unavailable).expect("hide source");
        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan unavailable source");
        assert!(report.failed >= 1);
        assert_eq!(report.missing, 0);
        assert_eq!(
            fixture
                .database
                .get_image(id)
                .expect("load")
                .expect("image")
                .status,
            ImageStatus::Ready
        );
        std::fs::rename(unavailable, &fixture.source).expect("restore source");
    }

    #[cfg(unix)]
    #[test]
    fn does_not_follow_symlinks() {
        use std::os::unix::fs::symlink;

        let fixture = Fixture::new();
        let outside = fixture.source.parent().expect("parent").join("outside");
        std::fs::create_dir(&outside).expect("outside");
        RgbImage::from_pixel(10, 10, Rgb([0, 0, 0]))
            .save(outside.join("hidden.png"))
            .expect("save");
        symlink(&outside, fixture.source.join("linked")).expect("symlink");
        let report = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan");
        assert_eq!(report.discovered, 0);
    }
}
