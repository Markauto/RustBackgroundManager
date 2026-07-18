use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use walkdir::WalkDir;

use crate::{
    AppPaths,
    analysis::{
        ImageAnalysis, analyze_image, probe_dimensions, within_import_bounds, write_thumbnail,
    },
    config::Config,
    db::{Database, ImageStatus, SourceRoot, path_bytes, path_from_bytes},
};

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
    size: u64,
    modified_ns: i64,
    hash: Option<String>,
    status: ImageStatus,
    analysis_key: Option<String>,
}

#[derive(Clone, Debug)]
enum ProcessOutcome {
    Complete,
    RecordedFailure(String),
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
    progress(ScanEvent::Started {
        files: discoveries.len(),
    });
    let analysis_key = analysis_key(config)?;
    let mut report = ScanReport {
        discovered: discoveries.len(),
        ..ScanReport::default()
    };
    for failure in traversal_failures {
        record_failure(&mut report, &mut progress, failure);
    }

    let mut seen = HashSet::with_capacity(discoveries.len());
    for (index, discovery) in discoveries.into_iter().enumerate() {
        seen.insert(discovery.path.clone());
        progress(ScanEvent::Processing {
            index: index + 1,
            path: discovery.path.clone(),
        });
        match process_file(
            database,
            paths,
            config,
            options,
            &analysis_key,
            &discovery,
            &mut report,
        ) {
            Ok(ProcessOutcome::Complete) => {}
            Ok(ProcessOutcome::RecordedFailure(error)) => record_failure(
                &mut report,
                &mut progress,
                FileFailure {
                    path: discovery.path,
                    error,
                },
            ),
            Err(error) => {
                let message = format!("{error:#}");
                let _ = store_failure(database, &discovery, &message);
                record_failure(
                    &mut report,
                    &mut progress,
                    FileFailure {
                        path: discovery.path,
                        error: message,
                    },
                );
            }
        }
    }
    report.missing = mark_missing(database, &seen, &sources, &unreliable_sources)?;
    progress(ScanEvent::Finished(report.clone()));
    Ok(report)
}

fn discover(sources: &[SourceRoot]) -> (Vec<Discovery>, Vec<FileFailure>, HashSet<i64>) {
    let mut by_path: HashMap<PathBuf, Discovery> = HashMap::new();
    let mut failures = Vec::new();
    let mut unreliable_sources = HashSet::new();
    // More specific roots win ownership when roots overlap.
    let mut roots = sources.to_vec();
    roots.sort_by_key(|source| source.path.components().count());
    for source in roots {
        for entry in WalkDir::new(&source.path).follow_links(false) {
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
    database: &Database,
    paths: &AppPaths,
    config: &Config,
    options: ScanOptions,
    analysis_key: &str,
    discovery: &Discovery,
    report: &mut ScanReport,
) -> Result<ProcessOutcome> {
    let existing = existing(database, &discovery.path)?;
    let dimensions = match probe_dimensions(&discovery.path) {
        Ok(dimensions) => dimensions,
        Err(error) => {
            let message = format!("{error:#}");
            store_corrupt(database, discovery, &message, analysis_key)?;
            return Ok(ProcessOutcome::RecordedFailure(message));
        }
    };
    if !within_import_bounds(dimensions.0, dimensions.1, &config.import) {
        store_out_of_bounds(database, discovery, dimensions, analysis_key)?;
        report.out_of_bounds += 1;
        return Ok(ProcessOutcome::Complete);
    }

    let hash = hash_file(&discovery.path)?;
    if let Some(record) = existing.as_ref()
        && !options.full
        && record.size == discovery.size
        && record.modified_ns == discovery.modified_ns
        && record.hash.as_deref() == Some(&hash)
        && record.status == ImageStatus::Ready
        && record.analysis_key.as_deref() == Some(analysis_key)
    {
        touch_existing(database, record.id, discovery)?;
        report.unchanged += 1;
        if options.no_ai {
            report.ai_deferred += 1;
        }
        return Ok(ProcessOutcome::Complete);
    }

    let (analysis, decoded) = analyze_image(&discovery.path, &config.analysis)?;
    let thumbnail = paths.thumbnails_dir.join(format!("{hash}.jpg"));
    if options.full || !thumbnail.exists() {
        write_thumbnail(&decoded, &thumbnail, config.analysis.thumbnail_long_edge)?;
    }
    store_analysis(
        database,
        discovery,
        &hash,
        analysis_key,
        &thumbnail,
        &analysis,
    )?;
    report.analyzed += 1;
    if options.no_ai {
        report.ai_deferred += 1;
    }
    Ok(ProcessOutcome::Complete)
}

fn existing(database: &Database, path: &Path) -> Result<Option<Existing>> {
    database.with_connection(|connection| {
        let row = connection
            .query_row(
                "SELECT id, size, modified_ns, blake3, status, analysis_key
                 FROM images WHERE path = ?1",
                [path_bytes(path)],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get::<_, i64>(1)? as u64,
                        row.get(2)?,
                        row.get(3)?,
                        row.get::<_, String>(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()?;
        row.map(|(id, size, modified_ns, hash, status, analysis_key)| {
            Ok(Existing {
                id,
                size,
                modified_ns,
                hash,
                status: ImageStatus::parse(&status)?,
                analysis_key,
            })
        })
        .transpose()
    })
}

fn store_analysis(
    database: &Database,
    discovery: &Discovery,
    hash: &str,
    analysis_key: &str,
    thumbnail: &Path,
    analysis: &ImageAnalysis,
) -> Result<()> {
    let now = Utc::now().timestamp_millis();
    database.with_transaction(|transaction| {
        transaction.execute(
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
                missing_since=NULL",
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
        )?;
        let image_id: i64 = transaction.query_row(
            "SELECT id FROM images WHERE path = ?1",
            [path_bytes(&discovery.path)],
            |row| row.get(0),
        )?;
        transaction.execute("DELETE FROM image_palette WHERE image_id = ?1", [image_id])?;
        for (rank, colour) in analysis.palette.iter().enumerate() {
            transaction.execute(
                "INSERT INTO image_palette(
                    image_id, rank, oklab_l, oklab_a, oklab_b, proportion, hex, name
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    image_id,
                    rank as i64,
                    colour.oklab.l,
                    colour.oklab.a,
                    colour.oklab.b,
                    colour.proportion,
                    colour.hex,
                    colour.name,
                ],
            )?;
        }
        // A changed image invalidates its prior embedding and derived scores.
        transaction.execute("DELETE FROM embeddings WHERE image_id = ?1", [image_id])?;
        transaction.execute("DELETE FROM label_scores WHERE image_id = ?1", [image_id])?;
        Ok(())
    })
}

fn store_out_of_bounds(
    database: &Database,
    discovery: &Discovery,
    dimensions: (u32, u32),
    analysis_key: &str,
) -> Result<()> {
    let now = Utc::now().timestamp_millis();
    database.with_transaction(|transaction| {
        transaction.execute(
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
                missing_since=NULL",
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
        )?;
        let image_id: i64 = transaction.query_row(
            "SELECT id FROM images WHERE path=?1",
            [path_bytes(&discovery.path)],
            |row| row.get(0),
        )?;
        transaction.execute("DELETE FROM image_palette WHERE image_id=?1", [image_id])?;
        transaction.execute("DELETE FROM embeddings WHERE image_id=?1", [image_id])?;
        transaction.execute("DELETE FROM label_scores WHERE image_id=?1", [image_id])?;
        Ok(())
    })
}

fn store_corrupt(
    database: &Database,
    discovery: &Discovery,
    error: &str,
    analysis_key: &str,
) -> Result<()> {
    store_status(
        database,
        discovery,
        ImageStatus::Corrupt,
        error,
        analysis_key,
    )
}

fn store_failure(database: &Database, discovery: &Discovery, error: &str) -> Result<()> {
    store_status(database, discovery, ImageStatus::Error, error, "")
}

fn store_status(
    database: &Database,
    discovery: &Discovery,
    status: ImageStatus,
    error: &str,
    analysis_key: &str,
) -> Result<()> {
    let now = Utc::now().timestamp_millis();
    database.with_transaction(|transaction| {
        transaction.execute(
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
                updated_at=excluded.updated_at, missing_since=NULL",
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
        )?;
        let image_id: i64 = transaction.query_row(
            "SELECT id FROM images WHERE path=?1",
            [path_bytes(&discovery.path)],
            |row| row.get(0),
        )?;
        transaction.execute("DELETE FROM image_palette WHERE image_id=?1", [image_id])?;
        transaction.execute("DELETE FROM embeddings WHERE image_id=?1", [image_id])?;
        transaction.execute("DELETE FROM label_scores WHERE image_id=?1", [image_id])?;
        Ok(())
    })
}

fn touch_existing(database: &Database, id: i64, discovery: &Discovery) -> Result<()> {
    database.with_connection(|connection| {
        connection.execute(
            "UPDATE images SET source_id=?1, size=?2, modified_ns=?3, updated_at=?4,
             missing_since=NULL WHERE id=?5",
            params![
                discovery.source_id,
                discovery.size as i64,
                discovery.modified_ns,
                Utc::now().timestamp_millis(),
                id,
            ],
        )?;
        Ok(())
    })
}

fn mark_missing(
    database: &Database,
    seen: &HashSet<PathBuf>,
    sources: &[SourceRoot],
    unreliable_sources: &HashSet<i64>,
) -> Result<usize> {
    database.with_transaction(|transaction| {
        let candidates = {
            let mut statement =
                transaction.prepare("SELECT id, path, status, source_id FROM images")?;
            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    path_from_bytes(row.get_ref(1)?.as_blob()?),
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        let now = Utc::now().timestamp_millis();
        let mut changed = 0;
        for (id, path, status, source_id) in candidates {
            let belongs_to_source = sources
                .iter()
                .any(|source| path.starts_with(&source.path));
            if belongs_to_source
                && !unreliable_sources.contains(&source_id)
                && !seen.contains(&path)
                && status != ImageStatus::Missing.as_str()
            {
                transaction.execute(
                    "UPDATE images SET status='missing', missing_since=?1, updated_at=?1 WHERE id=?2",
                    params![now, id],
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

fn hash_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut reader, &mut hasher)?;
    Ok(hasher.finalize().to_hex().to_string())
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
        let second = scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("second scan");
        assert_eq!(second.unchanged, 1);
        assert_eq!(second.analyzed, 0);
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
