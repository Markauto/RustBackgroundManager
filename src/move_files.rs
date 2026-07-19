use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufReader, BufWriter, Write as _},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use rusqlite::params;
use rustix::fs::{CWD, RenameFlags, renameat_with};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use uuid::Uuid;

use crate::{
    AppPaths,
    db::{Database, ImageRecord, ImageStatus, path_bytes, path_from_bytes},
    filesystem::{absolute_lexical, blake3_file},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MovePlan {
    pub id: Uuid,
    pub destination_root: PathBuf,
    pub items: Vec<MovePlanItem>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MovePlanItem {
    pub image_id: i64,
    pub original_path: PathBuf,
    pub destination: PathBuf,
    pub hash: String,
    pub status: MoveItemStatus,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveItemStatus {
    Planned,
    Moved,
    Undone,
    Failed,
}

impl MoveItemStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::Moved => "moved",
            Self::Undone => "undone",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MoveResult {
    pub id: Uuid,
    pub moved: usize,
    pub status: String,
    pub manifest: PathBuf,
}

pub fn plan_move(images: &[ImageRecord], destination_root: &Path) -> Result<MovePlan> {
    if images.is_empty() {
        bail!("move selection is empty");
    }
    let destination_root = match destination_root.canonicalize() {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            absolute_lexical(destination_root)?
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "cannot resolve move destination {}",
                    destination_root.display()
                )
            });
        }
    };
    if optional_metadata(&destination_root)?.is_some_and(|metadata| !metadata.file_type().is_dir())
    {
        bail!(
            "move destination is not a directory: {}",
            destination_root.display()
        );
    }
    let mut targets = HashSet::with_capacity(images.len());
    let mut items = Vec::with_capacity(images.len());
    for image in images {
        if image.status != ImageStatus::Ready {
            bail!(
                "image {} is not available: {}",
                image.id,
                image.path.display()
            );
        }
        let filename = image
            .path
            .file_name()
            .with_context(|| format!("image path has no filename: {}", image.path.display()))?;
        let destination = destination_root.join(filename);
        if destination == image.path {
            bail!(
                "source and destination are identical: {}",
                image.path.display()
            );
        }
        if !targets.insert(destination.clone()) {
            bail!("multiple selected files target {}", destination.display());
        }
        if optional_metadata(&destination)?.is_some() {
            bail!("destination already exists: {}", destination.display());
        }
        let hash = image
            .hash
            .clone()
            .with_context(|| format!("image {} has not been hashed; run bgm scan", image.id))?;
        items.push(MovePlanItem {
            image_id: image.id,
            original_path: image.path.clone(),
            destination,
            hash,
            status: MoveItemStatus::Planned,
            error: None,
        });
    }
    Ok(MovePlan {
        id: Uuid::new_v4(),
        destination_root,
        items,
    })
}

pub fn apply_move(database: &Database, paths: &AppPaths, mut plan: MovePlan) -> Result<MoveResult> {
    fs::create_dir_all(&plan.destination_root).with_context(|| {
        format!(
            "failed to create move destination {}",
            plan.destination_root.display()
        )
    })?;
    let live_destination_root = plan.destination_root.canonicalize().with_context(|| {
        format!(
            "cannot resolve move destination {}",
            plan.destination_root.display()
        )
    })?;
    anyhow::ensure!(
        live_destination_root == plan.destination_root,
        "move destination changed since planning: expected {}, found {}",
        plan.destination_root.display(),
        live_destination_root.display()
    );
    validate_plan(&plan)?;
    insert_operation(database, &plan)?;
    let manifest = manifest_path(paths, plan.id);
    write_manifest(&manifest, &plan)?;

    let mut moved = 0;
    for index in 0..plan.items.len() {
        let result = move_one(
            &plan.items[index].original_path,
            &plan.items[index].destination,
            &plan.items[index].hash,
        );
        match result {
            Ok(()) => {
                plan.items[index].status = MoveItemStatus::Moved;
                moved += 1;
                let manifest_result = write_manifest(&manifest, &plan);
                let catalog_result =
                    update_moved_item(database, plan.id, index, &plan.items[index]);
                if let Some(error) = persistence_error(manifest_result, catalog_result) {
                    let _ = update_item_status(
                        database,
                        plan.id,
                        index,
                        MoveItemStatus::Moved,
                        Some(&error),
                    );
                    let _ = finish_operation(database, plan.id, "partial", Some(&error));
                    bail!(
                        "move {} stopped after {moved} file(s): the file moved but recovery state was incomplete: {error}; manifest: {}",
                        plan.id,
                        manifest.display()
                    );
                }
            }
            Err(error) => {
                let message = format!("{error:#}");
                plan.items[index].status = MoveItemStatus::Failed;
                plan.items[index].error = Some(message.clone());
                let manifest_result = write_manifest(&manifest, &plan);
                let item_result = update_item_status(
                    database,
                    plan.id,
                    index,
                    MoveItemStatus::Failed,
                    Some(&message),
                );
                let finish_result = finish_operation(database, plan.id, "partial", Some(&message));
                let persistence = persistence_error(manifest_result, item_result)
                    .into_iter()
                    .chain(
                        finish_result
                            .err()
                            .map(|error| format!("operation: {error:#}")),
                    )
                    .collect::<Vec<_>>();
                let persistence = if persistence.is_empty() {
                    String::new()
                } else {
                    format!("; recovery-state errors: {}", persistence.join("; "))
                };
                bail!(
                    "move {} stopped after {moved} file(s): {message}{persistence}; partial manifest: {}",
                    plan.id,
                    manifest.display()
                );
            }
        }
    }
    finish_operation(database, plan.id, "completed", None)?;
    Ok(MoveResult {
        id: plan.id,
        moved,
        status: "completed".into(),
        manifest,
    })
}

pub fn undo_move(database: &Database, paths: &AppPaths, operation_id: Uuid) -> Result<MoveResult> {
    let mut plan = load_operation(database, operation_id)?;
    let candidate_indices: Vec<_> = plan
        .items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            matches!(item.status, MoveItemStatus::Moved | MoveItemStatus::Undone).then_some(index)
        })
        .collect();
    if candidate_indices.is_empty() {
        bail!("move {operation_id} has no files that can be undone");
    }
    let candidates = candidate_indices
        .into_iter()
        .map(|index| undo_location(&plan.items[index]).map(|location| (index, location)))
        .collect::<Result<Vec<_>>>()?;

    let manifest = manifest_path(paths, operation_id);
    let mut undone = 0;
    for (index, location) in candidates.into_iter().rev() {
        if location == UndoLocation::Destination {
            if let Some(parent) = plan.items[index].original_path.parent() {
                fs::create_dir_all(parent)?;
            }
            move_one(
                &plan.items[index].destination,
                &plan.items[index].original_path,
                &plan.items[index].hash,
            )?;
        }
        plan.items[index].status = MoveItemStatus::Undone;
        undone += 1;
        let manifest_result = write_manifest(&manifest, &plan);
        let catalog_result = update_undone_item(database, operation_id, index, &plan.items[index]);
        if let Some(error) = persistence_error(manifest_result, catalog_result) {
            let _ = update_item_status(
                database,
                operation_id,
                index,
                MoveItemStatus::Undone,
                Some(&error),
            );
            let _ = finish_operation(database, operation_id, "partial", Some(&error));
            bail!(
                "undo {operation_id} stopped after {undone} file(s): the file was restored but recovery state was incomplete: {error}; manifest: {}",
                manifest.display()
            );
        }
    }
    database.with_connection(|connection| {
        let changed = connection.execute(
            "UPDATE move_operations SET status='undone', undone_at=?1, error=NULL WHERE id=?2",
            params![Utc::now().timestamp_millis(), operation_id.to_string()],
        )?;
        anyhow::ensure!(
            changed == 1,
            "move operation disappeared while finishing undo"
        );
        Ok(())
    })?;
    Ok(MoveResult {
        id: operation_id,
        moved: undone,
        status: "undone".into(),
        manifest,
    })
}

fn validate_plan(plan: &MovePlan) -> Result<()> {
    anyhow::ensure!(
        plan.destination_root.is_absolute(),
        "move destination root must be absolute"
    );
    let mut destinations = HashSet::with_capacity(plan.items.len());
    for item in &plan.items {
        anyhow::ensure!(
            item.destination.parent() == Some(plan.destination_root.as_path()),
            "move destination is outside the planned root: {}",
            item.destination.display()
        );
        if !destinations.insert(&item.destination) {
            bail!("duplicate destination: {}", item.destination.display());
        }
        if optional_metadata(&item.destination)?.is_some() {
            bail!("destination already exists: {}", item.destination.display());
        }
        let metadata = fs::symlink_metadata(&item.original_path)
            .with_context(|| format!("source is unavailable: {}", item.original_path.display()))?;
        if !metadata.file_type().is_file() {
            bail!(
                "source is not a regular file: {}",
                item.original_path.display()
            );
        }
        let current = blake3_file(&item.original_path)?;
        if current != item.hash {
            bail!(
                "source changed since scan: {}",
                item.original_path.display()
            );
        }
    }
    Ok(())
}

fn persistence_error(manifest: Result<()>, catalog: Result<()>) -> Option<String> {
    let mut failures = Vec::new();
    if let Err(error) = manifest {
        failures.push(format!("manifest: {error:#}"));
    }
    if let Err(error) = catalog {
        failures.push(format!("catalog: {error:#}"));
    }
    (!failures.is_empty()).then(|| failures.join("; "))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UndoLocation {
    Destination,
    Original,
}

fn undo_location(item: &MovePlanItem) -> Result<UndoLocation> {
    let original = optional_metadata(&item.original_path)?;
    let destination = optional_metadata(&item.destination)?;
    let (location, path, description) = match (original, destination) {
        (None, Some(metadata)) => (
            UndoLocation::Destination,
            &item.destination,
            (metadata, "destination"),
        ),
        (Some(metadata), None) => (
            UndoLocation::Original,
            &item.original_path,
            (metadata, "original"),
        ),
        (Some(_), Some(_)) => {
            bail!(
                "cannot undo: original and destination both exist: {} and {}",
                item.original_path.display(),
                item.destination.display()
            );
        }
        (None, None) => {
            bail!(
                "cannot undo: original and destination are unavailable: {} and {}",
                item.original_path.display(),
                item.destination.display()
            );
        }
    };
    let (metadata, description) = description;
    if !metadata.file_type().is_file() {
        bail!(
            "cannot undo: {description} is not a regular file: {}",
            path.display()
        );
    }
    let current = blake3_file(path).with_context(|| {
        format!(
            "cannot undo: {description} is unavailable: {}",
            path.display()
        )
    })?;
    if current != item.hash {
        bail!(
            "cannot undo: {description} hash changed: {}",
            path.display()
        );
    }
    Ok(location)
}

fn optional_metadata(path: &Path) -> Result<Option<fs::Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("cannot inspect {}", path.display())),
    }
}

fn move_one(source: &Path, destination: &Path, expected_hash: &str) -> Result<()> {
    let destination_parent = destination.parent().context("destination has no parent")?;
    fs::create_dir_all(destination_parent)?;
    match renameat_with(CWD, source, CWD, destination, RenameFlags::NOREPLACE) {
        Ok(()) => {
            sync_parent(source)?;
            if source.parent() != destination.parent() {
                sync_parent(destination)?;
            }
            Ok(())
        }
        Err(error) if error == rustix::io::Errno::XDEV => {
            copy_across_filesystems(source, destination, expected_hash)
        }
        Err(error) => {
            Err(std::io::Error::from_raw_os_error(error.raw_os_error())).with_context(|| {
                format!(
                    "failed to move {} to {}",
                    source.display(),
                    destination.display()
                )
            })
        }
    }
}

fn copy_across_filesystems(source: &Path, destination: &Path, expected_hash: &str) -> Result<()> {
    copy_across_filesystems_with_remove(source, destination, expected_hash, |path| {
        fs::remove_file(path)
    })
}

fn copy_across_filesystems_with_remove(
    source: &Path,
    destination: &Path,
    expected_hash: &str,
    remove_source: impl FnOnce(&Path) -> std::io::Result<()>,
) -> Result<()> {
    let parent = destination.parent().context("destination has no parent")?;
    let source_file = File::open(source)?;
    let permissions = source_file.metadata()?.permissions();
    let mut reader = BufReader::new(source_file);
    let mut temporary = NamedTempFile::new_in(parent)?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        std::io::copy(&mut reader, &mut writer)?;
        writer.flush()?;
    }
    temporary.as_file().set_permissions(permissions)?;
    temporary.as_file().sync_all()?;
    let copied_hash = blake3_file(temporary.path())?;
    if copied_hash != expected_hash {
        bail!("copied file failed hash verification");
    }
    temporary
        .persist_noclobber(destination)
        .map_err(|error| error.error)
        .with_context(|| {
            format!(
                "destination appeared during move: {}",
                destination.display()
            )
        })?;
    sync_parent(destination)?;
    if let Err(error) = remove_source(source) {
        if let Err(rollback_error) = fs::remove_file(destination) {
            return Err(error).context(format!(
                "could not delete source or roll back destination (rollback: {rollback_error})"
            ));
        }
        if let Err(sync_error) = sync_parent(destination) {
            return Err(error).context(format!(
                "could not delete source; copied destination was removed but its directory could not be synced: {sync_error:#}"
            ));
        }
        return Err(error).context("could not delete source; copied destination was rolled back");
    }
    sync_parent(source)?;
    Ok(())
}

fn insert_operation(database: &Database, plan: &MovePlan) -> Result<()> {
    database.with_transaction(|transaction| {
        let operation_id = plan.id.to_string();
        transaction.execute(
            "INSERT INTO move_operations(id, status, destination_root, created_at)
             VALUES (?1, 'in_progress', ?2, ?3)",
            params![
                operation_id,
                path_bytes(&plan.destination_root),
                Utc::now().timestamp_millis(),
            ],
        )?;
        let mut insert_item = transaction.prepare(
            "INSERT INTO move_items(
                operation_id, ordinal, image_id, original_path, destination, blake3, status
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for (ordinal, item) in plan.items.iter().enumerate() {
            insert_item.execute(params![
                operation_id,
                ordinal as i64,
                item.image_id,
                path_bytes(&item.original_path),
                path_bytes(&item.destination),
                item.hash,
                item.status.as_str(),
            ])?;
        }
        Ok(())
    })
}

fn update_moved_item(
    database: &Database,
    operation_id: Uuid,
    ordinal: usize,
    item: &MovePlanItem,
) -> Result<()> {
    database.with_transaction(|transaction| {
        let item_changed = transaction.execute(
            "UPDATE move_items SET status='moved', error=NULL
             WHERE operation_id=?1 AND ordinal=?2",
            params![operation_id.to_string(), ordinal as i64],
        )?;
        anyhow::ensure!(
            item_changed == 1,
            "move item disappeared while recording move"
        );
        let image_changed = transaction.execute(
            "UPDATE images SET path=?1, size=?2, modified_ns=?3, updated_at=?4
             WHERE id=?5 AND path IN (?6, ?7)",
            params![
                path_bytes(&item.destination),
                fs::metadata(&item.destination)?.len() as i64,
                modified_ns(&item.destination)?,
                Utc::now().timestamp_millis(),
                item.image_id,
                path_bytes(&item.original_path),
                path_bytes(&item.destination),
            ],
        )?;
        anyhow::ensure!(
            image_changed == 1,
            "catalog image disappeared or changed path while recording move"
        );
        Ok(())
    })
}

fn update_undone_item(
    database: &Database,
    operation_id: Uuid,
    ordinal: usize,
    item: &MovePlanItem,
) -> Result<()> {
    database.with_transaction(|transaction| {
        let item_changed = transaction.execute(
            "UPDATE move_items SET status='undone', error=NULL
             WHERE operation_id=?1 AND ordinal=?2",
            params![operation_id.to_string(), ordinal as i64],
        )?;
        anyhow::ensure!(
            item_changed == 1,
            "move item disappeared while recording undo"
        );
        let image_changed = transaction.execute(
            "UPDATE images SET path=?1, size=?2, modified_ns=?3, updated_at=?4
             WHERE id=?5 AND path IN (?6, ?7)",
            params![
                path_bytes(&item.original_path),
                fs::metadata(&item.original_path)?.len() as i64,
                modified_ns(&item.original_path)?,
                Utc::now().timestamp_millis(),
                item.image_id,
                path_bytes(&item.destination),
                path_bytes(&item.original_path),
            ],
        )?;
        anyhow::ensure!(
            image_changed == 1,
            "catalog image disappeared or changed path while recording undo"
        );
        Ok(())
    })
}

fn update_item_status(
    database: &Database,
    operation_id: Uuid,
    ordinal: usize,
    status: MoveItemStatus,
    error: Option<&str>,
) -> Result<()> {
    database.with_connection(|connection| {
        let changed = connection.execute(
            "UPDATE move_items SET status=?1, error=?2 WHERE operation_id=?3 AND ordinal=?4",
            params![
                status.as_str(),
                error,
                operation_id.to_string(),
                ordinal as i64
            ],
        )?;
        anyhow::ensure!(changed == 1, "move item disappeared while updating status");
        Ok(())
    })
}

fn finish_operation(
    database: &Database,
    operation_id: Uuid,
    status: &str,
    error: Option<&str>,
) -> Result<()> {
    database.with_connection(|connection| {
        let changed = connection.execute(
            "UPDATE move_operations SET status=?1, completed_at=?2, error=?3 WHERE id=?4",
            params![
                status,
                Utc::now().timestamp_millis(),
                error,
                operation_id.to_string(),
            ],
        )?;
        anyhow::ensure!(
            changed == 1,
            "move operation disappeared while updating status"
        );
        Ok(())
    })
}

fn load_operation(database: &Database, operation_id: Uuid) -> Result<MovePlan> {
    database.with_connection(|connection| {
        let destination_root: Vec<u8> = connection
            .query_row(
                "SELECT destination_root FROM move_operations
                 WHERE id=?1 AND status IN ('completed', 'partial')",
                [operation_id.to_string()],
                |row| row.get(0),
            )
            .with_context(|| format!("move {operation_id} was not found or cannot be undone"))?;
        let mut statement = connection.prepare(
            "SELECT image_id, original_path, destination, blake3, status, error
             FROM move_items WHERE operation_id=?1 ORDER BY ordinal",
        )?;
        let rows = statement.query_map([operation_id.to_string()], |row| {
            let status: String = row.get(4)?;
            let status = match status.as_str() {
                "planned" => MoveItemStatus::Planned,
                "moved" => MoveItemStatus::Moved,
                "undone" => MoveItemStatus::Undone,
                "failed" => MoveItemStatus::Failed,
                _ => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Text,
                        format!("unknown move item status {status}").into(),
                    ));
                }
            };
            Ok(MovePlanItem {
                image_id: row.get(0)?,
                original_path: path_from_bytes(row.get_ref(1)?.as_blob()?),
                destination: path_from_bytes(row.get_ref(2)?.as_blob()?),
                hash: row.get(3)?,
                status,
                error: row.get(5)?,
            })
        })?;
        Ok(MovePlan {
            id: operation_id,
            destination_root: path_from_bytes(&destination_root),
            items: rows.collect::<rusqlite::Result<Vec<_>>>()?,
        })
    })
}

fn manifest_path(paths: &AppPaths, id: Uuid) -> PathBuf {
    paths.manifests_dir.join(format!("{id}.json"))
}

fn write_manifest(path: &Path, plan: &MovePlan) -> Result<()> {
    let parent = path.parent().context("manifest has no parent")?;
    fs::create_dir_all(parent)?;
    let mut temporary = NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temporary, plan)?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    sync_parent(path)?;
    Ok(())
}

fn modified_ns(path: &Path) -> Result<i64> {
    let modified = fs::metadata(path)?.modified()?;
    Ok(modified
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX)
        }))
}

fn sync_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use crate::{
        config::Config,
        scan::{ScanOptions, scan_catalog},
    };

    use super::*;

    #[test]
    fn move_is_dry_run_until_applied_and_undo_is_identical() {
        let directory = tempfile::tempdir().expect("tempdir");
        let paths = AppPaths::from_xdg_roots(
            directory.path().join("cfg"),
            directory.path().join("data"),
            directory.path().join("cache"),
            directory.path().join("state"),
        );
        paths.ensure_owned_dirs().expect("dirs");
        let source = directory.path().join("source");
        let destination = directory.path().join("destination");
        fs::create_dir(&source).expect("source");
        fs::create_dir(&destination).expect("destination");
        let original = source.join("wallpaper.png");
        RgbImage::from_pixel(16, 8, Rgb([20, 40, 60]))
            .save(&original)
            .expect("image");
        let original_bytes = fs::read(&original).expect("bytes");
        let database = Database::open(&paths.database).expect("database");
        database.add_source(&source).expect("source root");
        scan_catalog(
            &database,
            &paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan");
        let id = database
            .image_id_by_path(&original)
            .expect("lookup")
            .expect("id");
        let image = database.get_image(id).expect("load").expect("image");
        let plan = plan_move(&[image], &destination).expect("plan");
        assert!(original.exists(), "planning must be a dry run");
        assert!(!destination.join("wallpaper.png").exists());

        let result = apply_move(&database, &paths, plan).expect("apply");
        assert!(!original.exists());
        assert!(destination.join("wallpaper.png").exists());
        scan_catalog(
            &database,
            &paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("rescan after managed move");
        assert_eq!(
            database.get_image(id).expect("load").expect("image").status,
            ImageStatus::Ready,
            "a managed move outside registered roots must remain catalogued"
        );
        let moved_path = destination.join("wallpaper.png");
        fs::write(&moved_path, b"tampered after move").expect("tamper");
        assert!(
            undo_move(&database, &paths, result.id).is_err(),
            "undo must reject a changed destination"
        );
        fs::write(&moved_path, &original_bytes).expect("restore moved bytes");
        undo_move(&database, &paths, result.id).expect("undo");
        assert_eq!(fs::read(&original).expect("restored"), original_bytes);
        assert!(!destination.join("wallpaper.png").exists());
    }

    #[test]
    fn manifest_and_undo_recover_from_a_catalog_update_failure() {
        let directory = tempfile::tempdir().expect("tempdir");
        let paths = AppPaths::from_xdg_roots(
            directory.path().join("cfg"),
            directory.path().join("data"),
            directory.path().join("cache"),
            directory.path().join("state"),
        );
        paths.ensure_owned_dirs().expect("dirs");
        let source = directory.path().join("source");
        let destination = directory.path().join("destination");
        fs::create_dir(&source).expect("source");
        fs::create_dir(&destination).expect("destination");
        let original = source.join("wallpaper.png");
        RgbImage::from_pixel(16, 8, Rgb([20, 40, 60]))
            .save(&original)
            .expect("image");
        let original_bytes = fs::read(&original).expect("bytes");
        let database = Database::open(&paths.database).expect("database");
        database.add_source(&source).expect("source root");
        scan_catalog(
            &database,
            &paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan");
        let id = database
            .image_id_by_path(&original)
            .expect("lookup")
            .expect("id");
        let image = database.get_image(id).expect("load").expect("image");
        let plan = plan_move(&[image], &destination).expect("plan");
        let operation_id = plan.id;
        database
            .with_connection(|connection| {
                connection.execute_batch(
                    "CREATE TRIGGER reject_catalog_path_change
                     BEFORE UPDATE OF path ON images
                     WHEN NEW.path != OLD.path
                     BEGIN
                        SELECT RAISE(ABORT, 'injected catalog update failure');
                     END;",
                )?;
                Ok(())
            })
            .expect("failure trigger");

        let error = apply_move(&database, &paths, plan).expect_err("catalog update must fail");
        assert!(format!("{error:#}").contains("file moved but recovery state was incomplete"));
        let moved = destination.join("wallpaper.png");
        assert!(!original.exists());
        assert_eq!(fs::read(&moved).expect("moved bytes"), original_bytes);
        let manifest: MovePlan = serde_json::from_slice(
            &fs::read(manifest_path(&paths, operation_id)).expect("manifest"),
        )
        .expect("decode manifest");
        assert_eq!(manifest.items[0].status, MoveItemStatus::Moved);
        assert_eq!(
            load_operation(&database, operation_id)
                .expect("catalog operation")
                .items[0]
                .status,
            MoveItemStatus::Moved
        );

        undo_move(&database, &paths, operation_id).expect("recovery undo");
        assert_eq!(fs::read(&original).expect("restored bytes"), original_bytes);
        assert!(!moved.exists());
        assert_eq!(
            database
                .get_image(id)
                .expect("catalog image")
                .expect("image")
                .path,
            original
        );
    }

    #[test]
    fn retrying_undo_reconciles_an_already_restored_file() {
        let directory = tempfile::tempdir().expect("tempdir");
        let paths = AppPaths::from_xdg_roots(
            directory.path().join("cfg"),
            directory.path().join("data"),
            directory.path().join("cache"),
            directory.path().join("state"),
        );
        paths.ensure_owned_dirs().expect("dirs");
        let source = directory.path().join("source");
        let destination = directory.path().join("destination");
        fs::create_dir(&source).expect("source");
        fs::create_dir(&destination).expect("destination");
        let original = source.join("wallpaper.png");
        RgbImage::from_pixel(16, 8, Rgb([20, 40, 60]))
            .save(&original)
            .expect("image");
        let original_bytes = fs::read(&original).expect("bytes");
        let database = Database::open(&paths.database).expect("database");
        database.add_source(&source).expect("source root");
        scan_catalog(
            &database,
            &paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("scan");
        let id = database
            .image_id_by_path(&original)
            .expect("lookup")
            .expect("id");
        let image = database.get_image(id).expect("load").expect("image");
        let result = apply_move(
            &database,
            &paths,
            plan_move(&[image], &destination).expect("plan"),
        )
        .expect("apply");
        let moved = destination.join("wallpaper.png");
        database
            .with_connection(|connection| {
                connection.execute_batch(
                    "CREATE TRIGGER reject_catalog_path_change
                     BEFORE UPDATE OF path ON images
                     WHEN NEW.path != OLD.path
                     BEGIN
                        SELECT RAISE(ABORT, 'injected catalog update failure');
                     END;",
                )?;
                Ok(())
            })
            .expect("failure trigger");

        let error = undo_move(&database, &paths, result.id).expect_err("undo update must fail");
        assert!(format!("{error:#}").contains("file was restored"));
        assert_eq!(fs::read(&original).expect("restored bytes"), original_bytes);
        assert!(!moved.exists());
        assert_eq!(
            database
                .get_image(id)
                .expect("catalog image")
                .expect("image")
                .path,
            moved
        );
        assert_eq!(
            load_operation(&database, result.id)
                .expect("partial operation")
                .items[0]
                .status,
            MoveItemStatus::Undone
        );

        database
            .with_connection(|connection| {
                connection.execute_batch("DROP TRIGGER reject_catalog_path_change")?;
                Ok(())
            })
            .expect("remove failure trigger");
        undo_move(&database, &paths, result.id).expect("retry undo");
        assert_eq!(fs::read(&original).expect("final bytes"), original_bytes);
        assert!(!moved.exists());
        assert_eq!(
            database
                .get_image(id)
                .expect("catalog image")
                .expect("image")
                .path,
            original
        );
    }

    #[test]
    fn rejects_collisions_before_any_move() {
        let image = ImageRecord {
            id: 1,
            source_id: 1,
            path: PathBuf::from("/one/same.jpg"),
            size: 1,
            modified_ns: 0,
            hash: Some("hash".into()),
            status: ImageStatus::Ready,
            error: None,
            width: Some(1),
            height: Some(1),
            ratio: Some(1.0),
            orientation: Some("square".into()),
            common_ratio: Some("1:1".into()),
            dominant_hex: None,
            dominant_name: None,
            luminance: None,
            saturation: None,
            contrast: None,
            light_dark: None,
            thumbnail_path: None,
            palette: Vec::new(),
            ai_estimates: Vec::new(),
            favorite: false,
            tags: Vec::new(),
        };
        let mut other = image.clone();
        other.id = 2;
        other.path = PathBuf::from("/two/same.jpg");
        assert!(plan_move(&[image, other], Path::new("/destination")).is_err());
    }

    #[test]
    fn relative_move_destination_is_stored_as_an_absolute_path() {
        let mut image = ImageRecord {
            id: 1,
            source_id: 1,
            path: PathBuf::from("/source/wallpaper.jpg"),
            size: 1,
            modified_ns: 0,
            hash: Some("hash".into()),
            status: ImageStatus::Ready,
            error: None,
            width: Some(1),
            height: Some(1),
            ratio: Some(1.0),
            orientation: Some("square".into()),
            common_ratio: Some("1:1".into()),
            dominant_hex: None,
            dominant_name: None,
            luminance: None,
            saturation: None,
            contrast: None,
            light_dark: None,
            thumbnail_path: None,
            palette: Vec::new(),
            ai_estimates: Vec::new(),
            favorite: false,
            tags: Vec::new(),
        };
        let relative = PathBuf::from(format!("relative-destination-{}", Uuid::new_v4()));
        let plan = plan_move(std::slice::from_ref(&image), &relative).expect("relative plan");
        assert!(plan.destination_root.is_absolute());
        assert!(plan.items[0].destination.is_absolute());
        assert_eq!(
            plan.items[0].destination.file_name(),
            image.path.file_name()
        );

        image.path = plan.items[0].destination.clone();
        assert!(plan_move(&[image], &plan.destination_root).is_err());
    }

    #[test]
    fn validation_rejects_a_destination_outside_the_planned_root() {
        let directory = tempfile::tempdir().expect("tempdir");
        let source = directory.path().join("wallpaper.jpg");
        let destination_root = directory.path().join("destination");
        fs::write(&source, b"wallpaper bytes").expect("source file");
        fs::create_dir(&destination_root).expect("destination root");
        let plan = MovePlan {
            id: Uuid::new_v4(),
            destination_root,
            items: vec![MovePlanItem {
                image_id: 1,
                original_path: source.clone(),
                destination: directory.path().join("outside.jpg"),
                hash: blake3_file(&source).expect("source hash"),
                status: MoveItemStatus::Planned,
                error: None,
            }],
        };

        let error = validate_plan(&plan).expect_err("outside destination");
        assert!(format!("{error:#}").contains("outside the planned root"));
    }

    #[test]
    fn failed_cross_filesystem_source_removal_rolls_back_the_copy() {
        let directory = tempfile::tempdir().expect("tempdir");
        let source_directory = directory.path().join("source");
        let destination_directory = directory.path().join("destination");
        fs::create_dir(&source_directory).expect("source directory");
        fs::create_dir(&destination_directory).expect("destination directory");
        let source = source_directory.join("wallpaper.jpg");
        let destination = destination_directory.join("wallpaper.jpg");
        fs::write(&source, b"wallpaper bytes").expect("source file");
        let hash = blake3_file(&source).expect("source hash");

        let error = copy_across_filesystems_with_remove(&source, &destination, &hash, |_| {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected source removal failure",
            ))
        })
        .expect_err("source removal must fail");

        assert!(format!("{error:#}").contains("copied destination was rolled back"));
        assert_eq!(
            fs::read(&source).expect("source preserved"),
            b"wallpaper bytes"
        );
        assert!(!destination.exists());
        assert_eq!(
            fs::read_dir(&destination_directory)
                .expect("destination directory")
                .count(),
            0,
            "the failed copy must not leave a temporary file"
        );
    }
}
