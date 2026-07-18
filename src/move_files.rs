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
    if destination_root.exists() && !destination_root.is_dir() {
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
        if fs::symlink_metadata(&destination).is_ok() {
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
        destination_root: destination_root.to_owned(),
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
                update_moved_item(database, plan.id, index, &plan.items[index])?;
                write_manifest(&manifest, &plan)?;
            }
            Err(error) => {
                let message = format!("{error:#}");
                plan.items[index].status = MoveItemStatus::Failed;
                plan.items[index].error = Some(message.clone());
                finish_operation(database, plan.id, "partial", Some(&message))?;
                update_item_status(
                    database,
                    plan.id,
                    index,
                    MoveItemStatus::Failed,
                    Some(&message),
                )?;
                write_manifest(&manifest, &plan)?;
                bail!(
                    "move {} stopped after {moved} file(s): {message}; partial manifest: {}",
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
    let moved_indices: Vec<_> = plan
        .items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (item.status == MoveItemStatus::Moved).then_some(index))
        .collect();
    if moved_indices.is_empty() {
        bail!("move {operation_id} has no files that can be undone");
    }
    for index in &moved_indices {
        let item = &plan.items[*index];
        if fs::symlink_metadata(&item.original_path).is_ok() {
            bail!(
                "cannot undo: original path exists: {}",
                item.original_path.display()
            );
        }
        let current = hash_file(&item.destination).with_context(|| {
            format!(
                "cannot undo: destination is unavailable: {}",
                item.destination.display()
            )
        })?;
        if current != item.hash {
            bail!(
                "cannot undo: destination hash changed: {}",
                item.destination.display()
            );
        }
    }

    let manifest = manifest_path(paths, operation_id);
    let mut undone = 0;
    for index in moved_indices.into_iter().rev() {
        if let Some(parent) = plan.items[index].original_path.parent() {
            fs::create_dir_all(parent)?;
        }
        move_one(
            &plan.items[index].destination,
            &plan.items[index].original_path,
            &plan.items[index].hash,
        )?;
        plan.items[index].status = MoveItemStatus::Undone;
        update_undone_item(database, operation_id, index, &plan.items[index])?;
        write_manifest(&manifest, &plan)?;
        undone += 1;
    }
    database.with_connection(|connection| {
        connection.execute(
            "UPDATE move_operations SET status='undone', undone_at=?1, error=NULL WHERE id=?2",
            params![Utc::now().timestamp_millis(), operation_id.to_string()],
        )?;
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
    let mut destinations = HashSet::with_capacity(plan.items.len());
    for item in &plan.items {
        if !destinations.insert(&item.destination) {
            bail!("duplicate destination: {}", item.destination.display());
        }
        if fs::symlink_metadata(&item.destination).is_ok() {
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
        let current = hash_file(&item.original_path)?;
        if current != item.hash {
            bail!(
                "source changed since scan: {}",
                item.original_path.display()
            );
        }
    }
    Ok(())
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
    let copied_hash = hash_file(temporary.path())?;
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
    if let Err(error) = fs::remove_file(source) {
        let rollback = fs::remove_file(destination);
        if let Err(rollback_error) = rollback {
            return Err(error).context(format!(
                "could not delete source or roll back destination (rollback: {rollback_error})"
            ));
        }
        return Err(error).context("could not delete source; copied destination was rolled back");
    }
    sync_parent(source)?;
    Ok(())
}

fn insert_operation(database: &Database, plan: &MovePlan) -> Result<()> {
    database.with_transaction(|transaction| {
        transaction.execute(
            "INSERT INTO move_operations(id, status, destination_root, created_at)
             VALUES (?1, 'in_progress', ?2, ?3)",
            params![
                plan.id.to_string(),
                path_bytes(&plan.destination_root),
                Utc::now().timestamp_millis(),
            ],
        )?;
        for (ordinal, item) in plan.items.iter().enumerate() {
            transaction.execute(
                "INSERT INTO move_items(
                    operation_id, ordinal, image_id, original_path, destination, blake3, status
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    plan.id.to_string(),
                    ordinal as i64,
                    item.image_id,
                    path_bytes(&item.original_path),
                    path_bytes(&item.destination),
                    item.hash,
                    item.status.as_str(),
                ],
            )?;
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
        transaction.execute(
            "UPDATE move_items SET status='moved', error=NULL
             WHERE operation_id=?1 AND ordinal=?2",
            params![operation_id.to_string(), ordinal as i64],
        )?;
        transaction.execute(
            "UPDATE images SET path=?1, size=?2, modified_ns=?3, updated_at=?4
             WHERE id=?5 AND path=?6",
            params![
                path_bytes(&item.destination),
                fs::metadata(&item.destination)?.len() as i64,
                modified_ns(&item.destination)?,
                Utc::now().timestamp_millis(),
                item.image_id,
                path_bytes(&item.original_path),
            ],
        )?;
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
        transaction.execute(
            "UPDATE move_items SET status='undone', error=NULL
             WHERE operation_id=?1 AND ordinal=?2",
            params![operation_id.to_string(), ordinal as i64],
        )?;
        transaction.execute(
            "UPDATE images SET path=?1, size=?2, modified_ns=?3, updated_at=?4
             WHERE id=?5 AND path=?6",
            params![
                path_bytes(&item.original_path),
                fs::metadata(&item.original_path)?.len() as i64,
                modified_ns(&item.original_path)?,
                Utc::now().timestamp_millis(),
                item.image_id,
                path_bytes(&item.destination),
            ],
        )?;
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
        connection.execute(
            "UPDATE move_items SET status=?1, error=?2 WHERE operation_id=?3 AND ordinal=?4",
            params![
                status.as_str(),
                error,
                operation_id.to_string(),
                ordinal as i64
            ],
        )?;
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
        connection.execute(
            "UPDATE move_operations SET status=?1, completed_at=?2, error=?3 WHERE id=?4",
            params![
                status,
                Utc::now().timestamp_millis(),
                error,
                operation_id.to_string(),
            ],
        )?;
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

fn hash_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut reader, &mut hasher)?;
    Ok(hasher.finalize().to_hex().to_string())
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
}
