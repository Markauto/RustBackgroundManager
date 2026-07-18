use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use rustix::fs::{CWD, RenameFlags, renameat_with};
use serde::Serialize;
use tempfile::{Builder, NamedTempFile};
use toml_edit::{DocumentMut, Item, value};

use crate::{
    AppPaths,
    collection::{collection_images, get_collection},
    db::{Database, path_bytes, path_from_bytes},
};

pub const SUPPORTED_DISPLAYS: [&str; 4] = ["any", "DP-1", "DP-2", "HDMI-A-1"];

#[derive(Clone, Debug, Serialize)]
pub struct Binding {
    pub display: String,
    pub collection_id: i64,
    pub collection_name: String,
    pub pool_path: PathBuf,
    pub displaced_path: Option<String>,
    pub active: bool,
    pub refreshed_at: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UnbindResult {
    pub display: String,
    pub restored: bool,
    pub config_was_changed_elsewhere: bool,
}

pub fn bind(
    database: &Database,
    paths: &AppPaths,
    display: &str,
    collection_name: &str,
) -> Result<Binding> {
    validate_display(display)?;
    let collection = get_collection(database, collection_name)?
        .with_context(|| format!("collection not found: {collection_name}"))?;
    let images = collection_images(database, paths, collection_name)?;
    if images.is_empty() {
        bail!("refusing to bind empty collection: {collection_name}");
    }
    let pool_path = paths.pools_dir.join(display);
    materialize_pool(&pool_path, &images)?;

    let mut document = read_wpaperd_document(&paths.wpaperd_config)?;
    let current = section_path(&document, display)?;
    let existing = binding_for_display(database, display)?;
    let displaced = match existing.as_ref() {
        Some(binding) => binding.displaced_path.clone(),
        None => current,
    };
    back_up_config_once(paths)?;
    set_section_path(&mut document, display, &pool_path)?;
    write_wpaperd_document(&paths.wpaperd_config, &document)?;

    let refreshed_at = Utc::now().timestamp_millis();
    database.with_connection(|connection| {
        connection.execute(
            "INSERT INTO wpaperd_bindings(
                display, collection_id, pool_path, displaced_path, active, refreshed_at
             ) VALUES (?1, ?2, ?3, ?4, 1, ?5)
             ON CONFLICT(display) DO UPDATE SET
                collection_id=excluded.collection_id, pool_path=excluded.pool_path,
                displaced_path=COALESCE(wpaperd_bindings.displaced_path, excluded.displaced_path),
                active=1, refreshed_at=excluded.refreshed_at",
            params![
                display,
                collection.id,
                path_bytes(&pool_path),
                displaced,
                refreshed_at,
            ],
        )?;
        Ok(())
    })?;
    binding_for_display(database, display)?.context("binding disappeared after it was saved")
}

pub fn refresh(
    database: &Database,
    paths: &AppPaths,
    display: Option<&str>,
) -> Result<Vec<Binding>> {
    if let Some(display) = display {
        validate_display(display)?;
    }
    let bindings = list_bindings(database)?;
    let selected: Vec<_> = bindings
        .into_iter()
        .filter(|binding| display.is_none_or(|wanted| binding.display == wanted))
        .collect();
    if let Some(display) = display
        && selected.is_empty()
    {
        bail!("display is not bound: {display}");
    }
    let mut refreshed = Vec::with_capacity(selected.len());
    for binding in selected {
        let images = collection_images(database, paths, &binding.collection_name)?;
        if images.is_empty() {
            bail!(
                "refusing to replace {} with an empty collection ({})",
                binding.pool_path.display(),
                binding.collection_name
            );
        }
        materialize_pool(&binding.pool_path, &images)?;
        let now = Utc::now().timestamp_millis();
        database.with_connection(|connection| {
            connection.execute(
                "UPDATE wpaperd_bindings SET refreshed_at=?1 WHERE display=?2",
                params![now, binding.display],
            )?;
            Ok(())
        })?;
        refreshed
            .push(binding_for_display(database, &binding.display)?.context("binding vanished")?);
    }
    Ok(refreshed)
}

pub fn unbind(database: &Database, paths: &AppPaths, display: &str) -> Result<UnbindResult> {
    validate_display(display)?;
    let binding = binding_for_display(database, display)?
        .with_context(|| format!("display is not bound: {display}"))?;
    let mut document = read_wpaperd_document(&paths.wpaperd_config)?;
    let current = section_path(&document, display)?;
    let managed = path_string(&binding.pool_path)?;
    let points_to_managed = current.as_deref() == Some(managed.as_str());
    if points_to_managed {
        match &binding.displaced_path {
            Some(previous) => set_section_path_string(&mut document, display, previous),
            None => remove_section_path(&mut document, display),
        }
        write_wpaperd_document(&paths.wpaperd_config, &document)?;
    }
    database.with_connection(|connection| {
        connection.execute("DELETE FROM wpaperd_bindings WHERE display=?1", [display])?;
        Ok(())
    })?;
    if binding.pool_path.is_dir() {
        fs::remove_dir_all(&binding.pool_path).with_context(|| {
            format!(
                "failed to remove managed pool {}",
                binding.pool_path.display()
            )
        })?;
    }
    Ok(UnbindResult {
        display: display.to_owned(),
        restored: points_to_managed,
        config_was_changed_elsewhere: !points_to_managed,
    })
}

pub fn list_bindings(database: &Database) -> Result<Vec<Binding>> {
    database.with_connection(|connection| {
        let mut statement = connection.prepare(
            "SELECT b.display, b.collection_id, c.name, b.pool_path, b.displaced_path,
                    b.active, b.refreshed_at
             FROM wpaperd_bindings b JOIN collections c ON c.id=b.collection_id
             ORDER BY CASE b.display WHEN 'any' THEN 0 ELSE 1 END, b.display",
        )?;
        let rows = statement.query_map([], binding_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

fn binding_for_display(database: &Database, display: &str) -> Result<Option<Binding>> {
    database.with_connection(|connection| {
        connection
            .query_row(
                "SELECT b.display, b.collection_id, c.name, b.pool_path, b.displaced_path,
                        b.active, b.refreshed_at
                 FROM wpaperd_bindings b JOIN collections c ON c.id=b.collection_id
                 WHERE b.display=?1",
                [display],
                binding_from_row,
            )
            .optional()
            .map_err(Into::into)
    })
}

fn binding_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Binding> {
    Ok(Binding {
        display: row.get(0)?,
        collection_id: row.get(1)?,
        collection_name: row.get(2)?,
        pool_path: path_from_bytes(row.get_ref(3)?.as_blob()?),
        displaced_path: row.get(4)?,
        active: row.get(5)?,
        refreshed_at: row.get(6)?,
    })
}

fn materialize_pool(pool_path: &Path, images: &[crate::db::ImageRecord]) -> Result<()> {
    if images.is_empty() {
        bail!("refusing to materialize an empty wpaperd pool");
    }
    let parent = pool_path.parent().context("managed pool has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = Builder::new().prefix(".bgm-pool-").tempdir_in(parent)?;
    for image in images {
        let extension = image
            .path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("img");
        let hash = image.hash.as_deref().unwrap_or("unhashed");
        let short_hash = &hash[..hash.len().min(12)];
        let link = temporary
            .path()
            .join(format!("{:010}-{short_hash}.{extension}", image.id));
        #[cfg(unix)]
        std::os::unix::fs::symlink(&image.path, &link).with_context(|| {
            format!("failed to link {} into managed pool", image.path.display())
        })?;
        #[cfg(not(unix))]
        compile_error!("bgm's wpaperd integration requires Unix symlinks");
    }
    FileSync::directory(temporary.path())?;
    let temporary_path = temporary.keep();
    if pool_path.exists() {
        renameat_with(CWD, &temporary_path, CWD, pool_path, RenameFlags::EXCHANGE)
            .map_err(errno_to_io)
            .context("failed to atomically exchange managed wpaperd pool")?;
        fs::remove_dir_all(&temporary_path)?;
    } else {
        renameat_with(CWD, &temporary_path, CWD, pool_path, RenameFlags::NOREPLACE)
            .map_err(errno_to_io)
            .context("failed to install managed wpaperd pool")?;
    }
    FileSync::directory(parent)?;
    Ok(())
}

fn validate_display(display: &str) -> Result<()> {
    if !SUPPORTED_DISPLAYS.contains(&display) {
        bail!(
            "unsupported display {display}; expected one of {}",
            SUPPORTED_DISPLAYS.join(", ")
        );
    }
    Ok(())
}

fn read_wpaperd_document(path: &Path) -> Result<DocumentMut> {
    if !path.exists() {
        return Ok(DocumentMut::new());
    }
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    text.parse::<DocumentMut>()
        .with_context(|| format!("invalid wpaperd TOML in {}", path.display()))
}

fn section_path(document: &DocumentMut, display: &str) -> Result<Option<String>> {
    let Some(section) = document.get(display) else {
        return Ok(None);
    };
    let Some(table) = section.as_table_like() else {
        bail!("wpaperd section [{display}] is not a table");
    };
    let Some(path) = table.get("path") else {
        return Ok(None);
    };
    path.as_str()
        .map(|value| Some(value.to_owned()))
        .context(format!("wpaperd [{display}].path is not a string"))
}

fn set_section_path(document: &mut DocumentMut, display: &str, path: &Path) -> Result<()> {
    let path = path_string(path)?;
    set_section_path_string(document, display, &path);
    Ok(())
}

fn set_section_path_string(document: &mut DocumentMut, display: &str, path: &str) {
    if !document.contains_key(display) {
        document[display] = Item::Table(toml_edit::Table::new());
    }
    document[display]["path"] = value(path);
}

fn remove_section_path(document: &mut DocumentMut, display: &str) {
    let remove_empty_section = document
        .get_mut(display)
        .and_then(Item::as_table_like_mut)
        .is_some_and(|table| {
            table.remove("path");
            table.is_empty()
        });
    if remove_empty_section {
        document.remove(display);
    }
}

fn write_wpaperd_document(path: &Path, document: &DocumentMut) -> Result<()> {
    let parent = path.parent().context("wpaperd config has no parent")?;
    fs::create_dir_all(parent)?;
    let mut temporary = NamedTempFile::new_in(parent)?;
    use std::io::Write as _;
    temporary.write_all(document.to_string().as_bytes())?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to replace {}", path.display()))?;
    FileSync::directory(parent)?;
    Ok(())
}

fn back_up_config_once(paths: &AppPaths) -> Result<()> {
    fs::create_dir_all(&paths.backups_dir)?;
    let backup = paths.backups_dir.join("wpaperd-config.toml");
    let absent = paths.backups_dir.join("wpaperd-config.absent");
    if backup.exists() || absent.exists() {
        return Ok(());
    }
    if paths.wpaperd_config.exists() {
        let mut source = fs::File::open(&paths.wpaperd_config)?;
        let mut destination = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&backup)?;
        std::io::copy(&mut source, &mut destination)?;
        destination.sync_all()?;
    } else {
        let marker = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&absent)?;
        marker.sync_all()?;
    }
    FileSync::directory(&paths.backups_dir)?;
    Ok(())
}

fn path_string(path: &Path) -> Result<String> {
    path.to_str().map(str::to_owned).with_context(|| {
        format!(
            "wpaperd cannot represent non-UTF-8 path: {}",
            path.display()
        )
    })
}

fn errno_to_io(error: rustix::io::Errno) -> std::io::Error {
    std::io::Error::from_raw_os_error(error.raw_os_error())
}

struct FileSync;

impl FileSync {
    fn directory(path: &Path) -> Result<()> {
        fs::File::open(path)?.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use crate::{
        collection::{delete_collection, save_collection},
        config::Config,
        filter::FilterSpecV1,
        scan::{ScanOptions, scan_catalog},
    };

    use super::*;

    struct Fixture {
        _directory: tempfile::TempDir,
        paths: AppPaths,
        database: Database,
        source: PathBuf,
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
            paths.ensure_owned_dirs().expect("dirs");
            let source = root.join("images");
            fs::create_dir(&source).expect("images");
            let database = Database::open(&paths.database).expect("database");
            database.add_source(&source).expect("source");
            Self {
                _directory: directory,
                paths,
                database,
                source,
            }
        }

        fn prepare_collection(&self) {
            RgbImage::from_pixel(32, 16, Rgb([10, 40, 80]))
                .save(self.source.join("wall.png"))
                .expect("image");
            scan_catalog(
                &self.database,
                &self.paths,
                &Config::default(),
                ScanOptions::default(),
            )
            .expect("scan");
            save_collection(&self.database, "all", &FilterSpecV1::default()).expect("collection");
        }
    }

    #[test]
    fn bind_preserves_unrelated_toml_and_unbind_restores_path() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        fs::create_dir_all(fixture.paths.wpaperd_config.parent().expect("parent")).expect("parent");
        let original = "# keep this comment\n[any]\npath = \"/old/walls\"\nduration = \"30m\"\n\n[default]\nmode = \"center\"\n";
        fs::write(&fixture.paths.wpaperd_config, original).expect("config");
        let binding = bind(&fixture.database, &fixture.paths, "any", "all").expect("bind");
        let changed = fs::read_to_string(&fixture.paths.wpaperd_config).expect("changed");
        assert!(changed.contains("# keep this comment"));
        assert!(changed.contains("duration = \"30m\""));
        assert!(changed.contains("[default]"));
        assert!(binding.pool_path.is_dir());
        assert_eq!(fs::read_dir(&binding.pool_path).expect("pool").count(), 1);

        let result = unbind(&fixture.database, &fixture.paths, "any").expect("unbind");
        assert!(result.restored);
        let restored = fs::read_to_string(&fixture.paths.wpaperd_config).expect("restored");
        assert!(restored.contains("path = \"/old/walls\""));
        assert!(restored.contains("duration = \"30m\""));
        assert!(
            fixture
                .paths
                .backups_dir
                .join("wpaperd-config.toml")
                .exists()
        );
    }

    #[test]
    fn refuses_empty_collection() {
        let fixture = Fixture::new();
        save_collection(&fixture.database, "empty", &FilterSpecV1::default()).expect("collection");
        assert!(bind(&fixture.database, &fixture.paths, "DP-1", "empty").is_err());
    }

    #[test]
    fn rebinding_does_not_treat_the_managed_path_as_displaced() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        bind(&fixture.database, &fixture.paths, "any", "all").expect("first bind");
        let rebound = bind(&fixture.database, &fixture.paths, "any", "all").expect("rebind");
        assert!(rebound.displaced_path.is_none());

        let result = unbind(&fixture.database, &fixture.paths, "any").expect("unbind");
        assert!(result.restored);
        let config = fs::read_to_string(&fixture.paths.wpaperd_config).expect("config");
        assert!(config.trim().is_empty());
    }

    #[test]
    fn bound_collection_must_be_unbound_before_deletion() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        bind(&fixture.database, &fixture.paths, "DP-1", "all").expect("bind");
        assert!(delete_collection(&fixture.database, "all").is_err());
        assert_eq!(list_bindings(&fixture.database).expect("bindings").len(), 1);
    }
}
