use std::{
    collections::{HashMap, hash_map::Entry},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use rustix::fs::{CWD, RenameFlags, renameat_with};
use serde::Serialize;
use tempfile::{Builder, NamedTempFile, TempDir};
use toml_edit::{DocumentMut, Item, value};

use crate::{
    AppPaths,
    collection::{collection_images, get_collection},
    db::{Database, path_bytes, path_from_bytes},
    filesystem::resolve_file_target,
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

#[derive(Clone, Debug, Default, Serialize)]
pub struct RefreshReport {
    pub refreshed: Vec<Binding>,
    pub failures: Vec<RefreshFailure>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RefreshFailure {
    pub display: String,
    pub collection_name: String,
    pub error: String,
}

impl RefreshReport {
    pub fn failure_summary(&self) -> Option<String> {
        if self.failures.is_empty() {
            return None;
        }
        Some(
            self.failures
                .iter()
                .map(|failure| {
                    format!(
                        "{} ({}): {}",
                        failure.display, failure.collection_name, failure.error
                    )
                })
                .collect::<Vec<_>>()
                .join("; "),
        )
    }
}

pub fn bind(
    database: &Database,
    paths: &AppPaths,
    display: &str,
    collection_name: &str,
) -> Result<Binding> {
    bind_with_config_writer(
        database,
        paths,
        display,
        collection_name,
        write_wpaperd_document,
    )
}

fn bind_with_config_writer(
    database: &Database,
    paths: &AppPaths,
    display: &str,
    collection_name: &str,
    write_config: impl FnOnce(&Path, &DocumentMut) -> Result<()>,
) -> Result<Binding> {
    validate_display(display)?;
    let collection = get_collection(database, collection_name)?
        .with_context(|| format!("collection not found: {collection_name}"))?;
    let images = collection_images(database, paths, collection_name)?;
    if images.is_empty() {
        bail!("refusing to bind empty collection: {collection_name}");
    }
    let pool_path = paths.pools_dir.join(display);
    let mut document = read_wpaperd_document(&paths.wpaperd_config)?;
    let current = section_path(&document, display)?;
    let config_already_managed = current.as_deref() == Some(path_string(&pool_path)?.as_str());
    let existing = binding_for_display(database, display)?;
    let displaced = match existing.as_ref() {
        Some(binding) if config_already_managed => binding.displaced_path.clone(),
        Some(_) => current.clone(),
        None if config_already_managed => None,
        None => current.clone(),
    };
    back_up_config_once(paths)?;
    if !config_already_managed {
        set_section_path(&mut document, display, &pool_path)?;
    }
    let refreshed_at = Utc::now().timestamp_millis();
    let binding = Binding {
        display: display.to_owned(),
        collection_id: collection.id,
        collection_name: collection.name,
        pool_path: pool_path.clone(),
        displaced_path: displaced,
        active: true,
        refreshed_at: Some(refreshed_at),
    };
    let pool_update = begin_pool_update(&pool_path, &images)?;
    if let Err(error) = store_binding(database, &binding) {
        return Err(bind_rollback_error(
            error.context("failed to save the wpaperd binding"),
            restore_binding(database, display, existing.as_ref()),
            pool_update.rollback(),
        ));
    }
    if !config_already_managed && let Err(error) = write_config(&paths.wpaperd_config, &document) {
        return match config_points_to_pool(paths, display, &pool_path) {
            Ok(false) => Err(bind_rollback_error(
                error.context("failed to update the wpaperd config"),
                restore_binding(database, display, existing.as_ref()),
                pool_update.rollback(),
            )),
            Ok(true) => {
                let cleanup = pool_update.commit();
                let context = match cleanup {
                    Ok(()) => {
                        "the config writer reported an error after the managed path became active; the recoverable binding was retained".to_owned()
                    }
                    Err(cleanup) => format!(
                        "the managed path became active and the binding was retained, but displaced-pool cleanup also failed: {cleanup:#}"
                    ),
                };
                Err(error).context(context)
            }
            Err(verification) => {
                let cleanup = pool_update.commit();
                let cleanup = cleanup
                    .err()
                    .map(|error| format!("; displaced-pool cleanup failed: {error:#}"))
                    .unwrap_or_default();
                Err(error).context(format!(
                    "could not verify whether the managed path became active ({verification:#}); the catalog binding and pool were retained for recovery{cleanup}"
                ))
            }
        };
    }
    if let Err(error) = pool_update.commit() {
        return Err(error).context(
            "the wpaperd binding committed, but the displaced pool could not be cleaned up",
        );
    }
    Ok(binding)
}

fn store_binding(database: &Database, binding: &Binding) -> Result<()> {
    database.with_connection(|connection| {
        let changed = connection.execute(
            "INSERT INTO wpaperd_bindings(
                display, collection_id, pool_path, displaced_path, active, refreshed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(display) DO UPDATE SET
                collection_id=excluded.collection_id, pool_path=excluded.pool_path,
                displaced_path=excluded.displaced_path,
                active=excluded.active, refreshed_at=excluded.refreshed_at",
            params![
                binding.display,
                binding.collection_id,
                path_bytes(&binding.pool_path),
                binding.displaced_path,
                binding.active,
                binding.refreshed_at,
            ],
        )?;
        anyhow::ensure!(changed == 1, "wpaperd binding was not saved");
        Ok(())
    })
}

fn restore_binding(database: &Database, display: &str, previous: Option<&Binding>) -> Result<()> {
    match previous {
        Some(binding) => store_binding(database, binding),
        None => database.with_connection(|connection| {
            connection.execute("DELETE FROM wpaperd_bindings WHERE display=?1", [display])?;
            Ok(())
        }),
    }
}

fn bind_rollback_error(
    error: anyhow::Error,
    catalog_rollback: Result<()>,
    pool_rollback: Result<()>,
) -> anyhow::Error {
    let mut failures = Vec::new();
    if let Err(rollback) = catalog_rollback {
        failures.push(format!("catalog rollback: {rollback:#}"));
    }
    if let Err(rollback) = pool_rollback {
        failures.push(format!("pool rollback: {rollback:#}"));
    }
    if failures.is_empty() {
        error.context("the previous binding state was restored")
    } else {
        error.context(format!(
            "binding rollback was incomplete: {}",
            failures.join("; ")
        ))
    }
}

fn config_points_to_pool(paths: &AppPaths, display: &str, pool_path: &Path) -> Result<bool> {
    let document = read_wpaperd_document(&paths.wpaperd_config)?;
    Ok(section_path(&document, display)?.as_deref() == Some(path_string(pool_path)?.as_str()))
}

pub fn refresh(
    database: &Database,
    paths: &AppPaths,
    display: Option<&str>,
) -> Result<RefreshReport> {
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
    let mut report = RefreshReport {
        refreshed: Vec::with_capacity(selected.len()),
        failures: Vec::new(),
    };
    let mut collection_cache = HashMap::new();
    let mut materialized = Vec::with_capacity(selected.len());
    for binding in selected {
        if let Err(error) = validate_binding_pool(paths, &binding) {
            record_refresh_failure(&mut report, binding, error);
            continue;
        }
        if let Entry::Vacant(entry) = collection_cache.entry(binding.collection_id) {
            let images = collection_images(database, paths, &binding.collection_name)
                .map_err(|error| format!("{error:#}"));
            entry.insert(images);
        }
        let result = match collection_cache
            .get(&binding.collection_id)
            .context("collection refresh cache entry disappeared")?
        {
            Ok(images) => materialize_binding(binding.clone(), images),
            Err(error) => Err(anyhow::anyhow!(error.clone())),
        };
        match result {
            Ok(binding) => materialized.push(binding),
            Err(error) => record_refresh_failure(&mut report, binding, error),
        }
    }
    persist_refreshed_bindings(database, materialized, &mut report);
    Ok(report)
}

fn materialize_binding(binding: Binding, images: &[crate::db::ImageRecord]) -> Result<Binding> {
    if images.is_empty() {
        bail!(
            "refusing to replace {} with an empty collection",
            binding.pool_path.display()
        );
    }
    materialize_pool(&binding.pool_path, images)?;
    Ok(binding)
}

fn persist_refreshed_bindings(
    database: &Database,
    bindings: Vec<Binding>,
    report: &mut RefreshReport,
) {
    if bindings.is_empty() {
        return;
    }
    let now = Utc::now().timestamp_millis();
    let batched = database.with_transaction(|transaction| {
        let mut update = transaction
            .prepare_cached("UPDATE wpaperd_bindings SET refreshed_at=?1 WHERE display=?2")?;
        for binding in &bindings {
            let changed = update.execute(params![now, binding.display])?;
            anyhow::ensure!(
                changed == 1,
                "binding {} vanished during refresh",
                binding.display
            );
        }
        Ok(())
    });
    if batched.is_ok() {
        report
            .refreshed
            .extend(bindings.into_iter().map(|mut binding| {
                binding.refreshed_at = Some(now);
                binding
            }));
        return;
    }

    // If one display was concurrently removed or a database trigger rejects it,
    // retry individually so healthy pools still receive accurate timestamps.
    for mut binding in bindings {
        let result = database.with_connection(|connection| {
            let changed = connection.execute(
                "UPDATE wpaperd_bindings SET refreshed_at=?1 WHERE display=?2",
                params![now, binding.display],
            )?;
            anyhow::ensure!(changed == 1, "binding vanished during refresh");
            Ok(())
        });
        match result {
            Ok(()) => {
                binding.refreshed_at = Some(now);
                report.refreshed.push(binding);
            }
            Err(error) => record_refresh_failure(report, binding, error),
        }
    }
}

fn record_refresh_failure(report: &mut RefreshReport, binding: Binding, error: anyhow::Error) {
    report.failures.push(RefreshFailure {
        display: binding.display,
        collection_name: binding.collection_name,
        error: format!("{error:#}"),
    });
}

pub fn unbind(database: &Database, paths: &AppPaths, display: &str) -> Result<UnbindResult> {
    validate_display(display)?;
    let binding = binding_for_display(database, display)?
        .with_context(|| format!("display is not bound: {display}"))?;
    validate_binding_pool(paths, &binding)?;
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
    match fs::symlink_metadata(&binding.pool_path) {
        Ok(metadata) if metadata.file_type().is_dir() => {
            fs::remove_dir_all(&binding.pool_path).with_context(|| {
                format!(
                    "failed to remove managed pool {}",
                    binding.pool_path.display()
                )
            })?;
            FileSync::directory(&paths.pools_dir)?;
        }
        Ok(_) => bail!(
            "refusing to remove non-directory managed pool path {}",
            binding.pool_path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to inspect managed pool {}",
                    binding.pool_path.display()
                )
            });
        }
    }
    database.with_connection(|connection| {
        let changed =
            connection.execute("DELETE FROM wpaperd_bindings WHERE display=?1", [display])?;
        anyhow::ensure!(changed == 1, "wpaperd binding vanished during unbind");
        Ok(())
    })?;
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

fn materialize_pool(pool_path: &Path, images: &[crate::db::ImageRecord]) -> Result<bool> {
    let update = begin_pool_update(pool_path, images)?;
    let changed = update.changed();
    update.commit()?;
    Ok(changed)
}

enum PoolUpdate {
    Unchanged,
    Created {
        pool_path: PathBuf,
        temporary: TempDir,
    },
    Exchanged {
        pool_path: PathBuf,
        previous: TempDir,
    },
}

impl PoolUpdate {
    const fn changed(&self) -> bool {
        !matches!(self, Self::Unchanged)
    }

    fn commit(self) -> Result<()> {
        match self {
            Self::Unchanged | Self::Created { .. } => Ok(()),
            Self::Exchanged {
                pool_path,
                previous,
            } => {
                let parent = pool_path.parent().context("managed pool has no parent")?;
                previous
                    .close()
                    .context("failed to remove the displaced managed pool")?;
                FileSync::directory(parent)
            }
        }
    }

    fn rollback(self) -> Result<()> {
        match self {
            Self::Unchanged => Ok(()),
            Self::Created {
                pool_path,
                temporary,
            } => {
                renameat_with(
                    CWD,
                    &pool_path,
                    CWD,
                    temporary.path(),
                    RenameFlags::NOREPLACE,
                )
                .map_err(errno_to_io)
                .context("failed to roll back the new managed wpaperd pool")?;
                let parent = pool_path.parent().context("managed pool has no parent")?;
                FileSync::directory(parent)?;
                temporary
                    .close()
                    .context("failed to remove the rolled-back managed pool")
            }
            Self::Exchanged {
                pool_path,
                previous,
            } => {
                if let Err(error) =
                    renameat_with(CWD, &pool_path, CWD, previous.path(), RenameFlags::EXCHANGE)
                {
                    let recovery_path = previous.keep();
                    return Err(errno_to_io(error)).context(format!(
                        "failed to restore the previous managed pool; it remains at {}",
                        recovery_path.display()
                    ));
                }
                let parent = pool_path.parent().context("managed pool has no parent")?;
                FileSync::directory(parent)?;
                previous
                    .close()
                    .context("failed to remove the rolled-back managed pool")
            }
        }
    }
}

fn begin_pool_update(pool_path: &Path, images: &[crate::db::ImageRecord]) -> Result<PoolUpdate> {
    if images.is_empty() {
        bail!("refusing to materialize an empty wpaperd pool");
    }
    if pool_is_current(pool_path, images)? {
        return Ok(PoolUpdate::Unchanged);
    }
    let parent = pool_path.parent().context("managed pool has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = Builder::new().prefix(".bgm-pool-").tempdir_in(parent)?;
    for image in images {
        let link = temporary.path().join(pool_link_name(image));
        #[cfg(unix)]
        std::os::unix::fs::symlink(&image.path, &link).with_context(|| {
            format!("failed to link {} into managed pool", image.path.display())
        })?;
        #[cfg(not(unix))]
        compile_error!("bgm's wpaperd integration requires Unix symlinks");
    }
    FileSync::directory(temporary.path())?;
    let update = match fs::symlink_metadata(pool_path) {
        Ok(metadata) if !metadata.file_type().is_dir() => {
            bail!(
                "refusing to replace non-directory managed pool path {}",
                pool_path.display()
            );
        }
        Ok(_) => {
            renameat_with(CWD, temporary.path(), CWD, pool_path, RenameFlags::EXCHANGE)
                .map_err(errno_to_io)
                .context("failed to atomically exchange managed wpaperd pool")?;
            PoolUpdate::Exchanged {
                pool_path: pool_path.to_owned(),
                previous: temporary,
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            renameat_with(
                CWD,
                temporary.path(),
                CWD,
                pool_path,
                RenameFlags::NOREPLACE,
            )
            .map_err(errno_to_io)
            .context("failed to install managed wpaperd pool")?;
            PoolUpdate::Created {
                pool_path: pool_path.to_owned(),
                temporary,
            }
        }
        Err(error) => return Err(error.into()),
    };
    if let Err(error) = FileSync::directory(parent) {
        let rollback = update.rollback();
        return Err(error).context(match rollback {
            Ok(()) => "failed to sync the managed pool installation; it was rolled back".into(),
            Err(rollback) => format!(
                "failed to sync the managed pool installation and to roll it back: {rollback:#}"
            ),
        });
    }
    Ok(update)
}

fn pool_is_current(pool_path: &Path, images: &[crate::db::ImageRecord]) -> Result<bool> {
    let metadata = match fs::symlink_metadata(pool_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_dir() {
        return Ok(false);
    }
    let mut expected = images
        .iter()
        .map(|image| (pool_link_name(image), image.path.as_path()))
        .collect::<HashMap<_, _>>();
    if expected.len() != images.len() {
        return Ok(false);
    }
    for entry in fs::read_dir(pool_path)? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            return Ok(false);
        };
        let Some(target) = expected.remove(&name) else {
            return Ok(false);
        };
        if !entry.file_type()?.is_symlink() || fs::read_link(entry.path())? != target {
            return Ok(false);
        }
    }
    Ok(expected.is_empty())
}

fn pool_link_name(image: &crate::db::ImageRecord) -> String {
    let extension = image
        .path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| {
            value
                .chars()
                .filter(char::is_ascii_alphanumeric)
                .take(10)
                .collect::<String>()
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "img".into());
    let short_hash = image
        .hash
        .as_deref()
        .filter(|hash| hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map_or("unhashed", |hash| &hash[..hash.len().min(12)]);
    format!("{:010}-{short_hash}.{extension}", image.id)
}

fn validate_binding_pool(paths: &AppPaths, binding: &Binding) -> Result<()> {
    validate_display(&binding.display)?;
    let expected = paths.pools_dir.join(&binding.display);
    anyhow::ensure!(
        binding.pool_path == expected,
        "refusing unsafe managed pool path {} for display {}; expected {}",
        binding.pool_path.display(),
        binding.display,
        expected.display()
    );
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
    let storage_path = resolve_file_target(path)?;
    let text = match fs::read_to_string(&storage_path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(DocumentMut::new());
        }
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
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
    let storage_path = resolve_file_target(path)?;
    let parent = storage_path
        .parent()
        .context("wpaperd config has no parent")?;
    fs::create_dir_all(parent)?;
    let mut temporary = NamedTempFile::new_in(parent)?;
    if let Ok(metadata) = fs::metadata(&storage_path)
        && metadata.is_file()
    {
        temporary
            .as_file()
            .set_permissions(metadata.permissions())?;
    }
    use std::io::Write as _;
    temporary.write_all(document.to_string().as_bytes())?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(&storage_path)
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
    let storage_path = resolve_file_target(&paths.wpaperd_config)?;
    match fs::File::open(&storage_path) {
        Ok(mut source) => {
            let mut destination = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&backup)?;
            std::io::copy(&mut source, &mut destination)?;
            destination.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let marker = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&absent)?;
            marker.sync_all()?;
        }
        Err(error) => return Err(error).context("failed to open the wpaperd config for backup"),
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
        collection::{add_tag, delete_collection, remove_tag, save_collection},
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

    fn pool_targets(pool_path: &Path) -> Vec<PathBuf> {
        let mut targets = fs::read_dir(pool_path)
            .expect("pool")
            .map(|entry| fs::read_link(entry.expect("pool entry").path()).expect("link target"))
            .collect::<Vec<_>>();
        targets.sort();
        targets
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
        let images = collection_images(&fixture.database, &fixture.paths, "all").expect("images");
        assert!(
            !materialize_pool(&binding.pool_path, &images).expect("unchanged pool"),
            "an identical pool must not be exchanged"
        );
        let link = fs::read_dir(&binding.pool_path)
            .expect("pool")
            .next()
            .expect("pool link")
            .expect("link entry")
            .path();
        fs::remove_file(link).expect("remove managed link");
        assert!(materialize_pool(&binding.pool_path, &images).expect("repair pool"));
        assert!(!materialize_pool(&binding.pool_path, &images).expect("repaired pool"));

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

    #[cfg(unix)]
    #[test]
    fn bind_and_unbind_preserve_a_symlinked_wpaperd_config() {
        use std::os::unix::fs::symlink;

        let fixture = Fixture::new();
        fixture.prepare_collection();
        let target = fixture._directory.path().join("real-wpaperd.toml");
        let original = "[any]\npath = \"/old/walls\"\n";
        fs::write(&target, original).expect("target config");
        fs::create_dir_all(fixture.paths.wpaperd_config.parent().expect("parent"))
            .expect("config parent");
        symlink(&target, &fixture.paths.wpaperd_config).expect("config symlink");

        bind(&fixture.database, &fixture.paths, "any", "all").expect("bind");
        assert!(
            fs::symlink_metadata(&fixture.paths.wpaperd_config)
                .expect("link metadata")
                .file_type()
                .is_symlink()
        );
        assert!(
            fs::read_to_string(&target)
                .expect("managed config")
                .contains(
                    fixture
                        .paths
                        .pools_dir
                        .join("any")
                        .to_string_lossy()
                        .as_ref()
                )
        );

        unbind(&fixture.database, &fixture.paths, "any").expect("unbind");
        assert!(
            fs::symlink_metadata(&fixture.paths.wpaperd_config)
                .expect("link metadata")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::read_to_string(&target).expect("restored target"),
            original
        );
        assert_eq!(
            fs::read_to_string(fixture.paths.backups_dir.join("wpaperd-config.toml"))
                .expect("backup"),
            original
        );
    }

    #[cfg(unix)]
    #[test]
    fn bind_rejects_and_preserves_a_dangling_wpaperd_config_symlink() {
        use std::os::unix::fs::symlink;

        let fixture = Fixture::new();
        fixture.prepare_collection();
        fs::create_dir_all(fixture.paths.wpaperd_config.parent().expect("parent"))
            .expect("config parent");
        symlink(
            fixture._directory.path().join("missing-wpaperd.toml"),
            &fixture.paths.wpaperd_config,
        )
        .expect("dangling symlink");

        let error = bind(&fixture.database, &fixture.paths, "any", "all")
            .expect_err("dangling config must be rejected");

        assert!(format!("{error:#}").contains("failed to resolve symlink"));
        assert!(
            fs::symlink_metadata(&fixture.paths.wpaperd_config)
                .expect("link preserved")
                .file_type()
                .is_symlink()
        );
        assert!(
            list_bindings(&fixture.database)
                .expect("bindings")
                .is_empty()
        );
        assert!(!fixture.paths.pools_dir.join("any").exists());
    }

    #[test]
    fn refuses_empty_collection() {
        let fixture = Fixture::new();
        save_collection(&fixture.database, "empty", &FilterSpecV1::default()).expect("collection");
        assert!(bind(&fixture.database, &fixture.paths, "DP-1", "empty").is_err());
    }

    #[test]
    fn bind_rolls_back_catalog_and_pool_when_config_write_fails() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        fs::create_dir_all(fixture.paths.wpaperd_config.parent().expect("parent"))
            .expect("config parent");
        let original = "[any]\npath = \"/old/walls\"\n";
        fs::write(&fixture.paths.wpaperd_config, original).expect("config");

        let error =
            bind_with_config_writer(&fixture.database, &fixture.paths, "any", "all", |_, _| {
                anyhow::bail!("injected config write failure")
            })
            .expect_err("config failure");

        assert!(format!("{error:#}").contains("previous binding state was restored"));
        assert_eq!(
            fs::read_to_string(&fixture.paths.wpaperd_config).expect("unchanged config"),
            original
        );
        assert!(
            list_bindings(&fixture.database)
                .expect("bindings")
                .is_empty()
        );
        assert!(!fixture.paths.pools_dir.join("any").exists());
        assert!(
            fs::read_dir(&fixture.paths.pools_dir)
                .expect("pools")
                .all(|entry| !entry
                    .expect("pool entry")
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".bgm-pool-"))
        );
    }

    #[test]
    fn failed_rebind_restores_the_previous_pool_and_catalog_binding() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        let original = bind(&fixture.database, &fixture.paths, "any", "all").expect("bind");
        let original_targets = pool_targets(&original.pool_path);

        let second_path = fixture.source.join("second.png");
        RgbImage::from_pixel(24, 12, Rgb([80, 40, 10]))
            .save(&second_path)
            .expect("second image");
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("rescan");
        let second_id = fixture
            .database
            .image_id_by_path(&second_path)
            .expect("lookup")
            .expect("second id");
        add_tag(&fixture.database, &[second_id], "second").expect("tag");
        save_collection(
            &fixture.database,
            "second",
            &FilterSpecV1 {
                tags: vec!["second".into()],
                ..FilterSpecV1::default()
            },
        )
        .expect("second collection");
        fixture
            .database
            .with_connection(|connection| {
                connection.execute_batch(
                    "CREATE TRIGGER reject_rebind
                     BEFORE UPDATE OF collection_id ON wpaperd_bindings
                     WHEN NEW.collection_id != OLD.collection_id
                     BEGIN
                        SELECT RAISE(ABORT, 'injected rebind failure');
                     END;",
                )?;
                Ok(())
            })
            .expect("failure trigger");

        let error =
            bind(&fixture.database, &fixture.paths, "any", "second").expect_err("rebind must fail");
        assert!(format!("{error:#}").contains("previous binding state was restored"));
        assert_eq!(pool_targets(&original.pool_path), original_targets);
        let bindings = list_bindings(&fixture.database).expect("bindings");
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].collection_name, "all");
        assert!(
            fs::read_dir(&fixture.paths.pools_dir)
                .expect("pools")
                .all(|entry| !entry
                    .expect("pool entry")
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".bgm-pool-"))
        );
    }

    #[test]
    fn bind_does_not_replace_a_non_directory_pool_path() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        let pool_path = fixture.paths.pools_dir.join("any");
        fs::write(&pool_path, b"not a managed directory").expect("pool collision");

        let error =
            bind(&fixture.database, &fixture.paths, "any", "all").expect_err("pool collision");
        assert!(format!("{error:#}").contains("refusing to replace non-directory"));
        assert_eq!(
            fs::read(&pool_path).expect("preserved collision"),
            b"not a managed directory"
        );
        assert!(
            list_bindings(&fixture.database)
                .expect("bindings")
                .is_empty()
        );
        assert!(!fixture.paths.wpaperd_config.exists());
    }

    #[test]
    fn corrupted_binding_cannot_touch_a_pool_outside_owned_state() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        bind(&fixture.database, &fixture.paths, "any", "all").expect("bind");
        let outside = fixture._directory.path().join("outside-pool");
        fs::create_dir(&outside).expect("outside pool");
        fs::write(outside.join("sentinel"), b"preserve").expect("sentinel");
        fixture
            .database
            .with_connection(|connection| {
                connection.execute(
                    "UPDATE wpaperd_bindings SET pool_path=?1 WHERE display='any'",
                    [path_bytes(&outside)],
                )?;
                Ok(())
            })
            .expect("corrupt binding");

        let report = refresh(&fixture.database, &fixture.paths, None).expect("refresh report");
        assert!(report.refreshed.is_empty());
        assert_eq!(report.failures.len(), 1);
        assert!(
            report.failures[0]
                .error
                .contains("unsafe managed pool path")
        );
        let error = unbind(&fixture.database, &fixture.paths, "any").expect_err("unsafe unbind");
        assert!(format!("{error:#}").contains("unsafe managed pool path"));
        assert_eq!(
            fs::read(outside.join("sentinel")).expect("preserved sentinel"),
            b"preserve"
        );
    }

    #[test]
    fn failed_pool_cleanup_keeps_the_binding_retryable() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        let binding = bind(&fixture.database, &fixture.paths, "any", "all").expect("bind");
        fs::remove_dir_all(&binding.pool_path).expect("remove managed pool fixture");
        fs::write(&binding.pool_path, b"collision").expect("pool collision");

        let error = unbind(&fixture.database, &fixture.paths, "any")
            .expect_err("non-directory pool must be preserved");

        assert!(format!("{error:#}").contains("non-directory managed pool"));
        assert_eq!(
            fs::read(&binding.pool_path).expect("collision preserved"),
            b"collision"
        );
        assert_eq!(
            list_bindings(&fixture.database)
                .expect("binding retained")
                .len(),
            1
        );

        fs::remove_file(&binding.pool_path).expect("clear collision");
        let retry = unbind(&fixture.database, &fixture.paths, "any").expect("retry unbind");
        assert!(
            !retry.restored,
            "the first attempt already restored the config"
        );
        assert!(
            list_bindings(&fixture.database)
                .expect("bindings")
                .is_empty()
        );
    }

    #[test]
    fn refresh_continues_after_an_empty_bound_collection() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        let first_path = fixture.source.join("wall.png");
        let first_id = fixture
            .database
            .image_id_by_path(&first_path)
            .expect("lookup")
            .expect("image id");
        add_tag(&fixture.database, &[first_id], "temporary").expect("tag");
        save_collection(
            &fixture.database,
            "temporary",
            &FilterSpecV1 {
                tags: vec!["temporary".into()],
                ..FilterSpecV1::default()
            },
        )
        .expect("tagged collection");
        bind(&fixture.database, &fixture.paths, "any", "temporary").expect("first bind");
        bind(&fixture.database, &fixture.paths, "DP-1", "all").expect("second bind");

        remove_tag(&fixture.database, &[first_id], "temporary").expect("remove tag");
        RgbImage::from_pixel(24, 12, Rgb([80, 40, 10]))
            .save(fixture.source.join("second.png"))
            .expect("second image");
        scan_catalog(
            &fixture.database,
            &fixture.paths,
            &Config::default(),
            ScanOptions::default(),
        )
        .expect("rescan");

        let report = refresh(&fixture.database, &fixture.paths, None).expect("refresh report");
        assert_eq!(
            report
                .refreshed
                .iter()
                .map(|binding| binding.display.as_str())
                .collect::<Vec<_>>(),
            ["DP-1"]
        );
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].display, "any");
        assert_eq!(report.failures[0].collection_name, "temporary");
        assert!(report.failures[0].error.contains("empty collection"));
        assert_eq!(
            fs::read_dir(fixture.paths.pools_dir.join("any"))
                .expect("preserved pool")
                .count(),
            1
        );
        assert_eq!(
            fs::read_dir(fixture.paths.pools_dir.join("DP-1"))
                .expect("refreshed pool")
                .count(),
            2
        );
    }

    #[test]
    fn refresh_batch_falls_back_without_losing_healthy_displays() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        bind(&fixture.database, &fixture.paths, "any", "all").expect("first bind");
        bind(&fixture.database, &fixture.paths, "DP-1", "all").expect("second bind");
        fixture
            .database
            .with_connection(|connection| {
                connection.execute("UPDATE wpaperd_bindings SET refreshed_at=1", [])?;
                connection.execute_batch(
                    "CREATE TRIGGER reject_dp_refresh
                     BEFORE UPDATE OF refreshed_at ON wpaperd_bindings
                     WHEN OLD.display='DP-1'
                     BEGIN
                        SELECT RAISE(ABORT, 'injected refresh failure');
                     END;",
                )?;
                Ok(())
            })
            .expect("failure trigger");

        let report = refresh(&fixture.database, &fixture.paths, None).expect("refresh");
        assert_eq!(
            report
                .refreshed
                .iter()
                .map(|binding| binding.display.as_str())
                .collect::<Vec<_>>(),
            ["any"]
        );
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].display, "DP-1");
        assert!(
            report.failures[0]
                .error
                .contains("injected refresh failure")
        );
        let timestamps = fixture
            .database
            .with_connection(|connection| {
                let mut statement = connection.prepare(
                    "SELECT display, refreshed_at FROM wpaperd_bindings ORDER BY display",
                )?;
                let rows = statement.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })?;
                Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
            })
            .expect("timestamps");
        assert_eq!(timestamps[0], ("DP-1".into(), 1));
        assert!(timestamps[1].1 > 1);
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
    fn rebinding_after_an_external_edit_restores_the_new_live_path() {
        let fixture = Fixture::new();
        fixture.prepare_collection();
        fs::create_dir_all(fixture.paths.wpaperd_config.parent().expect("parent"))
            .expect("config parent");
        fs::write(
            &fixture.paths.wpaperd_config,
            "[any]\npath = \"/original/walls\"\n",
        )
        .expect("original config");
        bind(&fixture.database, &fixture.paths, "any", "all").expect("first bind");
        fs::write(
            &fixture.paths.wpaperd_config,
            "[any]\npath = \"/external/edit\"\n",
        )
        .expect("external edit");

        let rebound = bind(&fixture.database, &fixture.paths, "any", "all").expect("rebind");
        assert_eq!(rebound.displaced_path.as_deref(), Some("/external/edit"));
        let result = unbind(&fixture.database, &fixture.paths, "any").expect("unbind");
        assert!(result.restored);
        let restored = fs::read_to_string(&fixture.paths.wpaperd_config).expect("restored config");
        assert!(restored.contains("path = \"/external/edit\""));
        assert!(!restored.contains("/original/walls"));
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
