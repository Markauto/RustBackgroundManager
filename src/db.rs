use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};

use crate::analysis::{Oklab, PaletteColor};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug)]
pub struct Database {
    path: PathBuf,
    connection: Mutex<Connection>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SourceRoot {
    pub id: i64,
    pub path: PathBuf,
    pub added_at: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStatus {
    Discovered,
    Ready,
    OutOfBounds,
    Missing,
    Corrupt,
    Error,
}

impl ImageStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Discovered => "discovered",
            Self::Ready => "ready",
            Self::OutOfBounds => "out_of_bounds",
            Self::Missing => "missing",
            Self::Corrupt => "corrupt",
            Self::Error => "error",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "discovered" => Ok(Self::Discovered),
            "ready" => Ok(Self::Ready),
            "out_of_bounds" => Ok(Self::OutOfBounds),
            "missing" => Ok(Self::Missing),
            "corrupt" => Ok(Self::Corrupt),
            "error" => Ok(Self::Error),
            _ => bail!("unknown image status in database: {value}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageRecord {
    pub id: i64,
    pub source_id: i64,
    pub path: PathBuf,
    pub size: u64,
    pub modified_ns: i64,
    pub hash: Option<String>,
    pub status: ImageStatus,
    pub error: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub ratio: Option<f64>,
    pub orientation: Option<String>,
    pub common_ratio: Option<String>,
    pub dominant_hex: Option<String>,
    pub dominant_name: Option<String>,
    pub luminance: Option<f32>,
    pub saturation: Option<f32>,
    pub contrast: Option<f32>,
    pub light_dark: Option<String>,
    pub thumbnail_path: Option<PathBuf>,
    pub palette: Vec<PaletteColor>,
    pub ai_estimates: Vec<AiEstimate>,
    pub favorite: bool,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AiEstimate {
    pub pack: String,
    pub label: String,
    pub score: f32,
}

impl Database {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let mut connection = Connection::open(path)
            .with_context(|| format!("failed to open catalog at {}", path.display()))?;
        configure(&connection)?;
        migrate(&mut connection)?;
        seed_label_packs(&connection)?;
        Ok(Self {
            path: path.to_owned(),
            connection: Mutex::new(connection),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn with_connection<T>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T>,
    ) -> Result<T> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex is poisoned"))?;
        operation(&connection)
    }

    pub fn with_transaction<T>(
        &self,
        operation: impl FnOnce(&Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex is poisoned"))?;
        let transaction = connection.transaction()?;
        let result = operation(&transaction)?;
        transaction.commit()?;
        Ok(result)
    }

    pub fn add_source(&self, path: &Path) -> Result<SourceRoot> {
        let canonical = path
            .canonicalize()
            .with_context(|| format!("cannot access source {}", path.display()))?;
        if !canonical.is_dir() {
            bail!("source is not a directory: {}", canonical.display());
        }
        let now = Utc::now().timestamp_millis();
        self.with_connection(|connection| {
            connection.execute(
                "INSERT OR IGNORE INTO source_roots(path, added_at) VALUES (?1, ?2)",
                params![path_bytes(&canonical), now],
            )?;
            source_by_path(connection, &canonical)?.context("failed to retrieve inserted source")
        })
    }

    pub fn list_sources(&self) -> Result<Vec<SourceRoot>> {
        self.with_connection(|connection| {
            let mut statement =
                connection.prepare("SELECT id, path, added_at FROM source_roots ORDER BY path")?;
            let rows = statement.query_map([], |row| {
                Ok(SourceRoot {
                    id: row.get(0)?,
                    path: path_from_bytes(row.get_ref(1)?.as_blob()?),
                    added_at: row.get(2)?,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(Into::into)
        })
    }

    pub fn remove_source(&self, path: &Path) -> Result<bool> {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_owned());
        self.with_connection(|connection| {
            let changed = connection.execute(
                "DELETE FROM source_roots WHERE path = ?1",
                [path_bytes(&canonical)],
            )?;
            Ok(changed != 0)
        })
    }

    pub fn get_image(&self, id: i64) -> Result<Option<ImageRecord>> {
        self.with_connection(|connection| load_image(connection, id))
    }

    pub fn image_id_by_path(&self, path: &Path) -> Result<Option<i64>> {
        self.with_connection(|connection| {
            connection
                .query_row(
                    "SELECT id FROM images WHERE path = ?1",
                    [path_bytes(path)],
                    |row| row.get(0),
                )
                .optional()
                .map_err(Into::into)
        })
    }
}

fn configure(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = FULL;
         PRAGMA busy_timeout = 5000;",
    )?;
    Ok(())
}

fn migrate(connection: &mut Connection) -> Result<()> {
    let version: u32 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version > SCHEMA_VERSION {
        bail!("catalog schema {version} is newer than this bgm supports ({SCHEMA_VERSION})");
    }
    if version == 0 {
        let transaction = connection.transaction()?;
        migration_v1(&transaction)?;
        transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        transaction.commit()?;
    }
    Ok(())
}

fn migration_v1(transaction: &Transaction<'_>) -> Result<()> {
    transaction.execute_batch(
        "CREATE TABLE source_roots (
            id          INTEGER PRIMARY KEY,
            path        BLOB NOT NULL UNIQUE,
            added_at    INTEGER NOT NULL
        );

        CREATE TABLE images (
            id              INTEGER PRIMARY KEY,
            source_id       INTEGER NOT NULL REFERENCES source_roots(id) ON DELETE CASCADE,
            path            BLOB NOT NULL UNIQUE,
            size            INTEGER NOT NULL,
            modified_ns     INTEGER NOT NULL,
            blake3          TEXT,
            status          TEXT NOT NULL,
            error           TEXT,
            width           INTEGER,
            height          INTEGER,
            ratio           REAL,
            orientation     TEXT,
            common_ratio    TEXT,
            dominant_hex    TEXT,
            dominant_name   TEXT,
            luminance       REAL,
            saturation      REAL,
            contrast        REAL,
            light_dark      TEXT,
            thumbnail_path  BLOB,
            analysis_key    TEXT,
            discovered_at   INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL,
            missing_since   INTEGER
        );
        CREATE INDEX images_source_idx ON images(source_id);
        CREATE INDEX images_status_idx ON images(status);
        CREATE INDEX images_dimensions_idx ON images(width, height);
        CREATE INDEX images_ratio_idx ON images(ratio);

        CREATE TABLE image_palette (
            image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            rank        INTEGER NOT NULL,
            oklab_l     REAL NOT NULL,
            oklab_a     REAL NOT NULL,
            oklab_b     REAL NOT NULL,
            proportion  REAL NOT NULL,
            hex         TEXT NOT NULL,
            name        TEXT NOT NULL,
            PRIMARY KEY(image_id, rank)
        );

        CREATE TABLE embeddings (
            image_id      INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            model_id      TEXT NOT NULL,
            dimension     INTEGER NOT NULL,
            vector        BLOB NOT NULL,
            normalized    INTEGER NOT NULL CHECK(normalized IN (0, 1)),
            created_at    INTEGER NOT NULL,
            PRIMARY KEY(image_id, model_id)
        );

        CREATE TABLE label_packs (
            id          INTEGER PRIMARY KEY,
            name        TEXT NOT NULL COLLATE NOCASE UNIQUE,
            kind        TEXT NOT NULL,
            labels_json TEXT NOT NULL,
            updated_at  INTEGER NOT NULL
        );
        CREATE TABLE label_scores (
            image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            pack_id     INTEGER NOT NULL REFERENCES label_packs(id) ON DELETE CASCADE,
            label       TEXT NOT NULL,
            score       REAL NOT NULL,
            PRIMARY KEY(image_id, pack_id, label)
        );

        CREATE TABLE tags (
            id      INTEGER PRIMARY KEY,
            name    TEXT NOT NULL UNIQUE COLLATE NOCASE
        );
        CREATE TABLE image_tags (
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            tag_id   INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY(image_id, tag_id)
        );
        CREATE TABLE favorites (
            image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            set_at   INTEGER NOT NULL
        );

        CREATE TABLE collections (
            id              INTEGER PRIMARY KEY,
            name            TEXT NOT NULL UNIQUE COLLATE NOCASE,
            filter_version  INTEGER NOT NULL,
            filter_json     TEXT NOT NULL,
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        );

        CREATE TABLE wpaperd_bindings (
            display          TEXT PRIMARY KEY,
            collection_id    INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            pool_path        BLOB NOT NULL,
            displaced_path   TEXT,
            active           INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
            refreshed_at     INTEGER
        );

        CREATE TABLE move_operations (
            id                TEXT PRIMARY KEY,
            status            TEXT NOT NULL,
            destination_root  BLOB NOT NULL,
            created_at        INTEGER NOT NULL,
            completed_at      INTEGER,
            undone_at         INTEGER,
            error             TEXT
        );
        CREATE TABLE move_items (
            operation_id   TEXT NOT NULL REFERENCES move_operations(id) ON DELETE CASCADE,
            ordinal        INTEGER NOT NULL,
            image_id       INTEGER REFERENCES images(id) ON DELETE SET NULL,
            original_path  BLOB NOT NULL,
            destination    BLOB NOT NULL,
            blake3         TEXT NOT NULL,
            status         TEXT NOT NULL,
            error          TEXT,
            PRIMARY KEY(operation_id, ordinal)
        );",
    )?;
    Ok(())
}

fn seed_label_packs(connection: &Connection) -> Result<()> {
    let now = Utc::now().timestamp_millis();
    let packs = [
        (
            "mood",
            "mood",
            r#"["calm","dramatic","dreamy","energetic","melancholic","mysterious","serene","warm"]"#,
        ),
        (
            "subject",
            "subject",
            r#"["abstract","animals","architecture","city","landscape","nature","people","space","technology"]"#,
        ),
        (
            "style",
            "style",
            r#"["3d render","anime","digital art","illustration","minimalist","painting","photograph","pixel art"]"#,
        ),
    ];
    for (name, kind, labels) in packs {
        connection.execute(
            "INSERT OR IGNORE INTO label_packs(name, kind, labels_json, updated_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![name, kind, labels, now],
        )?;
    }
    Ok(())
}

fn source_by_path(connection: &Connection, path: &Path) -> Result<Option<SourceRoot>> {
    connection
        .query_row(
            "SELECT id, path, added_at FROM source_roots WHERE path = ?1",
            [path_bytes(path)],
            |row| {
                Ok(SourceRoot {
                    id: row.get(0)?,
                    path: path_from_bytes(row.get_ref(1)?.as_blob()?),
                    added_at: row.get(2)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
}

fn load_image(connection: &Connection, id: i64) -> Result<Option<ImageRecord>> {
    let mut record = connection
        .query_row(
            "SELECT i.id, i.source_id, i.path, i.size, i.modified_ns, i.blake3,
                    i.status, i.error, i.width, i.height, i.ratio, i.orientation,
                    i.common_ratio, i.dominant_hex, i.dominant_name, i.luminance,
                    i.saturation, i.contrast, i.light_dark, i.thumbnail_path,
                    EXISTS(SELECT 1 FROM favorites f WHERE f.image_id = i.id)
             FROM images i WHERE i.id = ?1",
            [id],
            |row| {
                let status: String = row.get(6)?;
                Ok((
                    ImageRecord {
                        id: row.get(0)?,
                        source_id: row.get(1)?,
                        path: path_from_bytes(row.get_ref(2)?.as_blob()?),
                        size: row.get::<_, i64>(3)? as u64,
                        modified_ns: row.get(4)?,
                        hash: row.get(5)?,
                        status: ImageStatus::parse(&status).map_err(|error| {
                            rusqlite::Error::FromSqlConversionFailure(
                                6,
                                rusqlite::types::Type::Text,
                                error.into(),
                            )
                        })?,
                        error: row.get(7)?,
                        width: row.get(8)?,
                        height: row.get(9)?,
                        ratio: row.get(10)?,
                        orientation: row.get(11)?,
                        common_ratio: row.get(12)?,
                        dominant_hex: row.get(13)?,
                        dominant_name: row.get(14)?,
                        luminance: row.get(15)?,
                        saturation: row.get(16)?,
                        contrast: row.get(17)?,
                        light_dark: row.get(18)?,
                        thumbnail_path: row.get_ref(19)?.as_blob_or_null()?.map(path_from_bytes),
                        palette: Vec::new(),
                        ai_estimates: Vec::new(),
                        favorite: row.get(20)?,
                        tags: Vec::new(),
                    },
                    id,
                ))
            },
        )
        .optional()?;
    if let Some((image, image_id)) = record.as_mut() {
        let mut statement = connection.prepare(
            "SELECT t.name FROM tags t
             JOIN image_tags it ON it.tag_id = t.id
             WHERE it.image_id = ?1 ORDER BY t.name COLLATE NOCASE",
        )?;
        image.tags = statement
            .query_map([*image_id], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut statement = connection.prepare(
            "SELECT rank, oklab_l, oklab_a, oklab_b, proportion, hex, name
             FROM image_palette WHERE image_id=?1 ORDER BY rank",
        )?;
        image.palette = statement
            .query_map([*image_id], |row| {
                Ok(PaletteColor {
                    oklab: Oklab {
                        l: row.get(1)?,
                        a: row.get(2)?,
                        b: row.get(3)?,
                    },
                    proportion: row.get(4)?,
                    hex: row.get(5)?,
                    name: row.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut statement = connection.prepare(
            "SELECT lp.name, ls.label, ls.score FROM label_scores ls
             JOIN label_packs lp ON lp.id=ls.pack_id WHERE ls.image_id=?1
             ORDER BY lp.name, ls.score DESC",
        )?;
        image.ai_estimates = statement
            .query_map([*image_id], |row| {
                Ok(AiEstimate {
                    pack: row.get(0)?,
                    label: row.get(1)?,
                    score: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
    }
    Ok(record.map(|(image, _)| image))
}

pub(crate) fn path_bytes(path: &Path) -> Vec<u8> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        path.as_os_str().as_bytes().to_vec()
    }
    #[cfg(not(unix))]
    {
        path.to_string_lossy().as_bytes().to_vec()
    }
}

pub(crate) fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt as _;
        PathBuf::from(OsString::from_vec(bytes.to_vec()))
    }
    #[cfg(not(unix))]
    {
        PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_and_reopens_schema() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("catalog.sqlite3");
        let database = Database::open(&path).expect("create database");
        drop(database);
        let database = Database::open(&path).expect("reopen database");
        let version: u32 = database
            .with_connection(|connection| {
                connection
                    .query_row("PRAGMA user_version", [], |row| row.get(0))
                    .map_err(Into::into)
            })
            .expect("version");
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn source_paths_round_trip() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("db")).expect("database");
        let images = directory.path().join("wallpapers");
        std::fs::create_dir(&images).expect("images directory");
        let inserted = database.add_source(&images).expect("add source");
        assert_eq!(inserted.path, images.canonicalize().expect("canonical"));
        assert_eq!(database.list_sources().expect("sources"), vec![inserted]);
    }
}
