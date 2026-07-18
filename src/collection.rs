use std::collections::HashMap;

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{OptionalExtension, params, params_from_iter};
use serde::{Deserialize, Serialize};

use crate::{
    AppPaths,
    db::{Database, ImageRecord},
    filter::{FILTER_VERSION, FilterSpecV1},
    model,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedCollection {
    pub id: i64,
    pub name: String,
    pub filter: FilterSpecV1,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SearchResult {
    #[serde(flatten)]
    pub image: ImageRecord,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_score: Option<f32>,
}

pub fn search(database: &Database, filter: &FilterSpecV1) -> Result<Vec<ImageRecord>> {
    anyhow::ensure!(
        filter.semantic_text.is_none(),
        "semantic filters require search_resolved so ROCm scores are applied"
    );
    search_metadata(database, filter)
}

fn search_metadata(database: &Database, filter: &FilterSpecV1) -> Result<Vec<ImageRecord>> {
    let compiled = filter.to_sql()?;
    let ids = database.with_connection(|connection| {
        let sql = format!(
            "SELECT i.id FROM images i WHERE {} ORDER BY i.path",
            compiled.sql
        );
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(params_from_iter(compiled.parameters), |row| row.get(0))?;
        rows.collect::<rusqlite::Result<Vec<i64>>>()
            .map_err(Into::into)
    })?;
    ids.into_iter()
        .map(|id| {
            database
                .get_image(id)?
                .with_context(|| format!("image {id} disappeared during search"))
        })
        .collect()
}

pub fn search_resolved(
    database: &Database,
    paths: &AppPaths,
    filter: &FilterSpecV1,
) -> Result<Vec<SearchResult>> {
    let mut images = search_metadata(database, filter)?;
    let mut semantic = HashMap::new();
    if let Some(text) = &filter.semantic_text {
        let _ = model::analyze_missing(database, paths)?;
        semantic.extend(model::semantic_scores(database, paths, text)?);
        let minimum = filter.semantic_min_score.unwrap_or(f32::NEG_INFINITY);
        images.retain(|image| {
            semantic
                .get(&image.id)
                .is_some_and(|score| *score >= minimum)
        });
        images.sort_by(|left, right| {
            semantic
                .get(&right.id)
                .unwrap_or(&f32::NEG_INFINITY)
                .total_cmp(semantic.get(&left.id).unwrap_or(&f32::NEG_INFINITY))
                .then_with(|| left.path.cmp(&right.path))
        });
    }
    Ok(images
        .into_iter()
        .map(|image| SearchResult {
            semantic_score: semantic.get(&image.id).copied(),
            image,
        })
        .collect())
}

pub fn save_collection(
    database: &Database,
    name: &str,
    filter: &FilterSpecV1,
) -> Result<SavedCollection> {
    filter.validate()?;
    let name = normalized_name(name)?;
    let json = serde_json::to_string(filter)?;
    let now = Utc::now().timestamp_millis();
    database.with_connection(|connection| {
        connection.execute(
            "INSERT INTO collections(name, filter_version, filter_json, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(name) DO UPDATE SET
                filter_version=excluded.filter_version,
                filter_json=excluded.filter_json,
                updated_at=excluded.updated_at",
            params![name, FILTER_VERSION, json, now],
        )?;
        get_collection_connection(connection, &name)?.context("saved collection disappeared")
    })
}

pub fn list_collections(database: &Database) -> Result<Vec<SavedCollection>> {
    database.with_connection(|connection| {
        let mut statement = connection.prepare(
            "SELECT id, name, filter_version, filter_json, created_at, updated_at
             FROM collections ORDER BY name COLLATE NOCASE",
        )?;
        let rows = statement.query_map([], collection_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to decode saved collections")
    })
}

pub fn get_collection(database: &Database, name: &str) -> Result<Option<SavedCollection>> {
    database.with_connection(|connection| get_collection_connection(connection, name))
}

pub fn delete_collection(database: &Database, name: &str) -> Result<bool> {
    database.with_transaction(|transaction| {
        let bound_display: Option<String> = transaction
            .query_row(
                "SELECT b.display FROM wpaperd_bindings b
                 JOIN collections c ON c.id=b.collection_id
                 WHERE c.name=?1 COLLATE NOCASE LIMIT 1",
                [name],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(display) = bound_display {
            anyhow::bail!("collection is bound to wpaperd display {display}; unbind it first");
        }
        Ok(transaction.execute(
            "DELETE FROM collections WHERE name = ?1 COLLATE NOCASE",
            [name],
        )? != 0)
    })
}

pub fn collection_images(
    database: &Database,
    paths: &AppPaths,
    name: &str,
) -> Result<Vec<ImageRecord>> {
    let collection =
        get_collection(database, name)?.with_context(|| format!("collection not found: {name}"))?;
    Ok(search_resolved(database, paths, &collection.filter)?
        .into_iter()
        .map(|result| result.image)
        .collect())
}

pub fn add_tag(database: &Database, image_ids: &[i64], tag: &str) -> Result<usize> {
    let tag = normalized_name(tag)?;
    database.with_transaction(|transaction| {
        transaction.execute("INSERT OR IGNORE INTO tags(name) VALUES (?1)", [&tag])?;
        let tag_id: i64 = transaction.query_row(
            "SELECT id FROM tags WHERE name = ?1 COLLATE NOCASE",
            [&tag],
            |row| row.get(0),
        )?;
        let mut changed = 0;
        for image_id in image_ids {
            changed += transaction.execute(
                "INSERT OR IGNORE INTO image_tags(image_id, tag_id) VALUES (?1, ?2)",
                params![image_id, tag_id],
            )?;
        }
        Ok(changed)
    })
}

pub fn remove_tag(database: &Database, image_ids: &[i64], tag: &str) -> Result<usize> {
    database.with_transaction(|transaction| {
        let mut changed = 0;
        for image_id in image_ids {
            changed += transaction.execute(
                "DELETE FROM image_tags WHERE image_id = ?1 AND tag_id IN
                 (SELECT id FROM tags WHERE name = ?2 COLLATE NOCASE)",
                params![image_id, tag],
            )?;
        }
        transaction.execute(
            "DELETE FROM tags WHERE NOT EXISTS
             (SELECT 1 FROM image_tags it WHERE it.tag_id = tags.id)",
            [],
        )?;
        Ok(changed)
    })
}

pub fn set_favorite(database: &Database, image_ids: &[i64], favorite: bool) -> Result<usize> {
    database.with_transaction(|transaction| {
        let mut changed = 0;
        for image_id in image_ids {
            changed += if favorite {
                transaction.execute(
                    "INSERT OR IGNORE INTO favorites(image_id, set_at) VALUES (?1, ?2)",
                    params![image_id, Utc::now().timestamp_millis()],
                )?
            } else {
                transaction.execute("DELETE FROM favorites WHERE image_id = ?1", [image_id])?
            };
        }
        Ok(changed)
    })
}

fn get_collection_connection(
    connection: &rusqlite::Connection,
    name: &str,
) -> Result<Option<SavedCollection>> {
    connection
        .query_row(
            "SELECT id, name, filter_version, filter_json, created_at, updated_at
             FROM collections WHERE name = ?1 COLLATE NOCASE",
            [name],
            collection_from_row,
        )
        .optional()
        .context("failed to decode saved collection")
}

fn collection_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SavedCollection> {
    let version: u32 = row.get(2)?;
    if version != FILTER_VERSION {
        return Err(rusqlite::Error::FromSqlConversionFailure(
            2,
            rusqlite::types::Type::Integer,
            format!("unsupported filter version {version}").into(),
        ));
    }
    let json: String = row.get(3)?;
    let filter = serde_json::from_str(&json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, error.into())
    })?;
    Ok(SavedCollection {
        id: row.get(0)?,
        name: row.get(1)?,
        filter,
        created_at: row.get(4)?,
        updated_at: row.get(5)?,
    })
}

fn normalized_name(value: &str) -> Result<String> {
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "name cannot be empty");
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "name cannot contain control characters"
    );
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collections_round_trip_and_update() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("db")).expect("database");
        let first = save_collection(&database, "Wide", &FilterSpecV1::default()).expect("save");
        let changed = FilterSpecV1 {
            min_width: Some(1920),
            ..FilterSpecV1::default()
        };
        let second = save_collection(&database, "wide", &changed).expect("update");
        assert_eq!(first.id, second.id);
        assert_eq!(second.filter.min_width, Some(1920));
        assert_eq!(list_collections(&database).expect("list").len(), 1);
        assert!(delete_collection(&database, "WIDE").expect("delete"));
    }
}
