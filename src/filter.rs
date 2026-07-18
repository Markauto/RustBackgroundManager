use anyhow::{Result, bail};
use rusqlite::types::Value;
use serde::{Deserialize, Serialize};

use crate::analysis::{LightDark, Oklab, Orientation, colour_distance, parse_hex_colour};

pub const FILTER_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilterSpecV1 {
    pub version: u32,
    pub source_ids: Vec<i64>,
    pub paths: Vec<String>,
    pub min_width: Option<u32>,
    pub max_width: Option<u32>,
    pub min_height: Option<u32>,
    pub max_height: Option<u32>,
    pub orientations: Vec<Orientation>,
    pub aspect_ratios: Vec<f64>,
    pub aspect_tolerance: f64,
    pub light_dark: Vec<LightDark>,
    pub min_luminance: Option<f32>,
    pub max_luminance: Option<f32>,
    pub dominant_colours: Vec<ColourFilter>,
    pub palette_colours: Vec<ColourFilter>,
    pub ai_labels: Vec<AiLabelFilter>,
    pub semantic_text: Option<String>,
    pub semantic_min_score: Option<f32>,
    pub tags: Vec<String>,
    pub favorite: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColourFilter {
    pub hex: String,
    pub max_distance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AiLabelFilter {
    pub pack: String,
    pub label: String,
    pub min_score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterCandidate<'a> {
    pub source_id: i64,
    pub path: &'a str,
    pub width: u32,
    pub height: u32,
    pub orientation: Orientation,
    pub ratio: f64,
    pub light_dark: LightDark,
    pub luminance: f32,
    pub dominant: Oklab,
    pub palette: &'a [Oklab],
    pub ai_scores: &'a [(&'a str, &'a str, f32)],
    pub semantic_score: Option<f32>,
    pub tags: &'a [&'a str],
    pub favorite: bool,
}

#[derive(Clone, Debug)]
pub struct SqlFilter {
    pub sql: String,
    pub parameters: Vec<Value>,
}

impl Default for FilterSpecV1 {
    fn default() -> Self {
        Self {
            version: FILTER_VERSION,
            source_ids: Vec::new(),
            paths: Vec::new(),
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            orientations: Vec::new(),
            aspect_ratios: Vec::new(),
            aspect_tolerance: 0.03,
            light_dark: Vec::new(),
            min_luminance: None,
            max_luminance: None,
            dominant_colours: Vec::new(),
            palette_colours: Vec::new(),
            ai_labels: Vec::new(),
            semantic_text: None,
            semantic_min_score: None,
            tags: Vec::new(),
            favorite: None,
        }
    }
}

impl FilterSpecV1 {
    pub fn validate(&self) -> Result<()> {
        if self.version != FILTER_VERSION {
            bail!("unsupported filter version {}", self.version);
        }
        validate_min_max(self.min_width, self.max_width, "width")?;
        validate_min_max(self.min_height, self.max_height, "height")?;
        if !(0.0..=1.0).contains(&self.aspect_tolerance) {
            bail!("aspect_tolerance must be between 0 and 1");
        }
        if self
            .aspect_ratios
            .iter()
            .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
        {
            bail!("aspect ratios must be finite and positive");
        }
        if let (Some(minimum), Some(maximum)) = (self.min_luminance, self.max_luminance)
            && minimum > maximum
        {
            bail!("minimum luminance cannot exceed maximum luminance");
        }
        if self
            .min_luminance
            .into_iter()
            .chain(self.max_luminance)
            .any(|value| !(0.0..=1.0).contains(&value))
        {
            bail!("luminance bounds must be between 0 and 1");
        }
        for colour in self.dominant_colours.iter().chain(&self.palette_colours) {
            parse_hex_colour(&colour.hex)?;
            if !colour.max_distance.is_finite() || colour.max_distance < 0.0 {
                bail!("colour distance must be finite and non-negative");
            }
        }
        for label in &self.ai_labels {
            if label.pack.trim().is_empty() || label.label.trim().is_empty() {
                bail!("AI pack and label names cannot be empty");
            }
            if !(0.0..=1.0).contains(&label.min_score) {
                bail!("AI score thresholds must be between 0 and 1");
            }
        }
        if self
            .semantic_min_score
            .is_some_and(|score| !(-1.0..=1.0).contains(&score))
        {
            bail!("semantic score threshold must be between -1 and 1");
        }
        if self.semantic_min_score.is_some() && self.semantic_text.is_none() {
            bail!("semantic score threshold requires semantic text");
        }
        if self
            .semantic_text
            .as_ref()
            .is_some_and(|text| text.trim().is_empty())
        {
            bail!("semantic text cannot be empty");
        }
        Ok(())
    }

    /// Compile all database-addressable facets. Semantic text is applied by the
    /// model layer after it has produced candidate scores.
    pub fn to_sql(&self) -> Result<SqlFilter> {
        self.validate()?;
        let mut clauses = vec!["i.status = 'ready'".to_owned()];
        let mut parameters = Vec::new();

        add_or_values(
            &mut clauses,
            &mut parameters,
            "i.source_id",
            &self.source_ids,
        );
        if !self.paths.is_empty() {
            let path_clauses = self
                .paths
                .iter()
                .map(|path| {
                    parameters.push(Value::Text(format!("%{}%", escape_like(path))));
                    "lower(CAST(i.path AS TEXT)) LIKE ? ESCAPE '\\'".to_owned()
                })
                .collect::<Vec<_>>();
            clauses.push(format!("({})", path_clauses.join(" OR ")));
        }
        add_bound(
            &mut clauses,
            &mut parameters,
            "i.width",
            ">=",
            self.min_width,
        );
        add_bound(
            &mut clauses,
            &mut parameters,
            "i.width",
            "<=",
            self.max_width,
        );
        add_bound(
            &mut clauses,
            &mut parameters,
            "i.height",
            ">=",
            self.min_height,
        );
        add_bound(
            &mut clauses,
            &mut parameters,
            "i.height",
            "<=",
            self.max_height,
        );
        if !self.orientations.is_empty() {
            let values = self
                .orientations
                .iter()
                .map(|orientation| orientation.as_str().to_owned())
                .collect::<Vec<_>>();
            add_or_values(&mut clauses, &mut parameters, "i.orientation", &values);
        }
        if !self.aspect_ratios.is_empty() {
            let mut ratios = Vec::with_capacity(self.aspect_ratios.len());
            for ratio in &self.aspect_ratios {
                parameters.push(Value::Real(*ratio));
                parameters.push(Value::Real(*ratio));
                parameters.push(Value::Real(self.aspect_tolerance));
                ratios.push("(abs(i.ratio - ?) / ?) <= ?".to_owned());
            }
            clauses.push(format!("({})", ratios.join(" OR ")));
        }
        if !self.light_dark.is_empty() {
            let values = self
                .light_dark
                .iter()
                .map(|classification| classification.as_str().to_owned())
                .collect::<Vec<_>>();
            add_or_values(&mut clauses, &mut parameters, "i.light_dark", &values);
        }
        add_bound(
            &mut clauses,
            &mut parameters,
            "i.luminance",
            ">=",
            self.min_luminance,
        );
        add_bound(
            &mut clauses,
            &mut parameters,
            "i.luminance",
            "<=",
            self.max_luminance,
        );
        add_colour_facet(&mut clauses, &mut parameters, &self.dominant_colours, true)?;
        add_colour_facet(&mut clauses, &mut parameters, &self.palette_colours, false)?;
        if !self.ai_labels.is_empty() {
            let predicates = self
                .ai_labels
                .iter()
                .map(|label| {
                    parameters.push(Value::Text(label.pack.clone()));
                    parameters.push(Value::Text(label.label.clone()));
                    parameters.push(Value::Real(f64::from(label.min_score)));
                    "EXISTS (SELECT 1 FROM label_scores ls JOIN label_packs lp ON lp.id = ls.pack_id \
                     WHERE ls.image_id = i.id AND lp.name = ? COLLATE NOCASE
                       AND ls.label = ? COLLATE NOCASE AND ls.score >= ?)"
                        .to_owned()
                })
                .collect::<Vec<_>>();
            clauses.push(format!("({})", predicates.join(" OR ")));
        }
        if !self.tags.is_empty() {
            let placeholders = vec!["?"; self.tags.len()].join(", ");
            for tag in &self.tags {
                parameters.push(Value::Text(tag.clone()));
            }
            clauses.push(format!(
                "EXISTS (SELECT 1 FROM image_tags it JOIN tags t ON t.id = it.tag_id \
                 WHERE it.image_id = i.id AND t.name COLLATE NOCASE IN ({placeholders}))"
            ));
        }
        if let Some(favorite) = self.favorite {
            clauses.push(if favorite {
                "EXISTS (SELECT 1 FROM favorites f WHERE f.image_id = i.id)".to_owned()
            } else {
                "NOT EXISTS (SELECT 1 FROM favorites f WHERE f.image_id = i.id)".to_owned()
            });
        }

        Ok(SqlFilter {
            sql: clauses.join(" AND "),
            parameters,
        })
    }

    pub fn matches(&self, candidate: &FilterCandidate<'_>) -> Result<bool> {
        self.validate()?;
        let facets = [
            self.source_ids.is_empty() || self.source_ids.contains(&candidate.source_id),
            self.paths.is_empty()
                || self
                    .paths
                    .iter()
                    .any(|part| candidate.path.to_lowercase().contains(&part.to_lowercase())),
            self.min_width
                .is_none_or(|minimum| candidate.width >= minimum),
            self.max_width
                .is_none_or(|maximum| candidate.width <= maximum),
            self.min_height
                .is_none_or(|minimum| candidate.height >= minimum),
            self.max_height
                .is_none_or(|maximum| candidate.height <= maximum),
            self.orientations.is_empty() || self.orientations.contains(&candidate.orientation),
            self.aspect_ratios.is_empty()
                || self.aspect_ratios.iter().any(|ratio| {
                    ((candidate.ratio - ratio).abs() / ratio) <= self.aspect_tolerance
                }),
            self.light_dark.is_empty() || self.light_dark.contains(&candidate.light_dark),
            self.min_luminance
                .is_none_or(|minimum| candidate.luminance >= minimum),
            self.max_luminance
                .is_none_or(|maximum| candidate.luminance <= maximum),
            colour_matches(&self.dominant_colours, &[candidate.dominant])?,
            colour_matches(&self.palette_colours, candidate.palette)?,
            self.ai_labels.is_empty()
                || self.ai_labels.iter().any(|wanted| {
                    candidate.ai_scores.iter().any(|(pack, label, score)| {
                        pack.eq_ignore_ascii_case(&wanted.pack)
                            && label.eq_ignore_ascii_case(&wanted.label)
                            && *score >= wanted.min_score
                    })
                }),
            self.semantic_text.is_none()
                || candidate.semantic_score.is_some_and(|score| {
                    self.semantic_min_score
                        .is_none_or(|minimum| score >= minimum)
                }),
            self.tags.is_empty()
                || self.tags.iter().any(|wanted| {
                    candidate
                        .tags
                        .iter()
                        .any(|tag| tag.eq_ignore_ascii_case(wanted))
                }),
            self.favorite
                .is_none_or(|wanted| candidate.favorite == wanted),
        ];
        Ok(facets.into_iter().all(|matches| matches))
    }
}

fn validate_min_max(minimum: Option<u32>, maximum: Option<u32>, name: &str) -> Result<()> {
    if let (Some(minimum), Some(maximum)) = (minimum, maximum)
        && minimum > maximum
    {
        bail!("minimum {name} cannot exceed maximum {name}");
    }
    Ok(())
}

fn escape_like(value: &str) -> String {
    value
        .to_lowercase()
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn add_or_values<T: Clone + Into<Value>>(
    clauses: &mut Vec<String>,
    parameters: &mut Vec<Value>,
    column: &str,
    values: &[T],
) {
    if values.is_empty() {
        return;
    }
    let placeholders = vec!["?"; values.len()].join(", ");
    parameters.extend(values.iter().cloned().map(Into::into));
    clauses.push(format!("{column} IN ({placeholders})"));
}

fn add_bound<T: Into<Value>>(
    clauses: &mut Vec<String>,
    parameters: &mut Vec<Value>,
    column: &str,
    comparison: &str,
    value: Option<T>,
) {
    if let Some(value) = value {
        clauses.push(format!("{column} {comparison} ?"));
        parameters.push(value.into());
    }
}

fn add_colour_facet(
    clauses: &mut Vec<String>,
    parameters: &mut Vec<Value>,
    colours: &[ColourFilter],
    dominant_only: bool,
) -> Result<()> {
    if colours.is_empty() {
        return Ok(());
    }
    let predicates = colours
        .iter()
        .map(|filter| {
            let colour = parse_hex_colour(&filter.hex)?;
            parameters.push(Value::Real(f64::from(colour.l)));
            parameters.push(Value::Real(f64::from(colour.l)));
            parameters.push(Value::Real(f64::from(colour.a)));
            parameters.push(Value::Real(f64::from(colour.a)));
            parameters.push(Value::Real(f64::from(colour.b)));
            parameters.push(Value::Real(f64::from(colour.b)));
            parameters.push(Value::Real(f64::from(filter.max_distance.powi(2))));
            let rank = if dominant_only { "AND p.rank = 0" } else { "" };
            Ok(format!(
                "EXISTS (SELECT 1 FROM image_palette p WHERE p.image_id = i.id {rank} \
                 AND ((p.oklab_l - ?) * (p.oklab_l - ?) + \
                          (p.oklab_a - ?) * (p.oklab_a - ?) + \
                          (p.oklab_b - ?) * (p.oklab_b - ?)) <= ?)"
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    clauses.push(format!("({})", predicates.join(" OR ")));
    Ok(())
}

fn colour_matches(filters: &[ColourFilter], candidates: &[Oklab]) -> Result<bool> {
    if filters.is_empty() {
        return Ok(true);
    }
    for filter in filters {
        let target = parse_hex_colour(&filter.hex)?;
        if candidates
            .iter()
            .any(|candidate| colour_distance(target, *candidate) <= filter.max_distance)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate<'a>() -> FilterCandidate<'a> {
        FilterCandidate {
            source_id: 7,
            path: "/wallpapers/blue-mountain.jpg",
            width: 1920,
            height: 1080,
            orientation: Orientation::Landscape,
            ratio: 16.0 / 9.0,
            light_dark: LightDark::Dark,
            luminance: 0.3,
            dominant: parse_hex_colour("#203060").expect("colour"),
            palette: &[],
            ai_scores: &[("mood", "calm", 0.8), ("subject", "nature", 0.7)],
            semantic_score: None,
            tags: &["desktop", "blue"],
            favorite: true,
        }
    }

    #[test]
    fn facets_are_and_values_are_or() {
        let filter = FilterSpecV1 {
            source_ids: vec![2, 7],
            orientations: vec![Orientation::Portrait, Orientation::Landscape],
            tags: vec!["red".into(), "blue".into()],
            ai_labels: vec![AiLabelFilter {
                pack: "mood".into(),
                label: "calm".into(),
                min_score: 0.75,
            }],
            favorite: Some(true),
            ..FilterSpecV1::default()
        };
        assert!(filter.matches(&candidate()).expect("filter"));

        let mut wrong_favorite = filter;
        wrong_favorite.favorite = Some(false);
        assert!(!wrong_favorite.matches(&candidate()).expect("filter"));
    }

    #[test]
    fn serializes_with_explicit_version() {
        let json = serde_json::to_string(&FilterSpecV1::default()).expect("serialize");
        assert!(json.contains("\"version\":1"));
        let decoded: FilterSpecV1 = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.version, FILTER_VERSION);
    }

    #[test]
    fn semantic_facet_uses_the_resolved_cosine_score() {
        let filter = FilterSpecV1 {
            semantic_text: Some("misty forest".into()),
            semantic_min_score: Some(0.25),
            ..FilterSpecV1::default()
        };
        let mut candidate = candidate();
        candidate.semantic_score = Some(0.30);
        assert!(filter.matches(&candidate).expect("matching score"));
        candidate.semantic_score = Some(0.20);
        assert!(!filter.matches(&candidate).expect("low score"));
    }

    #[test]
    fn compiled_sql_keeps_ratio_and_colour_parameters_in_order() {
        let connection = rusqlite::Connection::open_in_memory().expect("database");
        connection
            .execute_batch(
                "CREATE TABLE images(
                    id INTEGER PRIMARY KEY, status TEXT, source_id INTEGER,
                    path BLOB, width INTEGER, ratio REAL
                 );
                 CREATE TABLE image_palette(
                    image_id INTEGER, rank INTEGER, oklab_l REAL,
                    oklab_a REAL, oklab_b REAL
                 );",
            )
            .expect("schema");
        connection
            .execute(
                "INSERT INTO images(id, status, source_id, path, width, ratio)
                 VALUES (1, 'ready', 7, '/wall/blue.png', 1920, ?1)",
                [16.0 / 9.0],
            )
            .expect("image");
        let blue = parse_hex_colour("#203060").expect("blue");
        connection
            .execute(
                "INSERT INTO image_palette(image_id, rank, oklab_l, oklab_a, oklab_b)
                 VALUES (1, 0, ?1, ?2, ?3)",
                rusqlite::params![blue.l, blue.a, blue.b],
            )
            .expect("palette");
        let colour = ColourFilter {
            hex: "#203060".into(),
            max_distance: 0.001,
        };
        let filter = FilterSpecV1 {
            source_ids: vec![7],
            min_width: Some(1_000),
            aspect_ratios: vec![4.0 / 3.0, 16.0 / 9.0],
            dominant_colours: vec![colour.clone()],
            palette_colours: vec![colour],
            ..FilterSpecV1::default()
        };
        let compiled = filter.to_sql().expect("compile");
        let sql = format!("SELECT COUNT(*) FROM images i WHERE {}", compiled.sql);
        let count: i64 = connection
            .query_row(
                &sql,
                rusqlite::params_from_iter(compiled.parameters),
                |row| row.get(0),
            )
            .expect("execute filter");
        assert_eq!(count, 1);
    }

    #[test]
    fn path_filter_treats_like_metacharacters_literally() {
        let connection = rusqlite::Connection::open_in_memory().expect("database");
        connection
            .execute_batch(
                "CREATE TABLE images(id INTEGER PRIMARY KEY, status TEXT, path BLOB);
                 INSERT INTO images VALUES (1, 'ready', '/wall/100%real.png');
                 INSERT INTO images VALUES (2, 'ready', '/wall/1000real.png');",
            )
            .expect("schema");
        let filter = FilterSpecV1 {
            paths: vec!["100%real".into()],
            ..FilterSpecV1::default()
        };
        let compiled = filter.to_sql().expect("compile");
        let sql = format!("SELECT id FROM images i WHERE {}", compiled.sql);
        let ids = connection
            .prepare(&sql)
            .expect("prepare")
            .query_map(rusqlite::params_from_iter(compiled.parameters), |row| {
                row.get::<_, i64>(0)
            })
            .expect("query")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("ids");
        assert_eq!(ids, vec![1]);
    }
}
