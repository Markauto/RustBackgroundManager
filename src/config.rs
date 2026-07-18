use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::model::{MODEL_ID, MODEL_REVISION};

pub const CONFIG_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub version: u32,
    pub import: ImportConfig,
    pub analysis: AnalysisConfig,
    pub ai: AiConfig,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImportConfig {
    pub min_width: Option<u32>,
    pub max_width: Option<u32>,
    pub min_height: Option<u32>,
    pub max_height: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    pub palette_colors: u8,
    pub common_ratio_tolerance: f32,
    pub dark_threshold: f32,
    pub thumbnail_long_edge: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiConfig {
    pub enabled: bool,
    pub model: String,
    pub revision: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            import: ImportConfig::default(),
            analysis: AnalysisConfig::default(),
            ai: AiConfig::default(),
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            palette_colors: 5,
            common_ratio_tolerance: 0.03,
            dark_threshold: 0.5,
            thumbnail_long_edge: 512,
        }
    }
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: MODEL_ID.into(),
            // A revision is deliberately persisted so an installed model never floats.
            revision: MODEL_REVISION.into(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Self::from_toml(&text)
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let mut value: toml::Value = toml::from_str(text).context("invalid bgm configuration")?;
        migrate(&mut value)?;
        let config: Self = value
            .try_into()
            .context("invalid bgm configuration values")?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let parent = path
            .parent()
            .context("configuration path has no parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let mut temporary = NamedTempFile::new_in(parent)
            .with_context(|| format!("failed to create temporary file in {}", parent.display()))?;
        let text = toml::to_string_pretty(self).context("failed to serialize configuration")?;
        use std::io::Write as _;
        temporary.write_all(text.as_bytes())?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(path)
            .map_err(|error| error.error)
            .with_context(|| format!("failed to replace {}", path.display()))?;
        sync_parent(parent)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != CONFIG_VERSION {
            bail!(
                "unsupported configuration version {}; expected {CONFIG_VERSION}",
                self.version
            );
        }
        validate_bounds(&self.import)?;
        if self.analysis.palette_colors == 0 || self.analysis.palette_colors > 16 {
            bail!("analysis.palette_colors must be between 1 and 16");
        }
        if !(0.0..=1.0).contains(&self.analysis.common_ratio_tolerance) {
            bail!("analysis.common_ratio_tolerance must be between 0 and 1");
        }
        if !(0.0..=1.0).contains(&self.analysis.dark_threshold) {
            bail!("analysis.dark_threshold must be between 0 and 1");
        }
        if self.analysis.thumbnail_long_edge < 64 {
            bail!("analysis.thumbnail_long_edge must be at least 64");
        }
        if self.ai.model != MODEL_ID || self.ai.revision != MODEL_REVISION {
            bail!(
                "v1 only supports pinned {MODEL_ID}@{MODEL_REVISION}; custom model revisions are not supported"
            );
        }
        Ok(())
    }

    pub fn set(&mut self, key: &str, raw_value: &str) -> Result<()> {
        match key {
            "import.min_width" => self.import.min_width = optional_u32(raw_value)?,
            "import.max_width" => self.import.max_width = optional_u32(raw_value)?,
            "import.min_height" => self.import.min_height = optional_u32(raw_value)?,
            "import.max_height" => self.import.max_height = optional_u32(raw_value)?,
            "analysis.palette_colors" => {
                self.analysis.palette_colors = raw_value.parse().context("expected an integer")?;
            }
            "analysis.common_ratio_tolerance" => {
                self.analysis.common_ratio_tolerance =
                    raw_value.parse().context("expected a decimal value")?;
            }
            "analysis.dark_threshold" => {
                self.analysis.dark_threshold =
                    raw_value.parse().context("expected a decimal value")?;
            }
            "analysis.thumbnail_long_edge" => {
                self.analysis.thumbnail_long_edge =
                    raw_value.parse().context("expected an integer")?;
            }
            "ai.enabled" => {
                self.ai.enabled = raw_value.parse().context("expected true or false")?;
            }
            "ai.model" => self.ai.model = raw_value.to_owned(),
            "ai.revision" => self.ai.revision = raw_value.to_owned(),
            _ => bail!("unknown configuration key: {key}"),
        }
        self.validate()
    }
}

fn optional_u32(value: &str) -> Result<Option<u32>> {
    if matches!(value, "none" | "null" | "off") {
        Ok(None)
    } else {
        Ok(Some(
            value.parse().context("expected an integer or 'none'")?,
        ))
    }
}

fn validate_bounds(bounds: &ImportConfig) -> Result<()> {
    if let (Some(min), Some(max)) = (bounds.min_width, bounds.max_width)
        && min > max
    {
        bail!("import.min_width cannot exceed import.max_width");
    }
    if let (Some(min), Some(max)) = (bounds.min_height, bounds.max_height)
        && min > max
    {
        bail!("import.min_height cannot exceed import.max_height");
    }
    Ok(())
}

fn migrate(value: &mut toml::Value) -> Result<()> {
    let table = value
        .as_table_mut()
        .context("configuration root must be a TOML table")?;
    let version = table
        .get("version")
        .and_then(toml::Value::as_integer)
        .unwrap_or(0);
    match version {
        0 => {
            // Pre-release configurations used flat threshold keys.
            let dark = table.remove("dark_threshold");
            let tolerance = table.remove("ratio_tolerance");
            let analysis = table
                .entry("analysis")
                .or_insert_with(|| toml::Value::Table(toml::Table::new()))
                .as_table_mut()
                .context("analysis must be a table")?;
            if let Some(dark) = dark {
                analysis.entry("dark_threshold").or_insert(dark);
            }
            if let Some(tolerance) = tolerance {
                analysis
                    .entry("common_ratio_tolerance")
                    .or_insert(tolerance);
            }
            table.insert("version".into(), toml::Value::Integer(1));
        }
        1 => {}
        other => bail!("configuration version {other} is newer than this bgm supports"),
    }
    Ok(())
}

#[cfg(unix)]
fn sync_parent(parent: &Path) -> Result<()> {
    std::fs::File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent(_: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_specification() {
        let config = Config::default();
        assert_eq!(config.import, ImportConfig::default());
        assert_eq!(config.analysis.palette_colors, 5);
        assert_eq!(config.analysis.common_ratio_tolerance, 0.03);
        assert_eq!(config.analysis.dark_threshold, 0.5);
    }

    #[test]
    fn migrates_flat_pre_release_config() {
        let config = Config::from_toml("dark_threshold = 0.42\nratio_tolerance = 0.05\n")
            .expect("migration should succeed");
        assert_eq!(config.version, 1);
        assert_eq!(config.analysis.dark_threshold, 0.42);
        assert_eq!(config.analysis.common_ratio_tolerance, 0.05);
    }

    #[test]
    fn rejects_inverted_bounds() {
        let mut config = Config::default();
        config.import.min_width = Some(2000);
        config.import.max_width = Some(1000);
        assert!(config.validate().is_err());
    }
}
