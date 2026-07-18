use std::{env, path::PathBuf};

use anyhow::{Context, Result, bail};

/// Every path owned or edited by bgm.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub database: PathBuf,
    pub models_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub thumbnails_dir: PathBuf,
    pub state_dir: PathBuf,
    pub pools_dir: PathBuf,
    pub manifests_dir: PathBuf,
    pub backups_dir: PathBuf,
    pub wpaperd_config: PathBuf,
}

impl AppPaths {
    /// Resolve paths from the XDG environment, with freedesktop defaults.
    pub fn discover() -> Result<Self> {
        let home = env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .context("HOME is not set; cannot determine XDG directories")?;

        let config_home = xdg_dir("XDG_CONFIG_HOME", &home, ".config")?;
        let data_home = xdg_dir("XDG_DATA_HOME", &home, ".local/share")?;
        let cache_home = xdg_dir("XDG_CACHE_HOME", &home, ".cache")?;
        let state_home = xdg_dir("XDG_STATE_HOME", &home, ".local/state")?;
        Ok(Self::from_xdg_roots(
            config_home,
            data_home,
            cache_home,
            state_home,
        ))
    }

    pub fn from_xdg_roots(
        config_home: PathBuf,
        data_home: PathBuf,
        cache_home: PathBuf,
        state_home: PathBuf,
    ) -> Self {
        let config_dir = config_home.join("bgm");
        let data_dir = data_home.join("bgm");
        let cache_dir = cache_home.join("bgm");
        let state_dir = state_home.join("bgm");
        Self {
            config_file: config_dir.join("config.toml"),
            database: data_dir.join("catalog.sqlite3"),
            models_dir: data_dir.join("models"),
            thumbnails_dir: cache_dir.join("thumbnails"),
            pools_dir: state_dir.join("wpaperd"),
            manifests_dir: state_dir.join("moves"),
            backups_dir: state_dir.join("backups"),
            wpaperd_config: config_home.join("wpaperd/config.toml"),
            config_dir,
            data_dir,
            cache_dir,
            state_dir,
        }
    }

    pub fn ensure_owned_dirs(&self) -> Result<()> {
        for path in [
            &self.config_dir,
            &self.data_dir,
            &self.models_dir,
            &self.cache_dir,
            &self.thumbnails_dir,
            &self.state_dir,
            &self.pools_dir,
            &self.manifests_dir,
            &self.backups_dir,
        ] {
            std::fs::create_dir_all(path)
                .with_context(|| format!("failed to create {}", path.display()))?;
        }
        Ok(())
    }
}

fn xdg_dir(variable: &str, home: &std::path::Path, fallback: &str) -> Result<PathBuf> {
    match env::var_os(variable).filter(|value| !value.is_empty()) {
        Some(value) => {
            let path = PathBuf::from(value);
            if !path.is_absolute() {
                bail!("{variable} must be an absolute path")
            }
            Ok(path)
        }
        None => Ok(home.join(fallback)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_expected_xdg_layout() {
        let paths = AppPaths::from_xdg_roots(
            "/cfg".into(),
            "/data".into(),
            "/cache".into(),
            "/state".into(),
        );
        assert_eq!(paths.config_file, PathBuf::from("/cfg/bgm/config.toml"));
        assert_eq!(paths.database, PathBuf::from("/data/bgm/catalog.sqlite3"));
        assert_eq!(paths.thumbnails_dir, PathBuf::from("/cache/bgm/thumbnails"));
        assert_eq!(paths.pools_dir, PathBuf::from("/state/bgm/wpaperd"));
        assert_eq!(
            paths.wpaperd_config,
            PathBuf::from("/cfg/wpaperd/config.toml")
        );
    }
}
