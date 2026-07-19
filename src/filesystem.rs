use std::{
    fs::File,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result};

pub(crate) fn blake3_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    hasher
        .update_reader(file)
        .with_context(|| format!("failed to hash {}", path.display()))?;
    Ok(hasher.finalize().to_hex().to_string())
}

pub(crate) fn absolute_lexical(path: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(path)
        .with_context(|| format!("failed to resolve {}", path.display()))?;
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    Ok(normalized)
}

/// Return the file that an atomic replacement should target while preserving
/// an existing symlink at the user-facing path. Missing paths are returned
/// unchanged; dangling symlinks are rejected instead of being overwritten.
pub(crate) fn resolve_file_target(path: &Path) -> Result<PathBuf> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => path
            .canonicalize()
            .with_context(|| format!("failed to resolve symlink {}", path.display())),
        Ok(_) => Ok(path.to_owned()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(path.to_owned()),
        Err(error) => Err(error).with_context(|| format!("failed to inspect {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_hash_matches_in_memory_hash() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("input");
        let bytes = b"background-manager\0hash-fixture";
        std::fs::write(&path, bytes).expect("fixture");

        assert_eq!(
            blake3_file(&path).expect("file hash"),
            blake3::hash(bytes).to_hex().to_string()
        );
    }

    #[test]
    fn absolute_paths_are_lexically_normalized_without_filesystem_access() {
        let path = absolute_lexical(Path::new("missing/../wallpapers/./image.jpg"))
            .expect("absolute path");
        assert!(path.is_absolute());
        assert!(path.ends_with("wallpapers/image.jpg"));
        assert!(
            !path
                .components()
                .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolves_live_symlinks_and_rejects_dangling_ones() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("tempdir");
        let target = directory.path().join("target");
        let link = directory.path().join("link");
        std::fs::write(&target, b"target").expect("target");
        symlink(&target, &link).expect("symlink");
        assert_eq!(
            resolve_file_target(&link).expect("resolved target"),
            target.canonicalize().expect("canonical target")
        );

        let dangling = directory.path().join("dangling");
        symlink(directory.path().join("missing"), &dangling).expect("dangling symlink");
        assert!(resolve_file_target(&dangling).is_err());
    }
}
