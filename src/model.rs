use std::{
    fs,
    io::{IsTerminal as _, Read as _, Write as _},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use tempfile::{Builder, NamedTempFile};

use crate::{AppPaths, db::Database};

#[cfg(feature = "rocm")]
mod clip;

pub const MODEL_ID: &str = "openai/clip-vit-base-patch32";
pub const MODEL_REVISION: &str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268";
pub const EMBEDDING_DIMENSION: usize = 512;

const ARTIFACTS: [Artifact; 4] = [
    Artifact {
        name: "pytorch_model.bin",
        size: 605_247_071,
        sha256: "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
    },
    Artifact {
        name: "tokenizer.json",
        size: 2_224_041,
        sha256: "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
    },
    Artifact {
        name: "config.json",
        size: 4_186,
        sha256: "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
    },
    Artifact {
        name: "preprocessor_config.json",
        size: 316,
        sha256: "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    },
];

#[derive(Clone, Copy, Debug)]
struct Artifact {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InstallManifest {
    model: String,
    revision: String,
    artifacts: Vec<InstalledArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InstalledArtifact {
    name: String,
    size: u64,
    sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStatus {
    pub model: &'static str,
    pub revision: &'static str,
    pub directory: PathBuf,
    pub installed: bool,
    pub verified: bool,
    pub rocm_compiled: bool,
    pub problem: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AiReport {
    pub embedded: usize,
    pub scored: usize,
    pub failed: usize,
    pub failures: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LabelDefinition {
    pub name: String,
    #[serde(default)]
    pub prompts: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LabelPack {
    pub id: i64,
    pub name: String,
    pub kind: String,
    pub labels: Vec<LabelDefinition>,
    pub updated_at: i64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StoredLabel {
    Name(String),
    Detailed(LabelDefinition),
}

pub fn model_directory(paths: &AppPaths) -> PathBuf {
    paths
        .models_dir
        .join(format!("clip-vit-base-patch32-{}", &MODEL_REVISION[..12]))
}

pub fn status(paths: &AppPaths, verify: bool) -> ModelStatus {
    let directory = model_directory(paths);
    let installed = ARTIFACTS
        .iter()
        .all(|artifact| directory.join(artifact.name).is_file());
    if !installed {
        return ModelStatus {
            model: MODEL_ID,
            revision: MODEL_REVISION,
            directory,
            installed: false,
            verified: false,
            rocm_compiled: cfg!(feature = "rocm"),
            problem: Some("one or more pinned model files are missing".into()),
        };
    }
    let verification = if verify {
        verify_directory(&directory)
    } else {
        verify_manifest(&directory)
    };
    ModelStatus {
        model: MODEL_ID,
        revision: MODEL_REVISION,
        directory,
        installed,
        verified: verification.is_ok(),
        rocm_compiled: cfg!(feature = "rocm"),
        problem: verification.err().map(|error| format!("{error:#}")),
    }
}

pub fn install(paths: &AppPaths, assume_yes: bool) -> Result<ModelStatus> {
    install_with_progress(paths, assume_yes, |name, downloaded, total| {
        if downloaded == total || downloaded % (32 * 1024 * 1024) < 128 * 1024 {
            eprintln!("{name}: {downloaded}/{total} bytes");
        }
    })
}

pub fn install_with_progress(
    paths: &AppPaths,
    assume_yes: bool,
    mut progress: impl FnMut(&str, u64, u64),
) -> Result<ModelStatus> {
    let current = status(paths, true);
    if current.verified {
        return Ok(current);
    }
    if !assume_yes {
        if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
            bail!(
                "CLIP is not installed. Run `bgm model install --yes` before non-interactive AI work."
            );
        }
        eprint!(
            "Download the pinned OpenAI CLIP ViT-B/32 model (about 580 MiB) from Hugging Face? [y/N] "
        );
        std::io::stderr().flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            bail!("model installation cancelled");
        }
    }

    fs::create_dir_all(&paths.models_dir)?;
    let temporary = Builder::new()
        .prefix(".clip-download-")
        .tempdir_in(&paths.models_dir)?;
    for artifact in ARTIFACTS {
        download_artifact(temporary.path(), artifact, &mut progress)?;
    }
    write_install_manifest(temporary.path())?;
    verify_directory(temporary.path())?;

    let destination = model_directory(paths);
    let temporary_path = temporary.keep();
    if destination.exists() {
        rustix::fs::renameat_with(
            rustix::fs::CWD,
            &temporary_path,
            rustix::fs::CWD,
            &destination,
            rustix::fs::RenameFlags::EXCHANGE,
        )
        .map_err(errno_to_io)?;
        fs::remove_dir_all(&temporary_path)?;
    } else {
        rustix::fs::renameat_with(
            rustix::fs::CWD,
            &temporary_path,
            rustix::fs::CWD,
            &destination,
            rustix::fs::RenameFlags::NOREPLACE,
        )
        .map_err(errno_to_io)?;
    }
    fs::File::open(&paths.models_dir)?.sync_all()?;
    Ok(status(paths, false))
}

pub fn remove(paths: &AppPaths) -> Result<bool> {
    let directory = model_directory(paths);
    if !directory.exists() {
        return Ok(false);
    }
    fs::remove_dir_all(&directory)
        .with_context(|| format!("failed to remove {}", directory.display()))?;
    fs::File::open(&paths.models_dir)?.sync_all()?;
    Ok(true)
}

pub fn ensure_installed(paths: &AppPaths) -> Result<PathBuf> {
    let current = status(paths, false);
    if current.verified {
        return Ok(current.directory);
    }
    install(paths, false).map(|installed| installed.directory)
}

pub fn list_label_packs(database: &Database) -> Result<Vec<LabelPack>> {
    database.with_connection(|connection| {
        let mut statement = connection.prepare(
            "SELECT id, name, kind, labels_json, updated_at
             FROM label_packs ORDER BY name COLLATE NOCASE",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
            ))
        })?;
        rows.map(|row| {
            let (id, name, kind, json, updated_at) = row?;
            Ok(LabelPack {
                id,
                name,
                labels: decode_labels(&kind, &json)?,
                kind,
                updated_at,
            })
        })
        .collect()
    })
}

pub fn save_label_pack(
    database: &Database,
    name: &str,
    kind: &str,
    labels: &[LabelDefinition],
) -> Result<LabelPack> {
    let name = normalize_label_text(name, "pack name")?.to_lowercase();
    let kind = normalize_label_text(kind, "pack kind")?.to_lowercase();
    anyhow::ensure!(
        labels.len() >= 2,
        "a label pack needs at least two labels for ranked estimates"
    );
    let mut normalized = Vec::with_capacity(labels.len());
    let mut names = std::collections::HashSet::new();
    for label in labels {
        let label_name = normalize_label_text(&label.name, "label name")?;
        anyhow::ensure!(
            names.insert(label_name.to_lowercase()),
            "duplicate label in pack: {label_name}"
        );
        let prompts = if label.prompts.is_empty() {
            vec![default_prompt(&kind, &label_name)]
        } else {
            label
                .prompts
                .iter()
                .map(|prompt| normalize_label_text(prompt, "label prompt"))
                .collect::<Result<Vec<_>>>()?
        };
        normalized.push(LabelDefinition {
            name: label_name,
            prompts,
        });
    }
    let json = serde_json::to_string(&normalized)?;
    let now = Utc::now().timestamp_millis();
    let id = database.with_transaction(|transaction| {
        transaction.execute(
            "INSERT INTO label_packs(name, kind, labels_json, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(name) DO UPDATE SET kind=excluded.kind,
                labels_json=excluded.labels_json, updated_at=excluded.updated_at",
            params![name, kind, json, now],
        )?;
        let id: i64 = transaction.query_row(
            "SELECT id FROM label_packs WHERE name=?1 COLLATE NOCASE",
            [&name],
            |row| row.get(0),
        )?;
        // Embeddings remain valid. Only estimates derived from the old prompts
        // are invalidated, ready for `bgm label rescore`.
        transaction.execute("DELETE FROM label_scores WHERE pack_id=?1", [id])?;
        Ok(id)
    })?;
    list_label_packs(database)?
        .into_iter()
        .find(|pack| pack.id == id)
        .context("saved label pack disappeared")
}

pub fn delete_label_pack(database: &Database, name: &str) -> Result<bool> {
    anyhow::ensure!(
        !matches!(
            name.to_ascii_lowercase().as_str(),
            "mood" | "subject" | "style"
        ),
        "seeded label packs can be edited but not deleted"
    );
    database.with_connection(|connection| {
        Ok(connection.execute(
            "DELETE FROM label_packs WHERE name=?1 COLLATE NOCASE",
            [name],
        )? != 0)
    })
}

#[cfg(feature = "rocm")]
pub fn rescore_label_packs(
    database: &Database,
    paths: &AppPaths,
    pack: Option<&str>,
) -> Result<AiReport> {
    let directory = ensure_installed(paths)?;
    clip::rescore_label_packs(database, &directory, pack)
}

#[cfg(not(feature = "rocm"))]
pub fn rescore_label_packs(_: &Database, _: &AppPaths, _: Option<&str>) -> Result<AiReport> {
    bail!("label rescoring requires ROCm; CPU fallback is intentionally disabled")
}

#[cfg(feature = "rocm")]
pub fn analyze_missing(database: &Database, paths: &AppPaths) -> Result<AiReport> {
    let directory = ensure_installed(paths)?;
    clip::analyze_missing(database, &directory)
}

#[cfg(not(feature = "rocm"))]
pub fn analyze_missing(_: &Database, _: &AppPaths) -> Result<AiReport> {
    bail!("AI requires a bgm build with the `rocm` feature; CPU fallback is intentionally disabled")
}

#[cfg(feature = "rocm")]
pub fn semantic_scores(
    database: &Database,
    paths: &AppPaths,
    text: &str,
) -> Result<Vec<(i64, f32)>> {
    let directory = ensure_installed(paths)?;
    clip::semantic_scores(database, &directory, text)
}

#[cfg(not(feature = "rocm"))]
pub fn semantic_scores(_: &Database, _: &AppPaths, _: &str) -> Result<Vec<(i64, f32)>> {
    bail!("semantic search requires ROCm; CPU fallback is intentionally disabled")
}

pub fn model_key() -> String {
    format!("{MODEL_ID}@{MODEL_REVISION}")
}

fn decode_labels(kind: &str, json: &str) -> Result<Vec<LabelDefinition>> {
    let labels: Vec<StoredLabel> = serde_json::from_str(json)?;
    labels
        .into_iter()
        .map(|label| match label {
            StoredLabel::Name(name) => Ok(LabelDefinition {
                prompts: vec![default_prompt(kind, &name)],
                name,
            }),
            StoredLabel::Detailed(mut label) => {
                if label.prompts.is_empty() {
                    label.prompts.push(default_prompt(kind, &label.name));
                }
                Ok(label)
            }
        })
        .collect()
}

fn normalize_label_text(value: &str, field: &str) -> Result<String> {
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "{field} cannot be empty");
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "{field} cannot contain control characters"
    );
    Ok(value.to_owned())
}

pub(super) fn default_prompt(kind: &str, label: &str) -> String {
    match kind {
        "mood" => format!("a wallpaper with a {label} mood"),
        "subject" => format!("a wallpaper depicting {label}"),
        "style" => format!("a wallpaper in {label} style"),
        _ => format!("a wallpaper described as {label}"),
    }
}

#[cfg(any(feature = "rocm", test))]
pub(super) fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<_> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let sum: f32 = values.iter().sum();
    for value in &mut values {
        *value /= sum;
    }
    values
}

#[cfg(feature = "rocm")]
pub fn probe_rocm() -> Result<String> {
    clip::probe_rocm()
}

#[cfg(not(feature = "rocm"))]
pub fn probe_rocm() -> Result<String> {
    bail!("this bgm build does not include the required ROCm backend")
}

fn download_artifact(
    directory: &Path,
    artifact: Artifact,
    progress: &mut impl FnMut(&str, u64, u64),
) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{}",
        artifact.name
    );
    let response = ureq::get(&url)
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    let mut reader = response.into_reader();
    let mut temporary = NamedTempFile::new_in(directory)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 128 * 1024];
    let mut downloaded = 0_u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        temporary.write_all(&buffer[..count])?;
        hasher.update(&buffer[..count]);
        downloaded += count as u64;
        progress(artifact.name, downloaded, artifact.size);
    }
    temporary.as_file().sync_all()?;
    let digest = hex::encode(hasher.finalize());
    if downloaded != artifact.size {
        bail!(
            "{} has the wrong size: expected {}, received {downloaded}",
            artifact.name,
            artifact.size
        );
    }
    if digest != artifact.sha256 {
        bail!(
            "{} failed SHA-256 verification (expected {}, got {digest})",
            artifact.name,
            artifact.sha256
        );
    }
    temporary
        .persist(directory.join(artifact.name))
        .map_err(|error| error.error)?;
    Ok(())
}

fn verify_directory(directory: &Path) -> Result<()> {
    for artifact in ARTIFACTS {
        let path = directory.join(artifact.name);
        let metadata = fs::metadata(&path)
            .with_context(|| format!("missing model artifact {}", path.display()))?;
        if metadata.len() != artifact.size {
            bail!("model artifact has the wrong size: {}", path.display());
        }
        let digest = sha256_file(&path)?;
        if digest != artifact.sha256 {
            bail!("model artifact failed checksum: {}", path.display());
        }
    }
    Ok(())
}

fn verify_manifest(directory: &Path) -> Result<()> {
    let text = fs::read_to_string(directory.join("manifest.json"))?;
    let manifest: InstallManifest = serde_json::from_str(&text)?;
    if manifest.model != MODEL_ID || manifest.revision != MODEL_REVISION {
        bail!("installed model manifest does not match the pinned model");
    }
    for artifact in ARTIFACTS {
        let installed = manifest
            .artifacts
            .iter()
            .find(|installed| installed.name == artifact.name)
            .with_context(|| format!("{} is absent from the model manifest", artifact.name))?;
        if installed.size != artifact.size || installed.sha256 != artifact.sha256 {
            bail!("{} does not match the pinned model manifest", artifact.name);
        }
        if fs::metadata(directory.join(artifact.name))?.len() != artifact.size {
            bail!("{} has changed since installation", artifact.name);
        }
    }
    Ok(())
}

fn write_install_manifest(directory: &Path) -> Result<()> {
    let manifest = InstallManifest {
        model: MODEL_ID.into(),
        revision: MODEL_REVISION.into(),
        artifacts: ARTIFACTS
            .iter()
            .map(|artifact| InstalledArtifact {
                name: artifact.name.into(),
                size: artifact.size,
                sha256: artifact.sha256.into(),
            })
            .collect(),
    };
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("manifest.json"))?;
    serde_json::to_writer_pretty(&mut file, &manifest)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = std::io::BufReader::new(fs::File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn errno_to_io(error: rustix::io::Errno) -> std::io::Error {
    std::io::Error::from_raw_os_error(error.raw_os_error())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_artifacts_have_unique_names_and_sha256() {
        let mut names = std::collections::HashSet::new();
        for artifact in ARTIFACTS {
            assert!(names.insert(artifact.name));
            assert_eq!(artifact.sha256.len(), 64);
            assert!(artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn missing_model_reports_actionable_status() {
        let directory = tempfile::tempdir().expect("tempdir");
        let paths = AppPaths::from_xdg_roots(
            directory.path().join("cfg"),
            directory.path().join("data"),
            directory.path().join("cache"),
            directory.path().join("state"),
        );
        let current = status(&paths, false);
        assert!(!current.installed);
        assert!(current.problem.is_some());
    }

    #[test]
    fn label_scoring_is_a_ranked_probability_distribution() {
        let scores = softmax(&[1.0, 3.0, 2.0]);
        assert!(scores[1] > scores[2] && scores[2] > scores[0]);
        assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn custom_label_prompts_round_trip_without_touching_embeddings() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("catalog.sqlite3")).expect("database");
        let labels = vec![
            LabelDefinition {
                name: "neon".into(),
                prompts: vec!["a neon-lit wallpaper".into(), "bright neon lights".into()],
            },
            LabelDefinition {
                name: "muted".into(),
                prompts: Vec::new(),
            },
        ];
        let saved = save_label_pack(&database, "lighting", "custom", &labels).expect("save");
        assert_eq!(saved.labels[0].prompts.len(), 2);
        assert_eq!(saved.labels[1].prompts.len(), 1);
        assert!(saved.labels[1].prompts[0].contains("muted"));
        assert!(delete_label_pack(&database, "LIGHTING").expect("delete"));
        assert!(
            list_label_packs(&database)
                .expect("list")
                .iter()
                .all(|pack| pack.name != "lighting")
        );
    }
}
