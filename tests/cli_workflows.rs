use std::{fs, path::Path};

use assert_cmd::Command;
use image::{Rgb, RgbImage};
use serde_json::Value;

struct Fixture {
    _directory: tempfile::TempDir,
    root: std::path::PathBuf,
    source: std::path::PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().expect("tempdir");
        let root = directory.path().to_owned();
        let source = root.join("images");
        fs::create_dir(&source).expect("source");
        Self {
            _directory: directory,
            root,
            source,
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(assert_cmd::cargo::cargo_bin!("bgm"));
        command
            .env("HOME", &self.root)
            .env("XDG_CONFIG_HOME", self.root.join("config"))
            .env("XDG_DATA_HOME", self.root.join("data"))
            .env("XDG_CACHE_HOME", self.root.join("cache"))
            .env("XDG_STATE_HOME", self.root.join("state"));
        command
    }

    fn add_and_scan(&self, image: &Path) -> i64 {
        RgbImage::from_pixel(64, 32, Rgb([20, 60, 120]))
            .save(image)
            .expect("image");
        self.command()
            .args(["source", "add"])
            .arg(&self.source)
            .assert()
            .success();
        let output = self
            .command()
            .args(["scan", "--no-ai", "--json"])
            .output()
            .expect("scan");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).expect("scan json");
        assert_eq!(report["scan"]["analyzed"], 1);

        let output = self
            .command()
            .args(["search", "--json"])
            .output()
            .expect("search");
        assert!(output.status.success());
        let hits: Value = serde_json::from_slice(&output.stdout).expect("search json");
        hits[0]["id"].as_i64().expect("image id")
    }
}

#[test]
fn catalog_tags_favorites_and_collections_emit_valid_json() {
    let fixture = Fixture::new();
    let image = fixture.source.join("blue.png");
    let id = fixture.add_and_scan(&image);

    fixture
        .command()
        .args(["tag", "add", "desktop", &id.to_string()])
        .assert()
        .success();
    fixture
        .command()
        .args(["favorite", "set", &id.to_string()])
        .assert()
        .success();
    let output = fixture
        .command()
        .args(["search", "--tag", "desktop", "--favorite", "--json"])
        .output()
        .expect("filtered search");
    assert!(output.status.success());
    let hits: Value = serde_json::from_slice(&output.stdout).expect("hits");
    assert_eq!(hits.as_array().map(Vec::len), Some(1));
    assert_eq!(hits[0]["favorite"], true);
    assert_eq!(hits[0]["tags"][0], "desktop");

    fixture
        .command()
        .args(["collection", "save", "wide", "--min-width", "64"])
        .assert()
        .success();
    let output = fixture
        .command()
        .args(["collection", "list", "--json"])
        .output()
        .expect("collection list");
    assert!(output.status.success());
    let collections: Value = serde_json::from_slice(&output.stdout).expect("collections");
    assert_eq!(collections[0]["name"], "wide");
    assert_eq!(collections[0]["filter"]["min_width"], 64);
}

#[test]
fn cli_move_is_dry_run_then_apply_and_byte_identical_undo() {
    let fixture = Fixture::new();
    let image = fixture.source.join("move-me.png");
    let id = fixture.add_and_scan(&image);
    let original = fs::read(&image).expect("original bytes");
    let destination = fixture.root.join("moved");

    fixture
        .command()
        .args(["move", "--image-id", &id.to_string(), "--to"])
        .arg(&destination)
        .arg("--json")
        .assert()
        .success();
    assert!(image.exists(), "dry-run must not move the file");

    let output = fixture
        .command()
        .args(["move", "--image-id", &id.to_string(), "--to"])
        .arg(&destination)
        .args(["--apply", "--json"])
        .output()
        .expect("apply");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).expect("move result");
    let operation = result["id"].as_str().expect("operation id");
    assert!(!image.exists());
    assert_eq!(
        fs::read(destination.join("move-me.png")).expect("moved"),
        original
    );

    let output = fixture
        .command()
        .args(["move", "undo", operation, "--json"])
        .output()
        .expect("undo");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let undo: Value = serde_json::from_slice(&output.stdout).expect("undo result");
    assert_eq!(undo["status"], "undone");
    assert_eq!(fs::read(&image).expect("restored"), original);
}
