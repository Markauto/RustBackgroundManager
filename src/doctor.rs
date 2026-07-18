use std::{
    env,
    path::{Path, PathBuf},
};

use serde::Serialize;

use crate::{AppPaths, config::Config, db::Database, model};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckLevel {
    Pass,
    Warning,
    Fail,
}

#[derive(Clone, Debug, Serialize)]
pub struct DoctorCheck {
    pub name: String,
    pub level: CheckLevel,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DoctorReport {
    pub healthy: bool,
    pub checks: Vec<DoctorCheck>,
}

pub fn run(paths: &AppPaths) -> DoctorReport {
    let mut checks = Vec::new();
    check_path_layout(paths, &mut checks);
    check_config(paths, &mut checks);
    check_database(paths, &mut checks);
    check_command("wpaperd", false, &mut checks);
    check_command("xdg-open", false, &mut checks);
    check_terminal(&mut checks);
    check_rocm(&mut checks);
    check_model(paths, &mut checks);
    DoctorReport {
        healthy: !checks.iter().any(|check| check.level == CheckLevel::Fail),
        checks,
    }
}

fn check_path_layout(paths: &AppPaths, checks: &mut Vec<DoctorCheck>) {
    let paths_to_check = [
        ("configuration", &paths.config_dir),
        ("data", &paths.data_dir),
        ("cache", &paths.cache_dir),
        ("state", &paths.state_dir),
    ];
    let unavailable: Vec<_> = paths_to_check
        .into_iter()
        .filter(|(_, path)| !path.is_dir())
        .map(|(name, path)| format!("{name}: {}", path.display()))
        .collect();
    if unavailable.is_empty() {
        pass(checks, "xdg", "all bgm XDG directories are available");
    } else {
        fail(
            checks,
            "xdg",
            format!(
                "missing application directories: {}",
                unavailable.join(", ")
            ),
        );
    }
}

fn check_config(paths: &AppPaths, checks: &mut Vec<DoctorCheck>) {
    match Config::load(&paths.config_file) {
        Ok(config) => pass(
            checks,
            "config",
            format!(
                "version {} at {}",
                config.version,
                paths.config_file.display()
            ),
        ),
        Err(error) => fail(checks, "config", format!("{error:#}")),
    }
}

fn check_database(paths: &AppPaths, checks: &mut Vec<DoctorCheck>) {
    match Database::open(&paths.database).and_then(|database| {
        database.with_connection(|connection| {
            let result: String =
                connection.query_row("PRAGMA quick_check", [], |row| row.get(0))?;
            anyhow::ensure!(result == "ok", "SQLite quick_check reported: {result}");
            Ok(())
        })
    }) {
        Ok(()) => pass(
            checks,
            "sqlite",
            format!("catalog is accessible at {}", paths.database.display()),
        ),
        Err(error) => fail(checks, "sqlite", format!("{error:#}")),
    }
}

fn check_command(command: &str, required: bool, checks: &mut Vec<DoctorCheck>) {
    match find_command(command) {
        Some(path) => pass(checks, command, format!("found at {}", path.display())),
        None if required => fail(checks, command, "command is not present in PATH"),
        None => warning(checks, command, "command is not present in PATH"),
    }
}

fn check_terminal(checks: &mut Vec<DoctorCheck>) {
    let term = env::var("TERM").unwrap_or_default();
    if env::var_os("KITTY_WINDOW_ID").is_some() || term.contains("kitty") {
        pass(checks, "kitty", format!("Kitty terminal detected ({term})"));
    } else {
        warning(
            checks,
            "kitty",
            "Kitty was not detected; TUI previews will use half-blocks and xdg-open",
        );
    }
}

fn check_rocm(checks: &mut Vec<DoctorCheck>) {
    if !Path::new("/dev/kfd").exists() {
        fail(checks, "rocm", "/dev/kfd is unavailable");
        return;
    }
    match model::probe_rocm() {
        Ok(message) => pass(checks, "rocm", message),
        Err(error) => fail(checks, "rocm", format!("{error:#}")),
    }
}

fn check_model(paths: &AppPaths, checks: &mut Vec<DoctorCheck>) {
    let status = model::status(paths, false);
    if status.verified {
        pass(
            checks,
            "clip",
            format!("pinned model installed at {}", status.directory.display()),
        );
    } else {
        warning(
            checks,
            "clip",
            "pinned model is not installed; run `bgm model install --yes`",
        );
    }
}

fn find_command(command: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|directory| directory.join(command))
            .find(|candidate| candidate.is_file())
    })
}

fn pass(checks: &mut Vec<DoctorCheck>, name: impl Into<String>, message: impl Into<String>) {
    checks.push(DoctorCheck {
        name: name.into(),
        level: CheckLevel::Pass,
        message: message.into(),
    });
}

fn warning(checks: &mut Vec<DoctorCheck>, name: impl Into<String>, message: impl Into<String>) {
    checks.push(DoctorCheck {
        name: name.into(),
        level: CheckLevel::Warning,
        message: message.into(),
    });
}

fn fail(checks: &mut Vec<DoctorCheck>, name: impl Into<String>, message: impl Into<String>) {
    checks.push(DoctorCheck {
        name: name.into(),
        level: CheckLevel::Fail,
        message: message.into(),
    });
}
