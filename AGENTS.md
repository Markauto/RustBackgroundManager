# Repository Guidelines

## Project Structure & Module Organization

`bgm` is a Rust 2024 CLI and Ratatui application. `src/main.rs` starts the binary, while `src/lib.rs` exposes the application modules. Code is organized by concern: scanning and analysis (`scan.rs`, `analysis.rs`), persistence (`db.rs`), filters and collections (`filter.rs`, `collection.rs`), safe file moves (`move_files.rs`), UI (`tui.rs`), and wpaperd integration (`wpaperd.rs`). The optional ROCm CLIP implementation lives under `src/model/`. Unit tests are colocated in `src/`; command-level tests live in `tests/cli_workflows.rs`, and approved UI snapshots live in `src/snapshots/`. Consult `spec.md` for behavioral requirements and `README.md` for user-facing workflows.

## Build, Test, and Development Commands

- `cargo build` builds the backend-free development binary.
- `cargo run -- scan --no-ai` exercises deterministic catalog scanning locally.
- `cargo run --` opens the TUI.
- `cargo build --release --features rocm` builds the production AMD/ROCm variant.
- `cargo fmt --check` verifies formatting.
- `cargo clippy --all-targets -- -D warnings` enforces lint cleanliness.
- `cargo test` runs unit, integration, and snapshot tests.
- `cargo check --all-targets --features rocm` verifies the optional AI backend without producing a release binary.

Rust 1.92 or newer is required; the pinned stable toolchain includes `rustfmt` and Clippy.

## Coding Style & Naming Conventions

Use standard `rustfmt` output (four-space indentation). Name modules, functions, and variables in `snake_case`; types and traits in `UpperCamelCase`; constants in `SCREAMING_SNAKE_CASE`. Keep functionality in the narrowest relevant module and return contextual errors rather than silently recovering. Unsafe Rust is forbidden, and all Clippy warnings configured in `Cargo.toml` are errors.

## Testing Guidelines

Add focused `#[test]` cases beside changed logic and integration tests for observable CLI behavior. Name tests after the behavior or invariant they prove. Use `tempfile` and temporary XDG roots; tests must never touch a developer's real catalog, wallpapers, or wpaperd configuration. Review snapshot changes intentionally. The hardware-gated ROCm inference test requires `BGM_CLIP_MODEL_DIR`; see `README.md` for its exact command. No numeric coverage threshold is defined, but safety-critical move, scan, SQL/filter, and configuration paths need regression tests.

## Commit & Pull Request Guidelines

History is currently sparse (`Init`, `Done v1`), so no formal commit convention exists. Use concise, imperative subjects that identify the change, for example `Reject overlapping scan roots`. Pull requests should explain user-visible behavior and safety implications, link relevant issues, list verification commands, and include screenshots or snapshot diffs for TUI changes. State whether ROCm checks were run or why they were unavailable.
