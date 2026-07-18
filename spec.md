# Background Manager (bgm)

  ## Summary

  Build a native Rust 2024 CLI/TUI for this Arch Linux, Hyprland, Kitty, wpaperd, and AMD ROCm 7.2 setup. bgm catalogs multiple image directories without altering originals,
  extracts visual metadata, classifies images locally, supports saved collections and safe file moves, and connects collections to wpaperd through managed symlink
  directories.

  ## Implementation Changes

  - Create one testable Rust package with a library and bgm binary. Use Clap, Ratatui/Crossterm, ratatui-image (https://docs.rs/ratatui-image/latest/ratatui_image/) for Kitty
    previews, SQLite, image, toml_edit, and Burn’s native ROCm backend (https://docs.rs/burn/latest/burn/).

  - Follow XDG locations:
      - Configuration: ~/.config/bgm/config.toml
      - Catalog/models: ~/.local/share/bgm/
      - Thumbnails: ~/.cache/bgm/
      - Symlink pools, move manifests, and backups: ~/.local/state/bgm/

  - Store versioned SQLite records for source roots, discovered images, deterministic analysis, CLIP embeddings/scores, custom tags, favorites, saved collections, wpaperd
    bindings, and reversible move operations.

  - Scan registered roots recursively without following symlinks. Incrementally detect changes using path, size, modification time, and BLAKE3; retain missing/corrupt-file
    status rather than crashing.

  - Support optional global import bounds for minimum/maximum width and height. Files outside those bounds retain a lightweight discovery record but skip thumbnails, palette
    extraction, hashing, and AI until bounds change.

  - Extract width, height, numeric ratio, orientation, nearest common ratio, five-colour Oklab palette with proportions, dominant colour as hex and basic colour name,
    luminance, saturation, contrast, and configurable light/dark classification.

  - Implement CLIP ViT-B/32 directly in Burn and require ROCm for AI work—no silent CPU fallback. On first AI use, interactively download and checksum the pinned OpenAI CLIP
    model (https://huggingface.co/openai/clip-vit-base-patch32); non-interactive use exits with instructions for bgm model install --yes.

  - Persist normalized image embeddings so editable label packs can be rescored without decoding images again. Seed mood, subject, and style packs, allow custom labels/
    prompts, and support arbitrary semantic-text search. Show AI results as ranked estimates, not authoritative tags.

  ## User Interfaces

  - Commands:
      - bgm or bgm tui
      - bgm doctor
      - bgm config show|set
      - bgm source add|list|remove
      - bgm scan [--full] [--no-ai]
      - bgm model install|status|remove
      - bgm search ... [--json]
      - bgm collection save|list|show|delete
      - bgm tag add|remove and bgm favorite set|unset
      - bgm move ... --to DIR [--apply] and bgm move undo ID
      - bgm wpaperd bind|refresh|status|unbind

  - Define one versioned FilterSpecV1, shared by CLI, TUI, and saved collections. It covers source/path, min/max dimensions, orientation, aspect ratio/tolerance, light/dark
    or luminance range, dominant/palette colour and distance, AI pack labels and score thresholds, semantic text, custom tags, and favorites. Different facets combine with
    AND; repeated values within one facet combine with OR.

  - Build a Ratatui browser with a result list, Kitty image preview, palette/metadata pane, filter editor, built-in example filters, collection-backed preset selection and
    saving, collection management, tag/favorite actions, move preview, and wpaperd binding dialog. Provide half-block rendering and xdg-open fallback when Kitty graphics
    are unavailable.

  - Keep scans and GPU analysis off the UI thread, expose progress and per-file failures, and serialize database writes safely.

  ## Filesystem and wpaperd Safety

  - Moves are dry-run by default. Before --apply, validate every destination, reject overwrites and duplicate targets, and record original path, destination, and hash.
  - Use rename on one filesystem or copy–fsync–rename–delete across filesystems. Stop on failure, retain a partial-operation manifest, and allow undo only when hashes and
    paths still match.

  - Materialize each bound collection into an atomically replaced directory of collision-safe symlinks. Refuse empty collections.
  - Support [any] plus overrides for DP-1, DP-2, and HDMI-A-1. Update only each section’s path in ~/.config/wpaperd/config.toml, preserving formatting and unrelated settings
    with toml_edit.

  - Back up the config before its first change and remember the displaced path per binding. Unbind restores it only if the current value still points to a managed pool.
    Refresh bindings after scans, moves, tags, or favorite changes. This relies on wpaperd’s directory paths and hot configuration reload
    (https://github.com/danyspin97/wpaperd#wallpaper-configuration).

  ## Test Plan and Defaults

  - Unit-test aspect categorization, scan bounds, Oklab palette extraction, colour distance, light/dark thresholds, filter composition, label scoring, and configuration
    migrations.

  - Integration-test incremental scans, corrupt images, overlapping roots, SQLite migrations, saved collections, JSON output, dry-run/apply/undo moves, collision handling,
    symlink refreshes, and wpaperd TOML preservation using temporary XDG directories.

  - Snapshot-test TUI screens with Ratatui’s test backend and mock image rendering.
  - Add a ROCm-gated CLIP test verifying that inference runs on the Radeon GPU and produces stable embedding dimensions/rankings.
  - Acceptance-test the current 817 JPEG/PNG files: searchable metadata, visible Kitty previews, working AI labels, global/per-display wpaperd collections, and byte-identical
    move undo.

  - Require cargo fmt --check, cargo clippy --all-targets -- -D warnings, and cargo test.
  - Defaults: no import size bounds, recursive scanning, five palette colours, 3% common-ratio tolerance, light/dark threshold 0.5, no filesystem watcher or daemon, no
    generated captions, and no portable distro packaging in v1.
