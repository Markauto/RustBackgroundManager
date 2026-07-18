# Background Manager

`bgm` is a native Rust CLI and Ratatui browser for cataloguing wallpaper directories, finding images by visual metadata or CLIP similarity, building saved collections, safely moving files, and exposing collections to wpaperd as managed symlink pools.

The catalog never modifies source images during a scan. Deterministic metadata and normalized CLIP embeddings live in SQLite; thumbnails, model files, move manifests, wpaperd pools, and backups follow the XDG directory layout.

## Build

Rust 1.92 or newer is required. The normal AMD build also requires a working ROCm HIP SDK: `hipconfig` must be in `PATH`, `libamdhip64` and `libhiprtc` must be linkable, and the user must be able to access `/dev/kfd`.

```sh
cargo build --release --features rocm
```

ROCm is the only AI backend. There is deliberately no CPU or CUDA fallback. A backend-free build is available for deterministic catalog work and development on systems without the HIP SDK:

```sh
cargo build --release
bgm scan --no-ai
```

AI and semantic commands return an explicit error in that build.

## First run

```sh
bgm doctor
bgm source add ~/Pictures/Backgrounds
bgm model install
bgm scan
bgm
```

`bgm` without arguments opens the TUI. Model installation asks before downloading the pinned OpenAI CLIP ViT-B/32 artifacts (about 580 MiB) and verifies every file with SHA-256. For scripts or other non-interactive sessions, install explicitly:

```sh
bgm model install --yes
```

The model and immutable Hugging Face revision are recorded in `config.toml`; v1 rejects unpinned alternatives.

## Search and collections

Every search surface uses the same versioned `FilterSpecV1`. Different facets are combined with AND; repeated values within one facet are combined with OR.

```sh
bgm search --min-width 2560 --orientation landscape --brightness dark
bgm search --ratio 16:9 --ratio 16:10 --tag desktop --favorite --json
bgm search --dominant-colour '#203060:0.08'
bgm search --palette-colour '#D08040:0.10'
bgm search --ai mood=calm:0.55 --ai subject=nature:0.45
bgm search --semantic 'misty mountains at night' --semantic-min-score 0.20

bgm collection save night --brightness dark --tag desktop
bgm collection list
bgm collection show night --json
```

Filters cover source IDs, path fragments, dimensions, orientation, aspect ratios, luminance/light-dark class, dominant and palette Oklab distance, AI label estimates, arbitrary semantic text, tags, and favorites. The TUI’s `/` editor exposes the complete filter as editable `FilterSpecV1` JSON alongside built-in examples and saved presets. TUI presets use the existing saved-collection store, so presets created there are also available to `bgm collection`, wpaperd bindings, and later TUI sessions.

CLIP outputs are displayed as ranked estimates rather than authoritative tags. Seeded `mood`, `subject`, and `style` label packs can be edited, and custom packs can supply multiple prompts per label:

```sh
bgm label list --json
bgm label set lighting --kind custom \
  --label 'neon=a neon-lit wallpaper' \
  --label 'neon=bright neon lights at night' \
  --label 'muted=a wallpaper with muted lighting'
bgm label rescore lighting
```

Rescoring reuses normalized image embeddings and does not decode the source images again.

## Tags, favorites, and safe moves

```sh
bgm tag add desktop 12 15 18
bgm favorite set 12 18

# Dry run only
bgm move --tag archive --to ~/Pictures/Archive

# Validate hashes and destinations, then perform the move
bgm move --tag archive --to ~/Pictures/Archive --apply --json
bgm move undo UUID-FROM-THE-PREVIOUS-COMMAND
```

A move rejects overwrites, duplicate targets, symlink replacements, unavailable files, and content changed since the last scan. Same-filesystem moves use no-replace rename; cross-filesystem moves use copy, fsync, hash verification, no-clobber persistence, then source deletion. Every applied or partial operation has both SQLite state and an atomic JSON manifest. Undo proceeds only when all paths and hashes still match.

## wpaperd

Bind a saved collection globally or to one of the supported display overrides:

```sh
bgm wpaperd bind any night
bgm wpaperd bind DP-1 portrait
bgm wpaperd bind DP-2 wide
bgm wpaperd bind HDMI-A-1 warm
bgm wpaperd status --json
bgm wpaperd refresh
bgm wpaperd unbind DP-1
```

Empty collections are refused. Each pool is built as collision-safe symlinks in a temporary sibling directory and atomically exchanged into place. Only the selected section’s `path` is edited with `toml_edit`; comments, formatting, and unrelated keys are retained. The original config is backed up once, and unbind restores the displaced path only if the live value still points at bgm’s managed pool.

Scans, moves, label/tag/favorite changes, and collection updates refresh active bindings. If a collection becomes empty, its existing pool remains in place and refresh reports an error rather than replacing it with an empty directory.

## TUI keys

- `↑`/`↓`, `j`/`k`, Page Up/Down, Home/End: navigate results
- `:`: open the command palette and run any non-TUI `bgm` command
- `/`: open the filter editor with built-in examples, saved presets, and full `FilterSpecV1` JSON
- `f`: toggle favorite
- `t`: add a tag
- `c`: save, load, list, or delete collections
- `m`: preview and confirm a move for the selected image
- `w`: bind or unbind a wpaperd display
- `s`: run scan and optional GPU analysis off the UI thread
- `o` or Enter: open the original with `xdg-open`
- `?`: help; `q`: quit

The command palette uses the same Clap command tree as the CLI, so its IntelliSense list follows the current subcommand and suggests flags and enumerated values. It also completes live collection names, registered source paths, label packs, wpaperd displays, catalog tags, and the selected image ID where relevant. Use Up/Down to choose a suggestion, Tab to accept it, Ctrl+P/Ctrl+N for command history, and Enter to validate and run. Commands execute in the background; their stdout and diagnostics open in a scrollable result panel, and the browser refreshes afterward. The optional leading `bgm` is accepted, quotes and `~/` paths work, and nested `tui` launches are refused. The palette invokes `bgm` directly rather than through a shell, so shell operators and environment-variable expansion are intentionally unavailable.

Inside the filter editor, Tab switches between the preset list and pretty-printed JSON. In the preset list, use the arrow keys and Enter to load one of the read-only examples or a saved collection into the editor. Ctrl+P saves the current validated JSON as a named preset; saving an existing name updates it. In the JSON pane, the arrow keys, Home/End, and Page Up/Down move through the document, Enter inserts a line, and Backspace/Delete edit text. Ctrl+S validates and applies the filter, Ctrl+R restores the default filter, and Esc cancels. Parse or validation errors stay open in the editor so they can be corrected. Pasted multiline JSON and preset names are supported.

Kitty graphics are selected when available. Other terminals use ratatui-image half-block rendering, and `xdg-open` remains available as a fallback.

## XDG data

| Purpose | Default path |
| --- | --- |
| Configuration | `~/.config/bgm/config.toml` |
| SQLite catalog | `~/.local/share/bgm/catalog.sqlite3` |
| Pinned model | `~/.local/share/bgm/models/` |
| Thumbnails | `~/.cache/bgm/thumbnails/` |
| Symlink pools | `~/.local/state/bgm/wpaperd/` |
| Move manifests | `~/.local/state/bgm/moves/` |
| wpaperd backup | `~/.local/state/bgm/backups/` |

`XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, and `XDG_STATE_HOME` are honored when set.

## Verification

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test

# Compile/lint the sole AI backend
cargo check --all-targets --features rocm
cargo clippy --all-targets --features rocm -- -D warnings

# Hardware/model-gated inference test
BGM_CLIP_MODEL_DIR=/path/to/verified/model \
  cargo test --features rocm rocm_clip_has_stable_embedding_dimensions_and_ranking -- --ignored
```

The test suite uses temporary XDG roots for catalog, move, and wpaperd workflows. It includes command-level JSON tests, byte-identical move undo, corrupt/missing and overlapping-root scans, executable SQL filter composition, TOML preservation, rebinding safety, and a Ratatui screen snapshot.
