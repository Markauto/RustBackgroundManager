use std::{io::Write as _, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Serialize;
use uuid::Uuid;

// The standard printing macros panic on a closed pipe. CLI output should
// instead return an I/O error so `main` can treat EPIPE as normal termination.
macro_rules! println {
    ($($argument:tt)*) => {{
        writeln!(std::io::stdout().lock(), $($argument)*)?;
    }};
}

macro_rules! print {
    ($($argument:tt)*) => {{
        write!(std::io::stdout().lock(), $($argument)*)?;
    }};
}

use crate::{
    AppPaths,
    analysis::{LightDark, Orientation},
    collection::{
        add_tag, delete_collection, get_collection, list_collections, remove_tag, save_collection,
        search_resolved, set_favorite,
    },
    config::Config,
    db::{Database, ImageRecord},
    doctor,
    filter::{AiLabelFilter, ColourFilter, FilterSpecV1},
    model,
    move_files::{apply_move, plan_move, undo_move},
    scan::{ScanOptions, scan_catalog},
    tui, wpaperd,
};

#[derive(Debug, Parser)]
#[command(name = "bgm", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Open the interactive wallpaper browser.
    Tui,
    /// Check the local environment and bgm state.
    Doctor {
        #[arg(long)]
        json: bool,
    },
    /// View or change bgm configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Register image source directories.
    Source {
        #[command(subcommand)]
        command: SourceCommand,
    },
    /// Incrementally catalog all registered sources.
    Scan {
        #[arg(long)]
        full: bool,
        #[arg(long)]
        no_ai: bool,
        #[arg(long)]
        json: bool,
    },
    /// Install or inspect the pinned CLIP model.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Edit CLIP estimate label packs and rescore saved embeddings.
    Label {
        #[command(subcommand)]
        command: LabelCommand,
    },
    /// Search the catalog using the shared filter specification.
    Search(SearchArgs),
    /// Manage saved filter collections.
    Collection {
        #[command(subcommand)]
        command: CollectionCommand,
    },
    /// Add or remove custom tags.
    Tag {
        #[command(subcommand)]
        command: TagCommand,
    },
    /// Set or unset favorites.
    Favorite {
        #[command(subcommand)]
        command: FavoriteCommand,
    },
    /// Preview, apply, or undo safe file moves.
    Move(MoveArgs),
    /// Connect saved collections to wpaperd.
    Wpaperd {
        #[command(subcommand)]
        command: WpaperdCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ConfigCommand {
    Show {
        #[arg(long)]
        json: bool,
    },
    Set {
        key: String,
        value: String,
    },
}

#[derive(Debug, Subcommand)]
enum SourceCommand {
    Add {
        directory: PathBuf,
    },
    List {
        #[arg(long)]
        json: bool,
    },
    Remove {
        directory: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    Install {
        #[arg(long)]
        yes: bool,
    },
    Status {
        #[arg(long)]
        verify: bool,
        #[arg(long)]
        json: bool,
    },
    Remove,
}

#[derive(Debug, Subcommand)]
enum LabelCommand {
    List {
        #[arg(long)]
        json: bool,
    },
    Set {
        name: String,
        #[arg(long, default_value = "custom")]
        kind: String,
        #[arg(long = "label", value_name = "NAME[=PROMPT]", required = true)]
        labels: Vec<String>,
    },
    Delete {
        name: String,
    },
    Rescore {
        name: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Args)]
struct SearchArgs {
    #[command(flatten)]
    filter: FilterArgs,
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Debug, Default, Args)]
struct FilterArgs {
    #[arg(long = "source", value_name = "ID")]
    source_ids: Vec<i64>,
    #[arg(long = "path", value_name = "TEXT")]
    paths: Vec<String>,
    #[arg(long)]
    min_width: Option<u32>,
    #[arg(long)]
    max_width: Option<u32>,
    #[arg(long)]
    min_height: Option<u32>,
    #[arg(long)]
    max_height: Option<u32>,
    #[arg(long = "orientation")]
    orientations: Vec<OrientationArg>,
    #[arg(long = "ratio", value_name = "RATIO")]
    ratios: Vec<String>,
    #[arg(long, default_value_t = 0.03)]
    ratio_tolerance: f64,
    #[arg(long = "brightness")]
    brightness: Vec<BrightnessArg>,
    #[arg(long)]
    min_luminance: Option<f32>,
    #[arg(long)]
    max_luminance: Option<f32>,
    #[arg(long = "dominant-colour", value_name = "HEX[:DISTANCE]")]
    dominant_colours: Vec<String>,
    #[arg(long = "palette-colour", value_name = "HEX[:DISTANCE]")]
    palette_colours: Vec<String>,
    #[arg(long = "ai", value_name = "PACK=LABEL[:SCORE]")]
    ai_labels: Vec<String>,
    #[arg(long)]
    semantic: Option<String>,
    #[arg(long)]
    semantic_min_score: Option<f32>,
    #[arg(long = "tag")]
    tags: Vec<String>,
    #[arg(long, conflicts_with = "not_favorite")]
    favorite: bool,
    #[arg(long, conflicts_with = "favorite")]
    not_favorite: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OrientationArg {
    Landscape,
    Portrait,
    Square,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BrightnessArg {
    Light,
    Dark,
}

#[derive(Debug, Subcommand)]
enum CollectionCommand {
    Save {
        name: String,
        #[command(flatten)]
        filter: Box<FilterArgs>,
    },
    List {
        #[arg(long)]
        json: bool,
    },
    Show {
        name: String,
        #[arg(long)]
        json: bool,
    },
    Delete {
        name: String,
    },
}

#[derive(Debug, Subcommand)]
enum TagCommand {
    Add {
        tag: String,
        #[arg(required = true)]
        image_ids: Vec<i64>,
    },
    Remove {
        tag: String,
        #[arg(required = true)]
        image_ids: Vec<i64>,
    },
}

#[derive(Debug, Subcommand)]
enum FavoriteCommand {
    Set {
        #[arg(required = true)]
        image_ids: Vec<i64>,
    },
    Unset {
        #[arg(required = true)]
        image_ids: Vec<i64>,
    },
}

#[derive(Debug, Args)]
struct MoveArgs {
    #[command(subcommand)]
    action: Option<MoveAction>,
    #[command(flatten)]
    filter: FilterArgs,
    #[arg(long = "image-id")]
    image_ids: Vec<i64>,
    #[arg(long)]
    all: bool,
    #[arg(long)]
    to: Option<PathBuf>,
    #[arg(long)]
    apply: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum MoveAction {
    Undo {
        id: Uuid,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum WpaperdCommand {
    Bind {
        display: String,
        collection: String,
    },
    Refresh {
        display: Option<String>,
    },
    Status {
        #[arg(long)]
        json: bool,
    },
    Unbind {
        display: String,
    },
}

struct AppContext {
    paths: AppPaths,
    config: Config,
    database: Database,
}

impl AppContext {
    fn load() -> Result<Self> {
        let paths = AppPaths::discover()?;
        paths.ensure_owned_dirs()?;
        let config = Config::load(&paths.config_file)?;
        if !paths.config_file.exists() {
            config.save(&paths.config_file)?;
        }
        let database = Database::open(&paths.database)?;
        Ok(Self {
            paths,
            config,
            database,
        })
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let mut context = AppContext::load()?;
    match cli.command.unwrap_or(Command::Tui) {
        Command::Tui => tui::run(&context.database, &context.paths, &context.config),
        Command::Doctor { json } => command_doctor(&context.paths, json),
        Command::Config { command } => command_config(&mut context, command),
        Command::Source { command } => command_source(&context, command),
        Command::Scan { full, no_ai, json } => command_scan(&context, full, no_ai, json),
        Command::Model { command } => command_model(&context, command),
        Command::Label { command } => command_label(&context, command),
        Command::Search(arguments) => command_search(&context, arguments),
        Command::Collection { command } => command_collection(&context, command),
        Command::Tag { command } => command_tag(&context, command),
        Command::Favorite { command } => command_favorite(&context, command),
        Command::Move(arguments) => command_move(&context, arguments),
        Command::Wpaperd { command } => command_wpaperd(&context, command),
    }
}

fn command_doctor(paths: &AppPaths, json: bool) -> Result<()> {
    let report = doctor::run(paths);
    if json {
        print_json(&report)?;
    } else {
        for check in &report.checks {
            let marker = match check.level {
                doctor::CheckLevel::Pass => "ok",
                doctor::CheckLevel::Warning => "warn",
                doctor::CheckLevel::Fail => "FAIL",
            };
            println!("[{marker:4}] {:12} {}", check.name, check.message);
        }
    }
    if !report.healthy {
        bail!("one or more required checks failed");
    }
    Ok(())
}

fn command_config(context: &mut AppContext, command: ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Show { json } => {
            if json {
                print_json(&context.config)
            } else {
                print!("{}", toml::to_string_pretty(&context.config)?);
                Ok(())
            }
        }
        ConfigCommand::Set { key, value } => {
            context.config.set(&key, &value)?;
            context.config.save(&context.paths.config_file)?;
            println!("set {key}");
            Ok(())
        }
    }
}

fn command_source(context: &AppContext, command: SourceCommand) -> Result<()> {
    match command {
        SourceCommand::Add { directory } => {
            let source = context.database.add_source(&directory)?;
            println!("{}\t{}", source.id, source.path.display());
            Ok(())
        }
        SourceCommand::List { json } => {
            let sources = context.database.list_sources()?;
            if json {
                print_json(&sources)
            } else {
                for source in sources {
                    println!("{}\t{}", source.id, source.path.display());
                }
                Ok(())
            }
        }
        SourceCommand::Remove { directory } => {
            if !context.database.remove_source(&directory)? {
                bail!("source was not registered: {}", directory.display());
            }
            refresh_after_change(context);
            println!("removed {}", directory.display());
            Ok(())
        }
    }
}

fn command_scan(context: &AppContext, full: bool, no_ai: bool, json: bool) -> Result<()> {
    let report = scan_catalog(
        &context.database,
        &context.paths,
        &context.config,
        ScanOptions { full, no_ai },
    )?;
    let ai = if no_ai || !context.config.ai.enabled {
        None
    } else {
        Some(model::analyze_missing(&context.database, &context.paths)?)
    };
    refresh_after_change(context);
    if json {
        #[derive(Serialize)]
        struct Output<'a> {
            scan: &'a crate::scan::ScanReport,
            ai: &'a Option<model::AiReport>,
        }
        print_json(&Output {
            scan: &report,
            ai: &ai,
        })
    } else {
        println!(
            "discovered={} analyzed={} unchanged={} excluded={} missing={} failed={}",
            report.discovered,
            report.analyzed,
            report.unchanged,
            report.out_of_bounds,
            report.missing,
            report.failed
        );
        for failure in report.failures {
            eprintln!("{}: {}", failure.path.display(), failure.error);
        }
        if let Some(ai) = ai {
            println!(
                "AI estimates: embedded={} scored={} failed={}",
                ai.embedded, ai.scored, ai.failed
            );
        }
        Ok(())
    }
}

fn command_model(context: &AppContext, command: ModelCommand) -> Result<()> {
    match command {
        ModelCommand::Install { yes } => {
            let status = model::install(&context.paths, yes)?;
            println!(
                "verified {} at {}",
                status.model,
                status.directory.display()
            );
            Ok(())
        }
        ModelCommand::Status { verify, json } => {
            let status = model::status(&context.paths, verify);
            if json {
                print_json(&status)
            } else {
                println!("model: {}", status.model);
                println!("revision: {}", status.revision);
                println!("directory: {}", status.directory.display());
                println!("installed: {}", status.installed);
                println!("verified: {}", status.verified);
                println!("ROCm build: {}", status.rocm_compiled);
                if let Some(problem) = status.problem {
                    println!("problem: {problem}");
                }
                Ok(())
            }
        }
        ModelCommand::Remove => {
            if model::remove(&context.paths)? {
                println!("removed pinned CLIP model");
            } else {
                println!("pinned CLIP model was not installed");
            }
            Ok(())
        }
    }
}

fn command_label(context: &AppContext, command: LabelCommand) -> Result<()> {
    match command {
        LabelCommand::List { json } => {
            let packs = model::list_label_packs(&context.database)?;
            if json {
                print_json(&packs)
            } else {
                for pack in packs {
                    println!("{}\t{}\t{} labels", pack.name, pack.kind, pack.labels.len());
                }
                Ok(())
            }
        }
        LabelCommand::Set { name, kind, labels } => {
            let definitions = parse_label_definitions(&labels)?;
            let pack = model::save_label_pack(&context.database, &name, &kind, &definitions)?;
            refresh_after_change(context);
            println!(
                "saved {} ({} labels); run `bgm label rescore {}` to update estimates",
                pack.name,
                pack.labels.len(),
                pack.name
            );
            Ok(())
        }
        LabelCommand::Delete { name } => {
            if !model::delete_label_pack(&context.database, &name)? {
                bail!("label pack not found: {name}");
            }
            refresh_after_change(context);
            println!("deleted {name}");
            Ok(())
        }
        LabelCommand::Rescore { name, json } => {
            let report =
                model::rescore_label_packs(&context.database, &context.paths, name.as_deref())?;
            refresh_after_change(context);
            if json {
                print_json(&report)
            } else {
                println!(
                    "rescored={} failed={} (saved image embeddings were reused)",
                    report.scored, report.failed
                );
                Ok(())
            }
        }
    }
}

fn command_search(context: &AppContext, arguments: SearchArgs) -> Result<()> {
    let filter = arguments.filter.to_filter()?;
    let hits = search_resolved(&context.database, &context.paths, &filter)?;
    if arguments.json {
        print_json(&hits)
    } else {
        for hit in hits {
            let dimensions = match (hit.image.width, hit.image.height) {
                (Some(width), Some(height)) => format!("{width}x{height}"),
                _ => "?x?".into(),
            };
            let score = hit
                .semantic_score
                .map_or_else(String::new, |score| format!("\tsemantic≈{score:.3}"));
            let favorite = if hit.image.favorite { "★" } else { " " };
            println!(
                "{}\t{}\t{}\t{}{}",
                hit.image.id,
                favorite,
                dimensions,
                hit.image.path.display(),
                score
            );
        }
        Ok(())
    }
}

fn command_collection(context: &AppContext, command: CollectionCommand) -> Result<()> {
    match command {
        CollectionCommand::Save { name, filter } => {
            let collection = save_collection(&context.database, &name, &filter.to_filter()?)?;
            refresh_after_change(context);
            println!("saved {}", collection.name);
            Ok(())
        }
        CollectionCommand::List { json } => {
            let collections = list_collections(&context.database)?;
            if json {
                print_json(&collections)
            } else {
                for collection in collections {
                    println!("{}\t{}", collection.id, collection.name);
                }
                Ok(())
            }
        }
        CollectionCommand::Show { name, json } => {
            let collection = get_collection(&context.database, &name)?
                .with_context(|| format!("collection not found: {name}"))?;
            if json {
                print_json(&collection)
            } else {
                println!("{}", serde_json::to_string_pretty(&collection.filter)?);
                Ok(())
            }
        }
        CollectionCommand::Delete { name } => {
            if !delete_collection(&context.database, &name)? {
                bail!("collection not found: {name}");
            }
            println!("deleted {name}");
            Ok(())
        }
    }
}

fn command_tag(context: &AppContext, command: TagCommand) -> Result<()> {
    let changed = match command {
        TagCommand::Add { tag, image_ids } => add_tag(&context.database, &image_ids, &tag)?,
        TagCommand::Remove { tag, image_ids } => remove_tag(&context.database, &image_ids, &tag)?,
    };
    refresh_after_change(context);
    println!("updated {changed} image(s)");
    Ok(())
}

fn command_favorite(context: &AppContext, command: FavoriteCommand) -> Result<()> {
    let changed = match command {
        FavoriteCommand::Set { image_ids } => set_favorite(&context.database, &image_ids, true)?,
        FavoriteCommand::Unset { image_ids } => set_favorite(&context.database, &image_ids, false)?,
    };
    refresh_after_change(context);
    println!("updated {changed} image(s)");
    Ok(())
}

fn command_move(context: &AppContext, arguments: MoveArgs) -> Result<()> {
    if let Some(MoveAction::Undo { id, json }) = arguments.action {
        let result = undo_move(&context.database, &context.paths, id)?;
        refresh_after_change(context);
        return if json {
            print_json(&result)
        } else {
            println!("undid {} file(s) from move {}", result.moved, result.id);
            Ok(())
        };
    }
    if !arguments.all && arguments.image_ids.is_empty() && arguments.filter.is_empty() {
        bail!("move needs --image-id, at least one filter, or an explicit --all");
    }
    let images = if arguments.image_ids.is_empty() {
        search_resolved(
            &context.database,
            &context.paths,
            &arguments.filter.to_filter()?,
        )?
        .into_iter()
        .map(|result| result.image)
        .collect()
    } else {
        load_images(&context.database, &arguments.image_ids)?
    };
    let destination = arguments.to.context("move destination is required")?;
    let plan = plan_move(&images, &destination)?;
    if !arguments.apply {
        if arguments.json {
            return print_json(&plan);
        }
        println!(
            "dry run: {} file(s) would move to {}",
            plan.items.len(),
            destination.display()
        );
        for item in plan.items {
            println!(
                "{} -> {}",
                item.original_path.display(),
                item.destination.display()
            );
        }
        println!("rerun with --apply to perform this move");
        return Ok(());
    }
    let result = apply_move(&context.database, &context.paths, plan)?;
    refresh_after_change(context);
    if arguments.json {
        print_json(&result)
    } else {
        println!(
            "move {} completed: {} file(s); manifest {}",
            result.id,
            result.moved,
            result.manifest.display()
        );
        Ok(())
    }
}

fn command_wpaperd(context: &AppContext, command: WpaperdCommand) -> Result<()> {
    match command {
        WpaperdCommand::Bind {
            display,
            collection,
        } => {
            let binding = wpaperd::bind(&context.database, &context.paths, &display, &collection)?;
            println!(
                "{} -> {} ({})",
                binding.display,
                binding.collection_name,
                binding.pool_path.display()
            );
            Ok(())
        }
        WpaperdCommand::Refresh { display } => {
            let bindings = wpaperd::refresh(&context.database, &context.paths, display.as_deref())?;
            println!("refreshed {} binding(s)", bindings.len());
            Ok(())
        }
        WpaperdCommand::Status { json } => {
            let bindings = wpaperd::list_bindings(&context.database)?;
            if json {
                print_json(&bindings)
            } else {
                for binding in bindings {
                    println!(
                        "{}\t{}\t{}",
                        binding.display,
                        binding.collection_name,
                        binding.pool_path.display()
                    );
                }
                Ok(())
            }
        }
        WpaperdCommand::Unbind { display } => {
            let result = wpaperd::unbind(&context.database, &context.paths, &display)?;
            if result.config_was_changed_elsewhere {
                println!(
                    "unbound {display}; wpaperd path was left unchanged because it was edited"
                );
            } else {
                println!("unbound {display} and restored its displaced path");
            }
            Ok(())
        }
    }
}

impl FilterArgs {
    fn to_filter(&self) -> Result<FilterSpecV1> {
        let favorite = if self.favorite {
            Some(true)
        } else if self.not_favorite {
            Some(false)
        } else {
            None
        };
        let filter = FilterSpecV1 {
            source_ids: self.source_ids.clone(),
            paths: self.paths.clone(),
            min_width: self.min_width,
            max_width: self.max_width,
            min_height: self.min_height,
            max_height: self.max_height,
            orientations: self
                .orientations
                .iter()
                .map(|orientation| match orientation {
                    OrientationArg::Landscape => Orientation::Landscape,
                    OrientationArg::Portrait => Orientation::Portrait,
                    OrientationArg::Square => Orientation::Square,
                })
                .collect(),
            aspect_ratios: self
                .ratios
                .iter()
                .map(|ratio| parse_ratio(ratio))
                .collect::<Result<Vec<_>>>()?,
            aspect_tolerance: self.ratio_tolerance,
            light_dark: self
                .brightness
                .iter()
                .map(|brightness| match brightness {
                    BrightnessArg::Light => LightDark::Light,
                    BrightnessArg::Dark => LightDark::Dark,
                })
                .collect(),
            min_luminance: self.min_luminance,
            max_luminance: self.max_luminance,
            dominant_colours: self
                .dominant_colours
                .iter()
                .map(|colour| parse_colour_filter(colour))
                .collect::<Result<Vec<_>>>()?,
            palette_colours: self
                .palette_colours
                .iter()
                .map(|colour| parse_colour_filter(colour))
                .collect::<Result<Vec<_>>>()?,
            ai_labels: self
                .ai_labels
                .iter()
                .map(|label| parse_ai_filter(label))
                .collect::<Result<Vec<_>>>()?,
            semantic_text: self.semantic.clone(),
            semantic_min_score: self.semantic_min_score,
            tags: self.tags.clone(),
            favorite,
            ..FilterSpecV1::default()
        };
        filter.validate()?;
        Ok(filter)
    }

    fn is_empty(&self) -> bool {
        self.source_ids.is_empty()
            && self.paths.is_empty()
            && self.min_width.is_none()
            && self.max_width.is_none()
            && self.min_height.is_none()
            && self.max_height.is_none()
            && self.orientations.is_empty()
            && self.ratios.is_empty()
            && self.brightness.is_empty()
            && self.min_luminance.is_none()
            && self.max_luminance.is_none()
            && self.dominant_colours.is_empty()
            && self.palette_colours.is_empty()
            && self.ai_labels.is_empty()
            && self.semantic.is_none()
            && self.tags.is_empty()
            && !self.favorite
            && !self.not_favorite
    }
}

fn parse_ratio(value: &str) -> Result<f64> {
    if let Some((width, height)) = value.split_once(':') {
        let width: f64 = width.parse().context("invalid ratio width")?;
        let height: f64 = height.parse().context("invalid ratio height")?;
        if height == 0.0 {
            bail!("ratio height cannot be zero");
        }
        Ok(width / height)
    } else {
        value
            .parse()
            .context("ratio must be decimal or WIDTH:HEIGHT")
    }
}

fn parse_colour_filter(value: &str) -> Result<ColourFilter> {
    let (hex, distance) = value
        .split_once(':')
        .map_or((value, 0.08), |(hex, distance)| {
            (hex, distance.parse::<f32>().unwrap_or(f32::NAN))
        });
    if distance.is_nan() {
        bail!("invalid colour distance in {value}");
    }
    Ok(ColourFilter {
        hex: hex.to_owned(),
        max_distance: distance,
    })
}

fn parse_ai_filter(value: &str) -> Result<AiLabelFilter> {
    let (pack, label_and_score) = value
        .split_once('=')
        .context("AI filter must be PACK=LABEL[:SCORE]")?;
    let (label, score) = label_and_score
        .rsplit_once(':')
        .map_or((label_and_score, 0.5), |(label, score)| {
            (label, score.parse::<f32>().unwrap_or(f32::NAN))
        });
    if pack.is_empty() || label.is_empty() || score.is_nan() {
        bail!("invalid AI filter: {value}");
    }
    Ok(AiLabelFilter {
        pack: pack.into(),
        label: label.into(),
        min_score: score,
    })
}

fn parse_label_definitions(values: &[String]) -> Result<Vec<model::LabelDefinition>> {
    let mut labels: Vec<model::LabelDefinition> = Vec::new();
    for value in values {
        let (name, prompt) = value
            .split_once('=')
            .map_or((value.as_str(), None), |(name, prompt)| {
                (name, Some(prompt))
            });
        let name = name.trim();
        if name.is_empty() {
            bail!("label name cannot be empty: {value}");
        }
        let index = labels
            .iter()
            .position(|label| label.name.eq_ignore_ascii_case(name));
        let prompt = prompt
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
            .map(str::to_owned);
        if let Some(index) = index {
            if let Some(prompt) = prompt {
                labels[index].prompts.push(prompt);
            }
        } else {
            labels.push(model::LabelDefinition {
                name: name.to_owned(),
                prompts: prompt.into_iter().collect(),
            });
        }
    }
    Ok(labels)
}

fn load_images(database: &Database, ids: &[i64]) -> Result<Vec<ImageRecord>> {
    ids.iter()
        .map(|id| {
            database
                .get_image(*id)?
                .with_context(|| format!("image not found: {id}"))
        })
        .collect()
}

fn refresh_after_change(context: &AppContext) {
    if let Err(error) = wpaperd::refresh(&context.database, &context.paths, None) {
        eprintln!("warning: wpaperd bindings were not refreshed: {error:#}");
    }
}

fn print_json(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ratio_forms() {
        assert!((parse_ratio("16:9").expect("ratio") - 16.0 / 9.0).abs() < f64::EPSILON);
        assert_eq!(parse_ratio("1.5").expect("decimal"), 1.5);
        assert!(parse_ratio("16:0").is_err());
    }

    #[test]
    fn cli_exposes_required_command_tree() {
        use clap::CommandFactory as _;
        let command = Cli::command();
        let names: Vec<_> = command
            .get_subcommands()
            .map(clap::Command::get_name)
            .collect();
        for required in [
            "tui",
            "doctor",
            "config",
            "source",
            "scan",
            "model",
            "label",
            "search",
            "collection",
            "tag",
            "favorite",
            "move",
            "wpaperd",
        ] {
            assert!(names.contains(&required), "missing {required}");
        }
    }
}
