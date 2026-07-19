use std::{collections::HashSet, io::Write as _, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum, error::ErrorKind};
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
    db::{Database, ImageRecord, load_images_by_id},
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CommandSuggestion {
    pub label: String,
    pub replacement: String,
    pub description: String,
    pub replace_start: usize,
    pub replace_end: usize,
    pub append_space: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CommandCompletionContext {
    pub completed: Vec<String>,
    pub prefix: String,
    pub replace_start: usize,
    pub replace_end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Quote {
    Single,
    Double,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CommandToken {
    value: String,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LexedCommandLine {
    tokens: Vec<CommandToken>,
    has_active_token: bool,
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
        /// Rehash and reanalyse images even when filesystem metadata is unchanged.
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
    /// Show the active configuration.
    Show {
        #[arg(long)]
        json: bool,
    },
    /// Set one configuration key.
    Set { key: String, value: String },
}

#[derive(Debug, Subcommand)]
enum SourceCommand {
    /// Register a source directory.
    Add { directory: PathBuf },
    /// List registered source directories.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Unregister a source directory.
    Remove { directory: PathBuf },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Download and verify the pinned CLIP model.
    Install {
        #[arg(long)]
        yes: bool,
    },
    /// Inspect the pinned model installation.
    Status {
        #[arg(long)]
        verify: bool,
        #[arg(long)]
        json: bool,
    },
    /// Remove the pinned model files.
    Remove,
}

#[derive(Debug, Subcommand)]
enum LabelCommand {
    /// List label packs.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Create or update a label pack.
    Set {
        name: String,
        #[arg(long, default_value = "custom")]
        kind: String,
        #[arg(long = "label", value_name = "NAME[=PROMPT]", required = true)]
        labels: Vec<String>,
    },
    /// Delete a custom label pack.
    Delete { name: String },
    /// Rescore saved embeddings with one or all label packs.
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
    /// Save a filter as a named collection.
    Save {
        name: String,
        #[command(flatten)]
        filter: Box<FilterArgs>,
    },
    /// List saved collections.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Show one saved collection's filter.
    Show {
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Delete a saved collection.
    Delete { name: String },
}

#[derive(Debug, Subcommand)]
enum TagCommand {
    /// Add a tag to one or more images.
    Add {
        tag: String,
        #[arg(required = true)]
        image_ids: Vec<i64>,
    },
    /// Remove a tag from one or more images.
    Remove {
        tag: String,
        #[arg(required = true)]
        image_ids: Vec<i64>,
    },
}

#[derive(Debug, Subcommand)]
enum FavoriteCommand {
    /// Mark one or more images as favorites.
    Set {
        #[arg(required = true)]
        image_ids: Vec<i64>,
    },
    /// Remove one or more images from favorites.
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
    /// Undo a previously applied move.
    Undo {
        id: Uuid,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum WpaperdCommand {
    /// Bind a display to a saved collection.
    Bind { display: String, collection: String },
    /// Refresh one or all managed wallpaper pools.
    Refresh { display: Option<String> },
    /// Show active wpaperd bindings.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Remove a display binding.
    Unbind { display: String },
}

pub(crate) fn command_completion_context(input: &str, cursor: usize) -> CommandCompletionContext {
    let cursor = cursor.min(input.chars().count());
    let prefix_text = input.chars().take(cursor).collect::<String>();
    let mut prefix_line =
        lex_command_line(&prefix_text, false).expect("incomplete command-line lexing cannot fail");
    let active = prefix_line.has_active_token.then(|| {
        prefix_line
            .tokens
            .pop()
            .expect("an active command token is present")
    });
    let mut completed = prefix_line
        .tokens
        .into_iter()
        .map(|token| token.value)
        .collect::<Vec<_>>();
    if completed.first().is_some_and(|word| word == "bgm") {
        completed.remove(0);
    }

    let (prefix, replace_start) = active.as_ref().map_or_else(
        || (String::new(), cursor),
        |token| (token.value.clone(), token.start),
    );
    let replace_end = active.map_or(cursor, |active| {
        lex_command_line(input, false)
            .expect("incomplete command-line lexing cannot fail")
            .tokens
            .into_iter()
            .find(|token| token.start == active.start && token.end >= cursor)
            .map_or(cursor, |token| token.end)
    });
    CommandCompletionContext {
        completed,
        prefix,
        replace_start,
        replace_end,
    }
}

pub(crate) fn command_suggestions(input: &str, cursor: usize) -> Vec<CommandSuggestion> {
    let context = command_completion_context(input, cursor);
    let mut root = Cli::command();
    root.build();
    let mut command = &root;
    for word in &context.completed {
        if let Some(subcommand) = command
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == word)
        {
            command = subcommand;
        }
    }

    let mut suggestions = Vec::new();
    if let Some((option, value_prefix)) = context.prefix.split_once('=')
        && let Some(argument) = find_argument(command, option)
    {
        for value in argument.get_possible_values() {
            let name = value.get_name();
            push_suggestion(
                &mut suggestions,
                &context,
                format!("{option}={name}"),
                format!("{option}={name}"),
                value
                    .get_help()
                    .map_or_else(String::new, ToString::to_string),
                value_prefix,
                true,
            );
        }
        sort_suggestions(&mut suggestions, &context.prefix);
        return suggestions;
    }

    if let Some(previous) = context.completed.last()
        && let Some(argument) = find_argument(command, previous)
    {
        let possible_values = argument.get_possible_values();
        if !possible_values.is_empty() {
            for value in possible_values {
                let name = value.get_name();
                push_suggestion(
                    &mut suggestions,
                    &context,
                    name.to_owned(),
                    name.to_owned(),
                    value
                        .get_help()
                        .map_or_else(String::new, ToString::to_string),
                    &context.prefix,
                    true,
                );
            }
            sort_suggestions(&mut suggestions, &context.prefix);
            return suggestions;
        }
        if argument.get_action().takes_values() {
            return suggestions;
        }
    }

    for subcommand in command.get_subcommands() {
        if subcommand.get_name() == "help"
            || (command.get_name() == "bgm" && subcommand.get_name() == "tui")
        {
            continue;
        }
        let name = subcommand.get_name();
        push_suggestion(
            &mut suggestions,
            &context,
            name.to_owned(),
            name.to_owned(),
            subcommand
                .get_about()
                .map_or_else(String::new, ToString::to_string),
            &context.prefix,
            true,
        );
    }
    for argument in command.get_arguments() {
        let description = argument
            .get_help()
            .map_or_else(String::new, ToString::to_string);
        if let Some(long) = argument.get_long() {
            let option = format!("--{long}");
            push_suggestion(
                &mut suggestions,
                &context,
                option.clone(),
                option,
                description.clone(),
                &context.prefix,
                true,
            );
        } else if let Some(short) = argument.get_short() {
            let option = format!("-{short}");
            push_suggestion(
                &mut suggestions,
                &context,
                option.clone(),
                option,
                description.clone(),
                &context.prefix,
                true,
            );
        }
    }
    push_suggestion(
        &mut suggestions,
        &context,
        "--help".into(),
        "--help".into(),
        "Print help for this command".into(),
        &context.prefix,
        true,
    );
    if command.get_name() == "bgm" {
        push_suggestion(
            &mut suggestions,
            &context,
            "--version".into(),
            "--version".into(),
            "Print the bgm version".into(),
            &context.prefix,
            true,
        );
    }
    sort_suggestions(&mut suggestions, &context.prefix);
    suggestions.dedup_by(|left, right| left.replacement == right.replacement);
    suggestions
}

pub(crate) fn command_value_suggestion(
    context: &CommandCompletionContext,
    value: &str,
    description: impl Into<String>,
) -> Option<CommandSuggestion> {
    if !matches_prefix(value, &context.prefix) {
        return None;
    }
    Some(CommandSuggestion {
        label: value.to_owned(),
        replacement: quote_command_argument(value),
        description: description.into(),
        replace_start: context.replace_start,
        replace_end: context.replace_end,
        append_space: true,
    })
}

pub(crate) fn parse_tui_command_line(input: &str) -> Result<Vec<String>> {
    let mut arguments = lex_command_line(input, true)?
        .tokens
        .into_iter()
        .map(|token| token.value)
        .collect::<Vec<_>>();
    if arguments.first().is_some_and(|word| word == "bgm") {
        arguments.remove(0);
    }
    if arguments.is_empty() {
        bail!("type a bgm command, for example `doctor` or `search --favorite`");
    }

    let parsed =
        Cli::try_parse_from(std::iter::once("bgm").chain(arguments.iter().map(String::as_str)));
    match parsed {
        Ok(Cli {
            command: None | Some(Command::Tui),
        }) => bail!("the command palette is already inside the TUI; choose another command"),
        Ok(_) => {}
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) => {}
        Err(error) => bail!("{}", error.to_string().trim()),
    }

    Ok(arguments
        .into_iter()
        .map(|value| expand_tilde(&value))
        .collect())
}

fn find_argument<'a>(command: &'a clap::Command, option: &str) -> Option<&'a clap::Arg> {
    command.get_arguments().find(|argument| {
        argument
            .get_long()
            .is_some_and(|long| option == format!("--{long}"))
            || argument
                .get_short()
                .is_some_and(|short| option == format!("-{short}"))
    })
}

#[allow(clippy::too_many_arguments)]
fn push_suggestion(
    suggestions: &mut Vec<CommandSuggestion>,
    context: &CommandCompletionContext,
    label: String,
    replacement: String,
    description: String,
    prefix: &str,
    append_space: bool,
) {
    if matches_prefix(&label, prefix) {
        suggestions.push(CommandSuggestion {
            label,
            replacement,
            description,
            replace_start: context.replace_start,
            replace_end: context.replace_end,
            append_space,
        });
    }
}

fn sort_suggestions(suggestions: &mut [CommandSuggestion], prefix: &str) {
    suggestions.sort_by_key(|suggestion| {
        let label = suggestion.label.to_ascii_lowercase();
        let prefix = prefix.to_ascii_lowercase();
        let rank = if label.starts_with(&prefix) { 0 } else { 1 };
        let option_rank = u8::from(label.starts_with('-'));
        (rank, option_rank, label)
    });
}

fn matches_prefix(value: &str, prefix: &str) -> bool {
    let value = value.to_ascii_lowercase();
    let prefix = prefix.to_ascii_lowercase();
    value.starts_with(&prefix) || (!prefix.is_empty() && value.contains(&prefix))
}

fn quote_command_argument(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-._/:@%+=,".contains(character))
    {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn expand_tilde(value: &str) -> String {
    let (option, raw_value) = value
        .split_once('=')
        .map_or((None, value), |(option, value)| (Some(option), value));
    let Some(rest) = raw_value.strip_prefix('~') else {
        return value.to_owned();
    };
    if !rest.is_empty() && !rest.starts_with('/') {
        return value.to_owned();
    }
    let Ok(user_home) = std::env::var("HOME") else {
        return value.to_owned();
    };
    let expanded = format!("{user_home}{rest}");
    option.map_or(expanded.clone(), |option| format!("{option}={expanded}"))
}

fn lex_command_line(input: &str, strict: bool) -> Result<LexedCommandLine> {
    let mut tokens = Vec::new();
    let mut value = String::new();
    let mut token_start = None;
    let mut quote = None;
    let mut escaped = false;
    let character_count = input.chars().count();

    for (index, character) in input.chars().enumerate() {
        if escaped {
            value.push(character);
            escaped = false;
            continue;
        }
        match quote {
            Some(Quote::Single) if character == '\'' => quote = None,
            Some(Quote::Single) => value.push(character),
            Some(Quote::Double) if character == '"' => quote = None,
            Some(Quote::Double) if character == '\\' => escaped = true,
            Some(Quote::Double) => value.push(character),
            None if character.is_whitespace() => {
                if let Some(start) = token_start.take() {
                    tokens.push(CommandToken {
                        value: std::mem::take(&mut value),
                        start,
                        end: index,
                    });
                }
            }
            None if character == '\'' => {
                token_start.get_or_insert(index);
                quote = Some(Quote::Single);
            }
            None if character == '"' => {
                token_start.get_or_insert(index);
                quote = Some(Quote::Double);
            }
            None if character == '\\' => {
                token_start.get_or_insert(index);
                escaped = true;
            }
            None => {
                token_start.get_or_insert(index);
                value.push(character);
            }
        }
    }

    if strict {
        if escaped {
            bail!("command ends with an unfinished escape");
        }
        if quote.is_some() {
            bail!("command contains an unclosed quote");
        }
    }
    let has_active_token = token_start.is_some();
    if let Some(start) = token_start {
        tokens.push(CommandToken {
            value,
            start,
            end: character_count,
        });
    }
    Ok(LexedCommandLine {
        tokens,
        has_active_token,
    })
}

struct AppContext {
    paths: AppPaths,
    config: Config,
    database: Database,
}

impl AppContext {
    fn load() -> Result<Self> {
        let (paths, config) = load_paths_and_config()?;
        let database = Database::open(&paths.database)?;
        Ok(Self {
            paths,
            config,
            database,
        })
    }
}

fn load_paths_and_config() -> Result<(AppPaths, Config)> {
    let paths = AppPaths::discover()?;
    paths.ensure_owned_dirs()?;
    let config = Config::load(&paths.config_file)?;
    match std::fs::symlink_metadata(&paths.config_file) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            config.save(&paths.config_file)?;
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect {}", paths.config_file.display()));
        }
    }
    Ok((paths, config))
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Command::Tui);
    match command {
        Command::Doctor { json } => {
            let (paths, _) = load_paths_and_config()?;
            command_doctor(&paths, json)
        }
        Command::Config { command } => {
            let (paths, mut config) = load_paths_and_config()?;
            command_config(&paths, &mut config, command)
        }
        Command::Model { command } => {
            let (paths, _) = load_paths_and_config()?;
            command_model(&paths, command)
        }
        command => run_catalog_command(&AppContext::load()?, command),
    }
}

fn run_catalog_command(context: &AppContext, command: Command) -> Result<()> {
    match command {
        Command::Tui => tui::run(&context.database, &context.paths, &context.config),
        Command::Source { command } => command_source(context, command),
        Command::Scan { full, no_ai, json } => command_scan(context, full, no_ai, json),
        Command::Label { command } => command_label(context, command),
        Command::Search(arguments) => command_search(context, arguments),
        Command::Collection { command } => command_collection(context, command),
        Command::Tag { command } => command_tag(context, command),
        Command::Favorite { command } => command_favorite(context, command),
        Command::Move(arguments) => command_move(context, arguments),
        Command::Wpaperd { command } => command_wpaperd(context, command),
        Command::Doctor { .. } | Command::Config { .. } | Command::Model { .. } => {
            unreachable!("lightweight commands are dispatched before opening the catalog")
        }
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

fn command_config(paths: &AppPaths, config: &mut Config, command: ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Show { json } => {
            if json {
                print_json(config)
            } else {
                print!("{}", toml::to_string_pretty(config)?);
                Ok(())
            }
        }
        ConfigCommand::Set { key, value } => {
            config.set(&key, &value)?;
            config.save(&paths.config_file)?;
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

fn command_model(paths: &AppPaths, command: ModelCommand) -> Result<()> {
    match command {
        ModelCommand::Install { yes } => {
            let status = model::install(paths, yes)?;
            println!(
                "verified {} at {}",
                status.model,
                status.directory.display()
            );
            Ok(())
        }
        ModelCommand::Status { verify, json } => {
            let status = model::status(paths, verify);
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
            if model::remove(paths)? {
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
            let report = wpaperd::refresh(&context.database, &context.paths, display.as_deref())?;
            if let Some(failures) = report.failure_summary() {
                bail!(
                    "refreshed {} binding(s); {} failed: {failures}",
                    report.refreshed.len(),
                    report.failures.len()
                );
            }
            println!("refreshed {} binding(s)", report.refreshed.len());
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
    let mut unique = HashSet::with_capacity(ids.len());
    for id in ids {
        if !unique.insert(*id) {
            bail!("duplicate image id: {id}");
        }
    }
    database.with_connection(|connection| {
        let images = load_images_by_id(connection, ids)?;
        if images.len() != ids.len() {
            let loaded = images.iter().map(|image| image.id).collect::<HashSet<_>>();
            let missing = ids
                .iter()
                .find(|id| !loaded.contains(*id))
                .context("an image disappeared while loading the move selection")?;
            bail!("image not found: {missing}");
        }
        Ok(images)
    })
}

fn refresh_after_change(context: &AppContext) {
    match wpaperd::refresh(&context.database, &context.paths, None) {
        Ok(report) => {
            if let Some(failures) = report.failure_summary() {
                eprintln!("warning: some wpaperd bindings were not refreshed: {failures}");
            }
        }
        Err(error) => eprintln!("warning: wpaperd bindings were not refreshed: {error:#}"),
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

    #[test]
    fn tui_command_line_supports_quotes_and_an_optional_binary_name() {
        let arguments =
            parse_tui_command_line("bgm search --semantic 'misty mountains at night' --favorite")
                .expect("valid command");

        assert_eq!(
            arguments,
            [
                "search",
                "--semantic",
                "misty mountains at night",
                "--favorite"
            ]
        );
    }

    #[test]
    fn tui_command_line_rejects_nested_tuis_and_invalid_syntax() {
        assert!(
            parse_tui_command_line("tui")
                .expect_err("nested TUI should fail")
                .to_string()
                .contains("already inside the TUI")
        );
        assert!(parse_tui_command_line("search --semantic 'unfinished").is_err());
        assert!(parse_tui_command_line("doctor; rm -rf /tmp/example").is_err());
    }

    #[test]
    fn command_completion_tracks_clap_subcommands_options_and_values() {
        let root = command_suggestions("col", 3);
        assert_eq!(
            root.iter()
                .map(|suggestion| suggestion.label.as_str())
                .collect::<Vec<_>>(),
            ["collection"]
        );
        assert!(
            command_suggestions("collection ", 11)
                .iter()
                .any(|suggestion| suggestion.label == "save")
        );
        let orientation = "search --orientation ";
        let values = command_suggestions(orientation, orientation.chars().count());
        for expected in ["landscape", "portrait", "square"] {
            assert!(
                values.iter().any(|suggestion| suggestion.label == expected),
                "missing {expected:?}"
            );
        }
    }

    #[test]
    fn completion_context_replaces_the_active_token_only() {
        let input = "collection sh --json";
        let context = command_completion_context(input, "collection sh".chars().count());

        assert_eq!(context.completed, ["collection"]);
        assert_eq!(context.prefix, "sh");
        assert_eq!((context.replace_start, context.replace_end), (11, 13));
    }
}
