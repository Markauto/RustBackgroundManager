use std::{
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use crossbeam_channel::{Receiver, Sender, TryRecvError, unbounded};
use crossterm::{
    cursor::Show,
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyModifiers,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
};
use ratatui_image::{Resize, StatefulImage, picker::Picker, protocol::StatefulProtocol};
use unicode_width::UnicodeWidthChar;

use crate::{
    AppPaths,
    analysis::{LightDark, Orientation},
    cli::{
        CommandCompletionContext, CommandSuggestion, command_completion_context,
        command_suggestions, command_value_suggestion, parse_tui_command_line,
    },
    collection::{
        SavedCollection, add_tag, delete_collection, list_collections, remove_tag, save_collection,
        search_resolved, set_favorite,
    },
    config::Config,
    db::{CatalogSummary, Database, ImageRecord},
    filter::{ColourFilter, FilterSpecV1},
    filter_completion::{FilterJsonCompletion, filter_json_completions},
    model,
    move_files::{MovePlan, MoveResult, apply_move, plan_move},
    scan::{ScanEvent, ScanOptions, ScanReport, scan_catalog_with_progress},
    wpaperd,
};

#[derive(Clone, Copy)]
enum InputAction {
    Source,
    Tag,
    Move,
    Bind,
}

enum Mode {
    Browse,
    FilterEditor(FilterEditor),
    Input {
        action: InputAction,
        value: String,
        error: Option<String>,
    },
    Collections(Box<CollectionsManager>),
    CommandPalette(CommandPalette),
    CommandOutput(CommandOutput),
    ConfirmMove(MovePlan),
    Help,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CollectionsFocus {
    List,
    Details,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CollectionDetailView {
    Summary,
    Json,
}

#[derive(Clone, Debug)]
struct CollectionNameEntry {
    value: String,
    cursor: usize,
}

#[derive(Clone, Debug)]
enum CollectionConfirmation {
    Overwrite { name: String, existing_name: String },
    Update(SavedCollection),
    Delete(SavedCollection),
}

#[derive(Clone, Debug)]
enum CollectionPrompt {
    Name(CollectionNameEntry),
    Confirm(Box<CollectionConfirmation>),
}

struct CollectionsManager {
    collections: Vec<SavedCollection>,
    bindings: Vec<wpaperd::Binding>,
    current_filter: FilterSpecV1,
    selected: usize,
    focus: CollectionsFocus,
    view: CollectionDetailView,
    detail_scroll: usize,
    detail_viewport_height: usize,
    prompt: Option<CollectionPrompt>,
    error: Option<String>,
    notice: Option<String>,
}

enum CollectionsManagerCommand {
    Continue,
    Close,
    Load(SavedCollection),
    Save(String),
    Update(SavedCollection),
    Delete(SavedCollection),
}

impl CollectionsManager {
    fn load(database: &Database, current_filter: &FilterSpecV1) -> Result<Self> {
        let collections = list_collections(database)?;
        let bindings = wpaperd::list_bindings(database)?;
        let selected = collections
            .iter()
            .position(|collection| collection.filter == *current_filter)
            .unwrap_or(0);
        Ok(Self {
            collections,
            bindings,
            current_filter: current_filter.clone(),
            selected,
            focus: CollectionsFocus::List,
            view: CollectionDetailView::Summary,
            detail_scroll: 0,
            detail_viewport_height: 1,
            prompt: None,
            error: None,
            notice: None,
        })
    }

    fn selected(&self) -> Option<&SavedCollection> {
        self.collections.get(self.selected)
    }

    fn bound_displays(&self, collection_id: i64) -> Vec<&str> {
        self.bindings
            .iter()
            .filter(|binding| binding.active && binding.collection_id == collection_id)
            .map(|binding| binding.display.as_str())
            .collect()
    }

    fn replace_store(
        &mut self,
        collections: Vec<SavedCollection>,
        bindings: Vec<wpaperd::Binding>,
        preferred_id: Option<i64>,
        fallback_selection: usize,
    ) {
        self.collections = collections;
        self.bindings = bindings;
        self.selected = preferred_id
            .and_then(|id| {
                self.collections
                    .iter()
                    .position(|collection| collection.id == id)
            })
            .unwrap_or_else(|| fallback_selection.min(self.collections.len().saturating_sub(1)));
        self.detail_scroll = 0;
    }

    fn handle_key(&mut self, key: KeyEvent) -> CollectionsManagerCommand {
        if self.prompt.is_some() {
            return self.handle_prompt_key(key);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('c') => CollectionsManagerCommand::Close,
            KeyCode::Tab | KeyCode::BackTab => {
                self.focus = match self.focus {
                    CollectionsFocus::List => CollectionsFocus::Details,
                    CollectionsFocus::Details => CollectionsFocus::List,
                };
                self.clear_feedback();
                CollectionsManagerCommand::Continue
            }
            KeyCode::Left => {
                self.focus = CollectionsFocus::List;
                self.clear_feedback();
                CollectionsManagerCommand::Continue
            }
            KeyCode::Right => {
                self.focus = CollectionsFocus::Details;
                self.clear_feedback();
                CollectionsManagerCommand::Continue
            }
            KeyCode::Char('v') => {
                self.view = match self.view {
                    CollectionDetailView::Summary => CollectionDetailView::Json,
                    CollectionDetailView::Json => CollectionDetailView::Summary,
                };
                self.detail_scroll = 0;
                self.clear_feedback();
                CollectionsManagerCommand::Continue
            }
            KeyCode::Char('s') => {
                self.prompt = Some(CollectionPrompt::Name(CollectionNameEntry {
                    value: String::new(),
                    cursor: 0,
                }));
                self.clear_feedback();
                CollectionsManagerCommand::Continue
            }
            KeyCode::Char('u') => {
                if let Some(collection) = self.selected().cloned() {
                    self.prompt = Some(CollectionPrompt::Confirm(Box::new(
                        CollectionConfirmation::Update(collection),
                    )));
                    self.clear_feedback();
                } else {
                    self.error = Some(
                        "There is no selected collection to update; press s to save one.".into(),
                    );
                    self.notice = None;
                }
                CollectionsManagerCommand::Continue
            }
            KeyCode::Char('d') => {
                let Some(collection) = self.selected().cloned() else {
                    self.error = Some(
                        "There is no selected collection to delete; press s to save one.".into(),
                    );
                    self.notice = None;
                    return CollectionsManagerCommand::Continue;
                };
                let displays = self.bound_displays(collection.id);
                if displays.is_empty() {
                    self.prompt = Some(CollectionPrompt::Confirm(Box::new(
                        CollectionConfirmation::Delete(collection),
                    )));
                    self.clear_feedback();
                } else {
                    self.error = Some(format!(
                        "Cannot delete {}: bound to wpaperd display(s) {}; unbind first.",
                        collection.name,
                        displays.join(", ")
                    ));
                    self.notice = None;
                }
                CollectionsManagerCommand::Continue
            }
            KeyCode::Enter => self.selected().cloned().map_or_else(
                || {
                    self.error = Some(
                        "There is no collection to load; press s to save the current filter."
                            .into(),
                    );
                    self.notice = None;
                    CollectionsManagerCommand::Continue
                },
                CollectionsManagerCommand::Load,
            ),
            KeyCode::Up | KeyCode::Char('k') => {
                self.navigate(-1);
                CollectionsManagerCommand::Continue
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.navigate(1);
                CollectionsManagerCommand::Continue
            }
            KeyCode::PageUp => {
                self.navigate(-self.page_size());
                CollectionsManagerCommand::Continue
            }
            KeyCode::PageDown => {
                self.navigate(self.page_size());
                CollectionsManagerCommand::Continue
            }
            KeyCode::Home | KeyCode::Char('g') => {
                if self.focus == CollectionsFocus::List {
                    self.select(0);
                } else {
                    self.detail_scroll = 0;
                    self.clear_feedback();
                }
                CollectionsManagerCommand::Continue
            }
            KeyCode::End | KeyCode::Char('G') => {
                if self.focus == CollectionsFocus::List {
                    self.select(self.collections.len().saturating_sub(1));
                } else {
                    self.detail_scroll = self.max_detail_scroll();
                    self.clear_feedback();
                }
                CollectionsManagerCommand::Continue
            }
            _ => CollectionsManagerCommand::Continue,
        }
    }

    fn handle_prompt_key(&mut self, key: KeyEvent) -> CollectionsManagerCommand {
        match self.prompt.as_mut() {
            Some(CollectionPrompt::Name(entry)) => match key.code {
                KeyCode::Esc => {
                    self.prompt = None;
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                KeyCode::Enter => CollectionsManagerCommand::Save(entry.value.clone()),
                KeyCode::Left => {
                    entry.cursor = entry.cursor.saturating_sub(1);
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                KeyCode::Right => {
                    entry.cursor = (entry.cursor + 1).min(entry.value.chars().count());
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                KeyCode::Home => {
                    entry.cursor = 0;
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                KeyCode::End => {
                    entry.cursor = entry.value.chars().count();
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                KeyCode::Backspace => {
                    if entry.cursor > 0 {
                        replace_chars(&mut entry.value, entry.cursor - 1, entry.cursor, "");
                        entry.cursor -= 1;
                    }
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                KeyCode::Delete => {
                    if entry.cursor < entry.value.chars().count() {
                        replace_chars(&mut entry.value, entry.cursor, entry.cursor + 1, "");
                    }
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                KeyCode::Char(character)
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !character.is_control() =>
                {
                    let mut text = String::new();
                    text.push(character);
                    replace_chars(&mut entry.value, entry.cursor, entry.cursor, &text);
                    entry.cursor += 1;
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                _ => CollectionsManagerCommand::Continue,
            },
            Some(CollectionPrompt::Confirm(confirmation)) => match key.code {
                KeyCode::Esc | KeyCode::Char('n' | 'N') => {
                    self.prompt = None;
                    self.error = None;
                    CollectionsManagerCommand::Continue
                }
                KeyCode::Enter | KeyCode::Char('y' | 'Y') => match confirmation.as_ref().clone() {
                    CollectionConfirmation::Overwrite { name, .. } => {
                        CollectionsManagerCommand::Save(name)
                    }
                    CollectionConfirmation::Update(collection) => {
                        CollectionsManagerCommand::Update(collection)
                    }
                    CollectionConfirmation::Delete(collection) => {
                        CollectionsManagerCommand::Delete(collection)
                    }
                },
                _ => CollectionsManagerCommand::Continue,
            },
            None => CollectionsManagerCommand::Continue,
        }
    }

    fn paste(&mut self, value: &str) {
        let Some(CollectionPrompt::Name(entry)) = self.prompt.as_mut() else {
            return;
        };
        let value = value
            .chars()
            .filter(|character| !matches!(character, '\r' | '\n') && !character.is_control())
            .collect::<String>();
        replace_chars(&mut entry.value, entry.cursor, entry.cursor, &value);
        entry.cursor += value.chars().count();
        self.error = None;
    }

    fn navigate(&mut self, amount: isize) {
        if self.focus == CollectionsFocus::List {
            let selected = self
                .selected
                .saturating_add_signed(amount)
                .min(self.collections.len().saturating_sub(1));
            self.select(selected);
        } else {
            self.detail_scroll = self
                .detail_scroll
                .saturating_add_signed(amount)
                .min(self.max_detail_scroll());
            self.clear_feedback();
        }
    }

    fn select(&mut self, selected: usize) {
        self.selected = selected.min(self.collections.len().saturating_sub(1));
        self.detail_scroll = 0;
        self.clear_feedback();
    }

    fn page_size(&self) -> isize {
        if self.focus == CollectionsFocus::List {
            10
        } else {
            isize::try_from(self.detail_viewport_height.saturating_sub(1).max(1))
                .unwrap_or(isize::MAX)
        }
    }

    fn max_detail_scroll(&self) -> usize {
        collection_detail_lines(self)
            .len()
            .saturating_sub(self.detail_viewport_height.max(1))
    }

    fn clear_feedback(&mut self) {
        self.error = None;
        self.notice = None;
    }
}

struct CommandPalette {
    value: String,
    cursor: usize,
    suggestions: Vec<CommandSuggestion>,
    selected_suggestion: usize,
    error: Option<String>,
    running: bool,
    history_index: Option<usize>,
    history_draft: String,
    selected_image_id: Option<i64>,
    selected_tags: Vec<String>,
}

struct CommandOutput {
    command: String,
    success: bool,
    stdout: String,
    stderr: String,
    scroll: usize,
    viewport_height: usize,
}

enum FilterEditorCommand {
    Continue,
    Apply,
    Reset,
    LoadPreset {
        name: String,
        filter: Box<FilterSpecV1>,
    },
    SavePreset(String),
    Cancel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FilterEditorFocus {
    Document,
    Presets,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FilterPresetSource {
    Example,
    Saved,
}

impl FilterPresetSource {
    const fn label(self) -> &'static str {
        match self {
            Self::Example => "Example",
            Self::Saved => "Saved collection",
        }
    }
}

#[derive(Clone, Debug)]
struct FilterPreset {
    name: String,
    description: String,
    source: FilterPresetSource,
    filter: FilterSpecV1,
}

impl FilterPreset {
    fn example(name: &str, description: &str, filter: FilterSpecV1) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            source: FilterPresetSource::Example,
            filter,
        }
    }
}

fn filter_presets(saved: Vec<SavedCollection>) -> Vec<FilterPreset> {
    let mut presets = vec![
        FilterPreset::example(
            "All wallpapers",
            "No facets; show every ready image.",
            FilterSpecV1::default(),
        ),
        FilterPreset::example(
            "Dark landscapes",
            "Landscape orientation AND dark analysis.",
            FilterSpecV1 {
                orientations: vec![Orientation::Landscape],
                light_dark: vec![LightDark::Dark],
                ..FilterSpecV1::default()
            },
        ),
        FilterPreset::example(
            "Large wallpapers",
            "At least 2560 × 1440 pixels.",
            FilterSpecV1 {
                min_width: Some(2560),
                min_height: Some(1440),
                ..FilterSpecV1::default()
            },
        ),
        FilterPreset::example(
            "Ultrawide",
            "Landscape images near 21:9 (5% tolerance).",
            FilterSpecV1 {
                orientations: vec![Orientation::Landscape],
                aspect_ratios: vec![21.0 / 9.0],
                aspect_tolerance: 0.05,
                ..FilterSpecV1::default()
            },
        ),
        FilterPreset::example(
            "Warm palette",
            "Any palette colour near warm orange in Oklab.",
            FilterSpecV1 {
                palette_colours: vec![ColourFilter {
                    hex: "#D08040".into(),
                    max_distance: 0.10,
                }],
                ..FilterSpecV1::default()
            },
        ),
        FilterPreset::example(
            "Favourites",
            "Only images marked as a favourite.",
            FilterSpecV1 {
                favorite: Some(true),
                ..FilterSpecV1::default()
            },
        ),
    ];
    presets.extend(saved.into_iter().map(|collection| FilterPreset {
        name: collection.name,
        description: "A filter saved here or with `bgm collection save`.".into(),
        source: FilterPresetSource::Saved,
        filter: collection.filter,
    }));
    presets
}

struct FilterEditor {
    lines: Vec<String>,
    cursor_line: usize,
    cursor_column: usize,
    preferred_column: Option<usize>,
    scroll_line: usize,
    scroll_column: usize,
    viewport_height: usize,
    presets: Vec<FilterPreset>,
    selected_preset: usize,
    focus: FilterEditorFocus,
    save_name: Option<String>,
    error: Option<String>,
    notice: Option<String>,
    completions: Option<FilterCompletionState>,
}

struct FilterCompletionState {
    items: Vec<FilterJsonCompletion>,
    selected: usize,
}

impl FilterEditor {
    fn new(value: String, saved: Vec<SavedCollection>) -> Self {
        let lines = value.split('\n').map(str::to_owned).collect();
        Self {
            lines,
            cursor_line: 0,
            cursor_column: 0,
            preferred_column: None,
            scroll_line: 0,
            scroll_column: 0,
            viewport_height: 10,
            presets: filter_presets(saved),
            selected_preset: 0,
            focus: FilterEditorFocus::Document,
            save_name: None,
            error: None,
            notice: None,
            completions: None,
        }
    }

    fn value(&self) -> String {
        self.lines.join("\n")
    }

    fn replace_document(&mut self, value: String) {
        self.lines = value.split('\n').map(str::to_owned).collect();
        self.cursor_line = 0;
        self.cursor_column = 0;
        self.preferred_column = None;
        self.scroll_line = 0;
        self.scroll_column = 0;
        self.error = None;
        self.completions = None;
    }

    fn handle_key(&mut self, key: KeyEvent) -> FilterEditorCommand {
        if self.save_name.is_some() {
            return self.handle_save_name_key(key);
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        if self.completions.is_some() {
            match key.code {
                KeyCode::Esc => {
                    self.completions = None;
                    return FilterEditorCommand::Continue;
                }
                KeyCode::Up => {
                    self.select_previous_completion();
                    return FilterEditorCommand::Continue;
                }
                KeyCode::Down => {
                    self.select_next_completion();
                    return FilterEditorCommand::Continue;
                }
                KeyCode::PageUp | KeyCode::BackTab => {
                    self.move_completion_selection(-5);
                    return FilterEditorCommand::Continue;
                }
                KeyCode::PageDown => {
                    self.move_completion_selection(5);
                    return FilterEditorCommand::Continue;
                }
                KeyCode::Enter | KeyCode::Tab => {
                    self.accept_completion();
                    return FilterEditorCommand::Continue;
                }
                KeyCode::Char(' ') | KeyCode::Null if control => {
                    self.open_completions();
                    return FilterEditorCommand::Continue;
                }
                _ => {}
            }
        }
        if control
            && self.focus == FilterEditorFocus::Document
            && matches!(key.code, KeyCode::Char(' ') | KeyCode::Null)
        {
            self.open_completions();
            return FilterEditorCommand::Continue;
        }
        match key.code {
            KeyCode::Esc => FilterEditorCommand::Cancel,
            KeyCode::Char('s' | 'S') if control => {
                self.completions = None;
                FilterEditorCommand::Apply
            }
            KeyCode::Enter if control => {
                self.completions = None;
                FilterEditorCommand::Apply
            }
            KeyCode::Char('r' | 'R') if control => {
                self.completions = None;
                FilterEditorCommand::Reset
            }
            KeyCode::Char('p' | 'P') if control => {
                self.begin_save_preset();
                FilterEditorCommand::Continue
            }
            KeyCode::Tab | KeyCode::BackTab => {
                self.focus = match self.focus {
                    FilterEditorFocus::Document => FilterEditorFocus::Presets,
                    FilterEditorFocus::Presets => FilterEditorFocus::Document,
                };
                self.error = None;
                self.completions = None;
                FilterEditorCommand::Continue
            }
            _ if self.focus == FilterEditorFocus::Presets => self.handle_preset_key(key),
            KeyCode::Home if control => {
                self.cursor_line = 0;
                self.cursor_column = 0;
                self.preferred_column = None;
                self.completions = None;
                FilterEditorCommand::Continue
            }
            KeyCode::End if control => {
                self.cursor_line = self.lines.len() - 1;
                self.cursor_column = self.current_line_len();
                self.preferred_column = None;
                self.completions = None;
                FilterEditorCommand::Continue
            }
            KeyCode::Left => {
                self.move_left();
                FilterEditorCommand::Continue
            }
            KeyCode::Right => {
                self.move_right();
                FilterEditorCommand::Continue
            }
            KeyCode::Up => {
                self.move_vertical(-1);
                FilterEditorCommand::Continue
            }
            KeyCode::Down => {
                self.move_vertical(1);
                FilterEditorCommand::Continue
            }
            KeyCode::PageUp => {
                self.move_vertical(-self.page_size());
                FilterEditorCommand::Continue
            }
            KeyCode::PageDown => {
                self.move_vertical(self.page_size());
                FilterEditorCommand::Continue
            }
            KeyCode::Home => {
                self.cursor_column = 0;
                self.preferred_column = None;
                self.completions = None;
                FilterEditorCommand::Continue
            }
            KeyCode::End => {
                self.cursor_column = self.current_line_len();
                self.preferred_column = None;
                self.completions = None;
                FilterEditorCommand::Continue
            }
            KeyCode::Backspace => {
                self.backspace();
                FilterEditorCommand::Continue
            }
            KeyCode::Delete => {
                self.delete();
                FilterEditorCommand::Continue
            }
            KeyCode::Enter => {
                self.insert_newline();
                FilterEditorCommand::Continue
            }
            KeyCode::Char(character) if !control => {
                self.insert_char(character);
                FilterEditorCommand::Continue
            }
            _ => FilterEditorCommand::Continue,
        }
    }

    fn handle_preset_key(&mut self, key: KeyEvent) -> FilterEditorCommand {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_preset = self.selected_preset.saturating_sub(1);
                FilterEditorCommand::Continue
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_preset =
                    (self.selected_preset + 1).min(self.presets.len().saturating_sub(1));
                FilterEditorCommand::Continue
            }
            KeyCode::PageUp => {
                self.selected_preset = self.selected_preset.saturating_sub(5);
                FilterEditorCommand::Continue
            }
            KeyCode::PageDown => {
                self.selected_preset =
                    (self.selected_preset + 5).min(self.presets.len().saturating_sub(1));
                FilterEditorCommand::Continue
            }
            KeyCode::Home => {
                self.selected_preset = 0;
                FilterEditorCommand::Continue
            }
            KeyCode::End => {
                self.selected_preset = self.presets.len().saturating_sub(1);
                FilterEditorCommand::Continue
            }
            KeyCode::Enter => self.presets.get(self.selected_preset).map_or(
                FilterEditorCommand::Continue,
                |preset| FilterEditorCommand::LoadPreset {
                    name: preset.name.clone(),
                    filter: Box::new(preset.filter.clone()),
                },
            ),
            KeyCode::Char('s' | 'S') => {
                self.begin_save_preset();
                FilterEditorCommand::Continue
            }
            KeyCode::Right | KeyCode::Char('e') => {
                self.focus = FilterEditorFocus::Document;
                FilterEditorCommand::Continue
            }
            _ => FilterEditorCommand::Continue,
        }
    }

    fn begin_save_preset(&mut self) {
        let existing_name = self
            .presets
            .get(self.selected_preset)
            .filter(|preset| {
                self.focus == FilterEditorFocus::Presets
                    && preset.source == FilterPresetSource::Saved
            })
            .map_or_else(String::new, |preset| preset.name.clone());
        self.save_name = Some(existing_name);
        self.error = None;
        self.notice = None;
        self.completions = None;
    }

    fn handle_save_name_key(&mut self, key: KeyEvent) -> FilterEditorCommand {
        match key.code {
            KeyCode::Esc => {
                self.save_name = None;
                self.error = None;
                FilterEditorCommand::Continue
            }
            KeyCode::Enter => {
                let name = self.save_name.take().unwrap_or_default();
                if name.trim().is_empty() {
                    self.error = Some("preset name cannot be empty".into());
                    self.save_name = Some(name);
                    FilterEditorCommand::Continue
                } else {
                    FilterEditorCommand::SavePreset(name)
                }
            }
            KeyCode::Backspace => {
                if let Some(name) = &mut self.save_name {
                    name.pop();
                }
                self.error = None;
                FilterEditorCommand::Continue
            }
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if !character.is_control()
                    && let Some(name) = &mut self.save_name
                {
                    name.push(character);
                }
                self.error = None;
                FilterEditorCommand::Continue
            }
            _ => FilterEditorCommand::Continue,
        }
    }

    fn paste(&mut self, value: &str) {
        if let Some(name) = &mut self.save_name {
            name.extend(value.chars().filter(|character| !character.is_control()));
            self.error = None;
        } else if self.focus == FilterEditorFocus::Document {
            self.insert_text(value);
        }
    }

    fn refresh_saved_presets(&mut self, saved: Vec<SavedCollection>, selected_name: &str) {
        self.presets = filter_presets(saved);
        self.selected_preset = self
            .presets
            .iter()
            .position(|preset| {
                preset.source == FilterPresetSource::Saved
                    && preset.name.eq_ignore_ascii_case(selected_name)
            })
            .unwrap_or(0);
    }

    fn insert_text(&mut self, value: &str) {
        for character in value.chars() {
            match character {
                '\r' => {}
                '\n' => self.insert_newline(),
                '\t' => {
                    self.insert_char(' ');
                    self.insert_char(' ');
                }
                character if !character.is_control() => self.insert_char(character),
                _ => {}
            }
        }
    }

    fn insert_char(&mut self, character: char) {
        let byte_index = char_to_byte_index(&self.lines[self.cursor_line], self.cursor_column);
        self.lines[self.cursor_line].insert(byte_index, character);
        self.cursor_column += 1;
        self.changed();
        if character.is_alphanumeric() || matches!(character, '"' | '_' | '-' | '.' | '#' | '/') {
            self.refresh_completions();
        } else {
            self.completions = None;
        }
    }

    fn insert_newline(&mut self) {
        let byte_index = char_to_byte_index(&self.lines[self.cursor_line], self.cursor_column);
        let remainder = self.lines[self.cursor_line].split_off(byte_index);
        self.cursor_line += 1;
        self.lines.insert(self.cursor_line, remainder);
        self.cursor_column = 0;
        self.changed();
        self.completions = None;
    }

    fn backspace(&mut self) {
        if self.cursor_column > 0 {
            self.cursor_column -= 1;
            let byte_index = char_to_byte_index(&self.lines[self.cursor_line], self.cursor_column);
            self.lines[self.cursor_line].remove(byte_index);
            self.changed();
            self.refresh_completions();
        } else if self.cursor_line > 0 {
            let current = self.lines.remove(self.cursor_line);
            self.cursor_line -= 1;
            self.cursor_column = self.current_line_len();
            self.lines[self.cursor_line].push_str(&current);
            self.changed();
            self.refresh_completions();
        }
    }

    fn delete(&mut self) {
        if self.cursor_column < self.current_line_len() {
            let byte_index = char_to_byte_index(&self.lines[self.cursor_line], self.cursor_column);
            self.lines[self.cursor_line].remove(byte_index);
            self.changed();
            self.refresh_completions();
        } else if self.cursor_line + 1 < self.lines.len() {
            let next = self.lines.remove(self.cursor_line + 1);
            self.lines[self.cursor_line].push_str(&next);
            self.changed();
            self.refresh_completions();
        }
    }

    fn move_left(&mut self) {
        if self.cursor_column > 0 {
            self.cursor_column -= 1;
        } else if self.cursor_line > 0 {
            self.cursor_line -= 1;
            self.cursor_column = self.current_line_len();
        }
        self.preferred_column = None;
        self.completions = None;
    }

    fn move_right(&mut self) {
        if self.cursor_column < self.current_line_len() {
            self.cursor_column += 1;
        } else if self.cursor_line + 1 < self.lines.len() {
            self.cursor_line += 1;
            self.cursor_column = 0;
        }
        self.preferred_column = None;
        self.completions = None;
    }

    fn move_vertical(&mut self, amount: isize) {
        let preferred = self.preferred_column.unwrap_or(self.cursor_column);
        self.cursor_line = self
            .cursor_line
            .saturating_add_signed(amount)
            .min(self.lines.len() - 1);
        self.cursor_column = preferred.min(self.current_line_len());
        self.preferred_column = Some(preferred);
        self.completions = None;
    }

    fn page_size(&self) -> isize {
        isize::try_from(self.viewport_height.saturating_sub(1).max(1)).unwrap_or(isize::MAX)
    }

    fn current_line_len(&self) -> usize {
        self.lines[self.cursor_line].chars().count()
    }

    fn changed(&mut self) {
        self.preferred_column = None;
        self.error = None;
        self.notice = None;
    }

    fn open_completions(&mut self) {
        self.refresh_completions();
        if self.completions.is_none() {
            self.notice = Some("No FilterSpecV1 completions are available here.".into());
        } else {
            self.notice = None;
            self.error = None;
        }
    }

    fn refresh_completions(&mut self) {
        let items = filter_json_completions(&self.value(), self.cursor_offset());
        if items.is_empty() {
            self.completions = None;
            return;
        }
        let selected = self
            .completions
            .as_ref()
            .map_or(0, |completion| completion.selected.min(items.len() - 1));
        self.completions = Some(FilterCompletionState { items, selected });
    }

    fn select_previous_completion(&mut self) {
        if let Some(completions) = &mut self.completions {
            completions.selected = completions
                .selected
                .checked_sub(1)
                .unwrap_or(completions.items.len() - 1);
        }
    }

    fn select_next_completion(&mut self) {
        if let Some(completions) = &mut self.completions {
            completions.selected = (completions.selected + 1) % completions.items.len();
        }
    }

    fn move_completion_selection(&mut self, amount: isize) {
        if let Some(completions) = &mut self.completions {
            completions.selected = completions
                .selected
                .saturating_add_signed(amount)
                .min(completions.items.len() - 1);
        }
    }

    fn accept_completion(&mut self) {
        let Some(completion) = self
            .completions
            .as_ref()
            .and_then(|completions| completions.items.get(completions.selected).cloned())
        else {
            self.completions = None;
            return;
        };
        let mut value = self.value();
        let start = char_to_byte_index(&value, completion.replace_start);
        let end = char_to_byte_index(&value, completion.replace_end);
        value.replace_range(start..end, &completion.replacement);
        let cursor = completion.replace_start + completion.cursor_after;
        self.lines = value.split('\n').map(str::to_owned).collect();
        self.set_cursor_offset(cursor);
        self.completions = None;
        self.preferred_column = None;
        self.error = None;
        self.notice = Some(format!("Inserted {}.", completion.label));
    }

    fn cursor_offset(&self) -> usize {
        self.lines
            .iter()
            .take(self.cursor_line)
            .map(|line| line.chars().count() + 1)
            .sum::<usize>()
            + self.cursor_column
    }

    fn set_cursor_offset(&mut self, offset: usize) {
        let mut remaining = offset;
        for (line_index, line) in self.lines.iter().enumerate() {
            let length = line.chars().count();
            if remaining <= length {
                self.cursor_line = line_index;
                self.cursor_column = remaining;
                return;
            }
            remaining = remaining.saturating_sub(length + 1);
        }
        self.cursor_line = self.lines.len() - 1;
        self.cursor_column = self.current_line_len();
    }

    fn ensure_cursor_visible(&mut self, width: usize, height: usize) {
        self.viewport_height = height.max(1);
        if self.cursor_line < self.scroll_line {
            self.scroll_line = self.cursor_line;
        } else if self.cursor_line >= self.scroll_line + self.viewport_height {
            self.scroll_line = self.cursor_line + 1 - self.viewport_height;
        }

        if self.cursor_column < self.scroll_column {
            self.scroll_column = self.cursor_column;
        }
        let line = &self.lines[self.cursor_line];
        while display_width(line, self.scroll_column, self.cursor_column) >= width.max(1)
            && self.scroll_column < self.cursor_column
        {
            self.scroll_column += 1;
        }
    }
}

fn char_to_byte_index(value: &str, char_index: usize) -> usize {
    value
        .char_indices()
        .nth(char_index)
        .map_or(value.len(), |(index, _)| index)
}

fn replace_chars(value: &mut String, start: usize, end: usize, replacement: &str) {
    let start = char_to_byte_index(value, start);
    let end = char_to_byte_index(value, end);
    value.replace_range(start..end, replacement);
}

fn display_width(value: &str, start: usize, end: usize) -> usize {
    value
        .chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .map(|character| character.width().unwrap_or_default())
        .sum()
}

fn visible_text(value: &str, start: usize, max_width: usize) -> String {
    let mut width = 0;
    value
        .chars()
        .skip(start)
        .take_while(|character| {
            let next_width = width + character.width().unwrap_or_default();
            if next_width > max_width {
                return false;
            }
            width = next_width;
            true
        })
        .collect()
}

enum CommandPaletteAction {
    Continue,
    Cancel,
    Submit(String),
}

impl CommandPalette {
    fn new(database: &Database, selected: Option<&ImageRecord>) -> Self {
        let mut palette = Self {
            value: String::new(),
            cursor: 0,
            suggestions: Vec::new(),
            selected_suggestion: 0,
            error: None,
            running: false,
            history_index: None,
            history_draft: String::new(),
            selected_image_id: selected.map(|image| image.id),
            selected_tags: selected.map_or_else(Vec::new, |image| image.tags.clone()),
        };
        palette.refresh_suggestions(database);
        palette
    }

    fn handle_key(
        &mut self,
        key: KeyEvent,
        history: &[String],
        database: &Database,
    ) -> CommandPaletteAction {
        if self.running {
            return if key.code == KeyCode::Esc {
                CommandPaletteAction::Cancel
            } else {
                CommandPaletteAction::Continue
            };
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => CommandPaletteAction::Cancel,
            KeyCode::Enter => CommandPaletteAction::Submit(self.value.clone()),
            KeyCode::Tab => {
                self.accept_suggestion(database);
                CommandPaletteAction::Continue
            }
            KeyCode::BackTab => {
                if !self.suggestions.is_empty() {
                    self.selected_suggestion = self
                        .selected_suggestion
                        .checked_sub(1)
                        .unwrap_or(self.suggestions.len() - 1);
                }
                CommandPaletteAction::Continue
            }
            KeyCode::Up if !control => {
                if !self.suggestions.is_empty() {
                    self.selected_suggestion = self
                        .selected_suggestion
                        .checked_sub(1)
                        .unwrap_or(self.suggestions.len() - 1);
                }
                CommandPaletteAction::Continue
            }
            KeyCode::Down if !control => {
                if !self.suggestions.is_empty() {
                    self.selected_suggestion =
                        (self.selected_suggestion + 1) % self.suggestions.len();
                }
                CommandPaletteAction::Continue
            }
            KeyCode::Char('p' | 'P') if control => {
                self.previous_history(history, database);
                CommandPaletteAction::Continue
            }
            KeyCode::Char('n' | 'N') if control => {
                self.next_history(history, database);
                CommandPaletteAction::Continue
            }
            KeyCode::Home => {
                self.cursor = 0;
                self.refresh_suggestions(database);
                CommandPaletteAction::Continue
            }
            KeyCode::Char('a' | 'A') if control => {
                self.cursor = 0;
                self.refresh_suggestions(database);
                CommandPaletteAction::Continue
            }
            KeyCode::End => {
                self.cursor = self.value.chars().count();
                self.refresh_suggestions(database);
                CommandPaletteAction::Continue
            }
            KeyCode::Char('e' | 'E') if control => {
                self.cursor = self.value.chars().count();
                self.refresh_suggestions(database);
                CommandPaletteAction::Continue
            }
            KeyCode::Char('u' | 'U') if control => {
                self.replace_chars(0, self.cursor, "");
                self.cursor = 0;
                self.edited(database);
                CommandPaletteAction::Continue
            }
            KeyCode::Char('w' | 'W') if control => {
                let mut start = self.cursor;
                let characters = self.value.chars().collect::<Vec<_>>();
                while start > 0 && characters[start - 1].is_whitespace() {
                    start -= 1;
                }
                while start > 0 && !characters[start - 1].is_whitespace() {
                    start -= 1;
                }
                self.replace_chars(start, self.cursor, "");
                self.cursor = start;
                self.edited(database);
                CommandPaletteAction::Continue
            }
            KeyCode::Left => {
                self.cursor = self.cursor.saturating_sub(1);
                self.refresh_suggestions(database);
                CommandPaletteAction::Continue
            }
            KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.value.chars().count());
                self.refresh_suggestions(database);
                CommandPaletteAction::Continue
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.replace_chars(self.cursor - 1, self.cursor, "");
                    self.cursor -= 1;
                    self.edited(database);
                }
                CommandPaletteAction::Continue
            }
            KeyCode::Delete => {
                if self.cursor < self.value.chars().count() {
                    self.replace_chars(self.cursor, self.cursor + 1, "");
                    self.edited(database);
                }
                CommandPaletteAction::Continue
            }
            KeyCode::Char(character) if !control && !character.is_control() => {
                let mut text = String::new();
                text.push(character);
                self.replace_chars(self.cursor, self.cursor, &text);
                self.cursor += 1;
                self.edited(database);
                CommandPaletteAction::Continue
            }
            _ => CommandPaletteAction::Continue,
        }
    }

    fn paste(&mut self, value: &str, database: &Database) {
        if self.running {
            return;
        }
        let value = value
            .chars()
            .filter(|character| !matches!(character, '\r' | '\n') && !character.is_control())
            .collect::<String>();
        self.replace_chars(self.cursor, self.cursor, &value);
        self.cursor += value.chars().count();
        self.edited(database);
    }

    fn accept_suggestion(&mut self, database: &Database) {
        let Some(suggestion) = self.suggestions.get(self.selected_suggestion).cloned() else {
            return;
        };
        self.replace_chars(
            suggestion.replace_start,
            suggestion.replace_end,
            &suggestion.replacement,
        );
        self.cursor = suggestion.replace_start + suggestion.replacement.chars().count();
        if suggestion.append_space
            && self
                .value
                .chars()
                .nth(self.cursor)
                .is_none_or(|character| !character.is_whitespace())
        {
            self.replace_chars(self.cursor, self.cursor, " ");
            self.cursor += 1;
        }
        self.edited(database);
    }

    fn previous_history(&mut self, history: &[String], database: &Database) {
        if history.is_empty() {
            return;
        }
        let index = match self.history_index {
            Some(index) => index.saturating_sub(1),
            None => {
                self.history_draft.clone_from(&self.value);
                history.len() - 1
            }
        };
        self.history_index = Some(index);
        self.value.clone_from(&history[index]);
        self.cursor = self.value.chars().count();
        self.error = None;
        self.refresh_suggestions(database);
    }

    fn next_history(&mut self, history: &[String], database: &Database) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < history.len() {
            self.history_index = Some(index + 1);
            self.value.clone_from(&history[index + 1]);
        } else {
            self.history_index = None;
            self.value.clone_from(&self.history_draft);
        }
        self.cursor = self.value.chars().count();
        self.error = None;
        self.refresh_suggestions(database);
    }

    fn edited(&mut self, database: &Database) {
        self.error = None;
        self.history_index = None;
        self.refresh_suggestions(database);
    }

    fn replace_chars(&mut self, start: usize, end: usize, replacement: &str) {
        let start = char_to_byte_index(&self.value, start);
        let end = char_to_byte_index(&self.value, end);
        self.value.replace_range(start..end, replacement);
    }

    fn refresh_suggestions(&mut self, database: &Database) {
        self.suggestions = palette_suggestions(
            &self.value,
            self.cursor,
            database,
            self.selected_image_id,
            &self.selected_tags,
        );
        self.selected_suggestion = self
            .selected_suggestion
            .min(self.suggestions.len().saturating_sub(1));
    }
}

impl CommandOutput {
    fn line_count(&self) -> usize {
        let mut lines = self.stdout.lines().count();
        if !self.stderr.is_empty() {
            lines += self.stderr.lines().count() + 1;
            lines += usize::from(!self.stdout.is_empty());
        }
        lines.max(1)
    }

    fn max_scroll(&self) -> usize {
        self.line_count()
            .saturating_sub(self.viewport_height.max(1))
    }

    fn move_scroll(&mut self, amount: isize) {
        self.scroll = self
            .scroll
            .saturating_add_signed(amount)
            .min(self.max_scroll());
    }
}

fn palette_suggestions(
    input: &str,
    cursor: usize,
    database: &Database,
    selected_image_id: Option<i64>,
    selected_tags: &[String],
) -> Vec<CommandSuggestion> {
    let mut suggestions = command_suggestions(input, cursor);
    let context = command_completion_context(input, cursor);
    add_contextual_suggestions(
        &mut suggestions,
        &context,
        database,
        selected_image_id,
        selected_tags,
    );
    suggestions.sort_by_key(|suggestion| {
        (
            u8::from(suggestion.label.starts_with('-')),
            suggestion.label.to_ascii_lowercase(),
        )
    });
    suggestions.dedup_by(|left, right| left.replacement == right.replacement);
    suggestions
}

fn add_contextual_suggestions(
    suggestions: &mut Vec<CommandSuggestion>,
    context: &CommandCompletionContext,
    database: &Database,
    selected_image_id: Option<i64>,
    selected_tags: &[String],
) {
    const CONFIG_KEYS: &[&str] = &[
        "ai.enabled",
        "analysis.common_ratio_tolerance",
        "analysis.dark_threshold",
        "analysis.palette_colors",
        "analysis.thumbnail_long_edge",
        "import.max_height",
        "import.max_width",
        "import.min_height",
        "import.min_width",
    ];
    const DISPLAYS: &[&str] = &["any", "DP-1", "DP-2", "HDMI-A-1"];

    let words = context
        .completed
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let mut add = |value: &str, description: &str| {
        if let Some(suggestion) = command_value_suggestion(context, value, description) {
            suggestions.push(suggestion);
        }
    };
    match words.as_slice() {
        ["config", "set"] => {
            for key in CONFIG_KEYS {
                add(key, "configuration key");
            }
        }
        ["config", "set", "ai.enabled"] => {
            add("true", "enable local AI analysis");
            add("false", "disable local AI analysis");
        }
        ["source", "remove"] => {
            if let Ok(sources) = database.list_sources() {
                for source in sources {
                    add(
                        &source.path.to_string_lossy(),
                        "registered source directory",
                    );
                }
            }
        }
        ["collection", "show" | "delete"] | ["wpaperd", "bind", _] => {
            if let Ok(collections) = list_collections(database) {
                for collection in collections {
                    add(&collection.name, "saved collection");
                }
            }
        }
        ["wpaperd", "bind"] | ["wpaperd", "refresh"] => {
            for display in DISPLAYS {
                add(display, "wpaperd display");
            }
        }
        ["wpaperd", "unbind"] => {
            if let Ok(bindings) = wpaperd::list_bindings(database) {
                for binding in bindings {
                    add(&binding.display, "active wpaperd binding");
                }
            }
        }
        ["label", "delete" | "rescore"] => {
            if let Ok(packs) = model::list_label_packs(database) {
                for pack in packs {
                    add(&pack.name, "saved label pack");
                }
            }
        }
        ["search", "--tag"] | ["collection", "save", _, "--tag"] => {
            if let Ok(tags) = catalog_tags(database) {
                for tag in tags {
                    add(&tag, "catalog tag");
                }
            }
        }
        ["tag", "remove"] => {
            for tag in selected_tags {
                add(tag, "tag on the selected image");
            }
        }
        ["tag", "add" | "remove", _] | ["favorite", "set" | "unset"] | [.., "--image-id"] => {
            if let Some(id) = selected_image_id {
                add(&id.to_string(), "selected image ID");
            }
        }
        _ => {}
    }
}

fn catalog_tags(database: &Database) -> Result<Vec<String>> {
    database.with_connection(|connection| {
        let mut statement = connection.prepare("SELECT name FROM tags ORDER BY name")?;
        let rows = statement.query_map([], |row| row.get(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    })
}

enum BackgroundResult {
    Progress(ScanEvent),
    Finished {
        scan: ScanReport,
        ai: Option<model::AiReport>,
        wpaperd_warning: Option<String>,
    },
    Failed(String),
}

enum CommandBackgroundResult {
    Finished {
        success: bool,
        stdout: String,
        stderr: String,
    },
    Failed(String),
}

enum SemanticReloadResult {
    Finished(Vec<ImageRecord>),
    Failed(String),
}

enum WpaperdBackgroundResult {
    Refreshed(Option<String>),
    Bound(String),
    Failed(String),
}

enum MoveBackgroundResult {
    Finished(MoveResult),
    Failed(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EmptyState {
    NoSources,
    NeedsScan,
    NoReadyImages,
    NoMatches,
}

fn initial_status(catalog: CatalogSummary) -> String {
    match catalog {
        CatalogSummary { sources: 0, .. } => {
            "Welcome — press a to add your first wallpaper folder".into()
        }
        CatalogSummary { images: 0, .. } => {
            "Catalog is empty — press s to scan registered sources".into()
        }
        CatalogSummary {
            ready_images: 0, ..
        } => "No searchable wallpapers — run `: scan --no-ai` to review failures".into(),
        _ => "Ready — ? for help".into(),
    }
}

struct PreviewRequest {
    image_id: i64,
    path: PathBuf,
}

struct PreviewResult {
    image_id: i64,
    image: std::result::Result<image::DynamicImage, String>,
}

struct PreviewWorker {
    request_sender: Sender<PreviewRequest>,
    result_receiver: Receiver<PreviewResult>,
}

impl PreviewWorker {
    fn new() -> Self {
        let (request_sender, request_receiver) = unbounded::<PreviewRequest>();
        let (result_sender, result_receiver) = unbounded();
        thread::spawn(move || {
            while let Ok(mut request) = request_receiver.recv() {
                for queued in request_receiver.try_iter() {
                    request = queued;
                }
                let image = image::open(&request.path)
                    .map_err(|error| format!("failed to open {}: {error}", request.path.display()));
                if result_sender
                    .send(PreviewResult {
                        image_id: request.image_id,
                        image,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
        Self {
            request_sender,
            result_receiver,
        }
    }
}

struct App {
    images: Vec<ImageRecord>,
    selected: usize,
    filter: FilterSpecV1,
    catalog: CatalogSummary,
    mode: Mode,
    status: String,
    picker: Picker,
    preview: Option<StatefulProtocol>,
    preview_id: Option<i64>,
    preview_requested_id: Option<i64>,
    preview_worker: PreviewWorker,
    scan_receiver: Option<Receiver<BackgroundResult>>,
    scan_started: Option<Instant>,
    scan_total: Option<usize>,
    semantic_receiver: Option<Receiver<SemanticReloadResult>>,
    semantic_selected_id: Option<i64>,
    semantic_previous_filter: Option<FilterSpecV1>,
    semantic_started: Option<Instant>,
    wpaperd_receiver: Option<Receiver<WpaperdBackgroundResult>>,
    wpaperd_refresh_queued: bool,
    move_receiver: Option<Receiver<MoveBackgroundResult>>,
    move_started: Option<Instant>,
    command_receiver: Option<Receiver<CommandBackgroundResult>>,
    command_started: Option<Instant>,
    running_command: Option<String>,
    command_history: Vec<String>,
    pending_command_output: Option<CommandOutput>,
    should_quit: bool,
}

impl App {
    fn new(database: &Database, paths: &AppPaths) -> Result<Self> {
        let filter = FilterSpecV1::default();
        let images = search_resolved(database, paths, &filter)?
            .into_iter()
            .map(|result| result.image)
            .collect();
        let catalog = database.catalog_summary()?;
        let picker = if std::env::var_os("KITTY_WINDOW_ID").is_some()
            || std::env::var("TERM").is_ok_and(|term| term.contains("kitty"))
        {
            Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks())
        } else {
            Picker::halfblocks()
        };
        let mut app = Self {
            images,
            selected: 0,
            filter,
            catalog,
            mode: Mode::Browse,
            status: initial_status(catalog),
            picker,
            preview: None,
            preview_id: None,
            preview_requested_id: None,
            preview_worker: PreviewWorker::new(),
            scan_receiver: None,
            scan_started: None,
            scan_total: None,
            semantic_receiver: None,
            semantic_selected_id: None,
            semantic_previous_filter: None,
            semantic_started: None,
            wpaperd_receiver: None,
            wpaperd_refresh_queued: false,
            move_receiver: None,
            move_started: None,
            command_receiver: None,
            command_started: None,
            running_command: None,
            command_history: Vec::new(),
            pending_command_output: None,
            should_quit: false,
        };
        app.load_preview();
        Ok(app)
    }

    fn selected(&self) -> Option<&ImageRecord> {
        self.images.get(self.selected)
    }

    fn empty_state(&self) -> Option<EmptyState> {
        if !self.images.is_empty() {
            return None;
        }
        Some(if self.catalog.sources == 0 {
            EmptyState::NoSources
        } else if self.catalog.images == 0 {
            EmptyState::NeedsScan
        } else if self.catalog.ready_images == 0 {
            EmptyState::NoReadyImages
        } else {
            EmptyState::NoMatches
        })
    }

    fn ensure_catalog_mutation_idle(&self) -> Result<()> {
        if self.semantic_receiver.is_some() {
            anyhow::bail!("wait for the semantic filter refresh before changing the catalog");
        }
        if self.scan_receiver.is_some() {
            anyhow::bail!("wait for the background scan before changing the catalog");
        }
        if self.command_receiver.is_some() {
            anyhow::bail!("wait for the background command before changing the catalog");
        }
        if self.move_receiver.is_some() {
            anyhow::bail!("wait for the background move before changing the catalog");
        }
        if self.wpaperd_receiver.is_some() || self.wpaperd_refresh_queued {
            anyhow::bail!("wait for the wpaperd worker before changing the catalog");
        }
        Ok(())
    }

    fn reload(&mut self, database: &Database, paths: &AppPaths) -> Result<()> {
        let selected_id = self.selected().map(|image| image.id);
        self.catalog = database.catalog_summary()?;
        if self.semantic_receiver.is_some() {
            anyhow::bail!("wait for the semantic filter refresh to finish");
        }
        if self.filter.semantic_text.is_some() {
            if self.scan_receiver.is_some()
                || self.command_receiver.is_some()
                || self.wpaperd_receiver.is_some()
                || self.move_receiver.is_some()
            {
                anyhow::bail!(
                    "wait for the background scan or command before applying a semantic filter"
                );
            }
            if !model::status(paths, false).verified {
                anyhow::bail!(
                    "semantic TUI filters need the model installed first; run `bgm model install --yes`"
                );
            }
            let database_path = database.path().to_owned();
            let paths = paths.clone();
            let filter = self.filter.clone();
            let (sender, receiver) = unbounded();
            thread::spawn(move || {
                let result = model::without_interactive_install(|| -> Result<Vec<ImageRecord>> {
                    let database = Database::open(database_path)?;
                    Ok(search_resolved(&database, &paths, &filter)?
                        .into_iter()
                        .map(|result| result.image)
                        .collect())
                })
                .map(SemanticReloadResult::Finished)
                .unwrap_or_else(|error| SemanticReloadResult::Failed(format!("{error:#}")));
                let _ = sender.send(result);
            });
            self.semantic_receiver = Some(receiver);
            self.semantic_selected_id = selected_id;
            self.semantic_started = Some(Instant::now());
            self.status = "Resolving semantic filter in background…".into();
            return Ok(());
        }
        let images = search_resolved(database, paths, &self.filter)?
            .into_iter()
            .map(|result| result.image)
            .collect();
        self.replace_images(images, selected_id);
        Ok(())
    }

    fn replace_images(&mut self, images: Vec<ImageRecord>, selected_id: Option<i64>) {
        self.images = images;
        self.selected = selected_id
            .and_then(|id| self.images.iter().position(|image| image.id == id))
            .unwrap_or(0)
            .min(self.images.len().saturating_sub(1));
        self.preview_id = None;
        self.preview_requested_id = None;
        self.load_preview();
    }

    fn poll_semantic_reload(&mut self) {
        let Some(receiver) = self.semantic_receiver.clone() else {
            return;
        };
        match receiver.try_recv() {
            Ok(SemanticReloadResult::Finished(images)) => {
                let selected_id = self.semantic_selected_id.take();
                let elapsed = self
                    .semantic_started
                    .take()
                    .map_or(0.0, |started| started.elapsed().as_secs_f32());
                self.semantic_receiver = None;
                self.semantic_previous_filter = None;
                self.replace_images(images, selected_id);
                self.status = format!(
                    "Semantic filter refreshed in {elapsed:.1}s — {} result(s)",
                    self.images.len()
                );
            }
            Ok(SemanticReloadResult::Failed(error)) => {
                if let Some(previous_filter) = self.semantic_previous_filter.take() {
                    self.filter = previous_filter;
                }
                self.semantic_receiver = None;
                self.semantic_selected_id = None;
                self.semantic_started = None;
                self.status = format!("Semantic filter failed: {error}");
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                if let Some(previous_filter) = self.semantic_previous_filter.take() {
                    self.filter = previous_filter;
                }
                self.semantic_receiver = None;
                self.semantic_selected_id = None;
                self.semantic_started = None;
                self.status = "Semantic filter worker stopped unexpectedly".into();
            }
        }
    }

    fn start_wpaperd_refresh(&mut self, database: &Database, paths: &AppPaths) {
        if self.wpaperd_receiver.is_some()
            || self.semantic_receiver.is_some()
            || self.scan_receiver.is_some()
            || self.command_receiver.is_some()
            || self.move_receiver.is_some()
        {
            self.wpaperd_refresh_queued = true;
            return;
        }
        let database_path = database.path().to_owned();
        let paths = paths.clone();
        let (sender, receiver) = unbounded();
        thread::spawn(move || {
            let result = match Database::open(database_path) {
                Ok(database) => {
                    WpaperdBackgroundResult::Refreshed(refresh_wpaperd_warning(&database, &paths))
                }
                Err(error) => WpaperdBackgroundResult::Failed(format!(
                    "wpaperd refresh could not open the catalog: {error:#}"
                )),
            };
            let _ = sender.send(result);
        });
        self.wpaperd_receiver = Some(receiver);
    }

    fn start_wpaperd_bind(
        &mut self,
        database: &Database,
        paths: &AppPaths,
        display: &str,
        collection: &str,
    ) -> Result<()> {
        if self.wpaperd_receiver.is_some() || self.wpaperd_refresh_queued {
            anyhow::bail!("wait for the wpaperd refresh to finish before binding a display");
        }
        if self.semantic_receiver.is_some()
            || self.scan_receiver.is_some()
            || self.command_receiver.is_some()
            || self.move_receiver.is_some()
        {
            anyhow::bail!("wait for other background work before binding a display");
        }
        let database_path = database.path().to_owned();
        let paths = paths.clone();
        let display = display.to_owned();
        let collection = collection.to_owned();
        let pending_status = format!("Binding {display} to {collection} in background…");
        let (sender, receiver) = unbounded();
        thread::spawn(move || {
            let result = model::without_interactive_install(|| -> Result<String> {
                let database = Database::open(database_path)?;
                let binding = wpaperd::bind(&database, &paths, &display, &collection)?;
                Ok(format!(
                    "Bound {} to {}",
                    binding.display, binding.collection_name
                ))
            })
            .map(WpaperdBackgroundResult::Bound)
            .unwrap_or_else(|error| WpaperdBackgroundResult::Failed(format!("{error:#}")));
            let _ = sender.send(result);
        });
        self.wpaperd_receiver = Some(receiver);
        self.status = pending_status;
        Ok(())
    }

    fn poll_wpaperd(&mut self, database: &Database, paths: &AppPaths) {
        let result =
            self.wpaperd_receiver
                .as_ref()
                .and_then(|receiver| match receiver.try_recv() {
                    Ok(result) => Some(result),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => Some(WpaperdBackgroundResult::Failed(
                        "wpaperd worker stopped unexpectedly".into(),
                    )),
                });
        if let Some(result) = result {
            self.wpaperd_receiver = None;
            match result {
                WpaperdBackgroundResult::Refreshed(warning) => {
                    append_status_warning(&mut self.status, warning);
                }
                WpaperdBackgroundResult::Bound(status) => self.status = status,
                WpaperdBackgroundResult::Failed(error) => {
                    append_status_warning(&mut self.status, Some(error));
                }
            }
        }
        if self.wpaperd_receiver.is_none()
            && self.wpaperd_refresh_queued
            && self.semantic_receiver.is_none()
            && self.scan_receiver.is_none()
            && self.command_receiver.is_none()
            && self.move_receiver.is_none()
        {
            self.wpaperd_refresh_queued = false;
            self.start_wpaperd_refresh(database, paths);
        }
    }

    fn load_preview(&mut self) {
        let Some(image) = self.selected() else {
            self.preview = None;
            self.preview_id = None;
            self.preview_requested_id = None;
            return;
        };
        if self.preview_id == Some(image.id) || self.preview_requested_id == Some(image.id) {
            return;
        }
        let id = image.id;
        let path = image.thumbnail_path.as_ref().unwrap_or(&image.path).clone();
        self.preview = None;
        self.preview_id = None;
        self.preview_requested_id = Some(id);
        if self
            .preview_worker
            .request_sender
            .send(PreviewRequest { image_id: id, path })
            .is_err()
        {
            self.preview_requested_id = None;
            self.preview_id = Some(id);
            self.status = "Preview worker stopped unexpectedly".into();
        }
    }

    fn poll_preview(&mut self) {
        let receiver = self.preview_worker.result_receiver.clone();
        loop {
            match receiver.try_recv() {
                Ok(result) if self.selected().map(|image| image.id) != Some(result.image_id) => {}
                Ok(PreviewResult { image_id, image }) => {
                    self.preview_requested_id = None;
                    self.preview_id = Some(image_id);
                    match image {
                        Ok(image) => {
                            self.preview = Some(self.picker.new_resize_protocol(image));
                        }
                        Err(error) => {
                            self.preview = None;
                            self.status = format!("Preview unavailable: {error}");
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.preview_requested_id.take().is_some() {
                        self.status = "Preview worker stopped unexpectedly".into();
                    }
                    break;
                }
            }
        }
    }

    fn move_selection(&mut self, amount: isize) {
        if self.images.is_empty() {
            return;
        }
        self.selected = self
            .selected
            .saturating_add_signed(amount)
            .min(self.images.len() - 1);
        self.load_preview();
    }

    fn start_scan(&mut self, database: &Database, paths: &AppPaths, config: &Config) {
        if self.scan_receiver.is_some() {
            self.status = "A scan is already running".into();
            return;
        }
        if self.semantic_receiver.is_some() {
            self.status = "Wait for the semantic filter refresh before scanning".into();
            return;
        }
        if self.wpaperd_receiver.is_some() {
            self.status = "Wait for the wpaperd worker before scanning".into();
            return;
        }
        if self.command_receiver.is_some() {
            self.status = "Wait for the background command before scanning".into();
            return;
        }
        if self.move_receiver.is_some() {
            self.status = "Wait for the background move before scanning".into();
            return;
        }
        if self.catalog.sources == 0 {
            self.status = "No sources registered — press a to add a wallpaper folder".into();
            return;
        }
        let database_path = database.path().to_owned();
        let paths = paths.clone();
        let config = config.clone();
        let (sender, receiver) = unbounded();
        thread::spawn(move || {
            let result = (|| -> Result<BackgroundResult> {
                let database = Database::open(database_path)?;
                let progress_sender = sender.clone();
                let scan = scan_catalog_with_progress(
                    &database,
                    &paths,
                    &config,
                    ScanOptions {
                        full: false,
                        no_ai: !config.ai.enabled,
                    },
                    move |event| {
                        if !matches!(event, ScanEvent::Finished(_)) {
                            let _ = progress_sender.send(BackgroundResult::Progress(event));
                        }
                    },
                )?;
                let ai = if config.ai.enabled && model::status(&paths, false).verified {
                    Some(model::without_interactive_install(|| {
                        model::analyze_missing(&database, &paths)
                    })?)
                } else {
                    None
                };
                let wpaperd_warning = refresh_wpaperd_warning(&database, &paths);
                Ok(BackgroundResult::Finished {
                    scan,
                    ai,
                    wpaperd_warning,
                })
            })()
            .unwrap_or_else(|error| BackgroundResult::Failed(format!("{error:#}")));
            let _ = sender.send(result);
        });
        self.scan_receiver = Some(receiver);
        self.scan_started = Some(Instant::now());
        self.scan_total = None;
        self.status = "Scanning in background…".into();
    }

    fn poll_scan(&mut self, database: &Database, paths: &AppPaths) -> Result<()> {
        let Some(receiver) = self.scan_receiver.clone() else {
            return Ok(());
        };
        loop {
            match receiver.try_recv() {
                Ok(BackgroundResult::Progress(ScanEvent::Started { files })) => {
                    self.scan_total = Some(files);
                    self.status = format!("Scanning 0/{files}…");
                }
                Ok(BackgroundResult::Progress(ScanEvent::Processing { index, path })) => {
                    let total = self
                        .scan_total
                        .map_or_else(|| "?".into(), |value| value.to_string());
                    self.status = format!("Scanning {index}/{total}: {}", path.display());
                }
                Ok(BackgroundResult::Progress(ScanEvent::Failed(failure))) => {
                    self.status = format!(
                        "Scan warning for {}: {}",
                        failure.path.display(),
                        failure.error
                    );
                }
                Ok(BackgroundResult::Progress(ScanEvent::Finished(_))) => {}
                Ok(BackgroundResult::Finished {
                    scan,
                    ai,
                    wpaperd_warning,
                }) => {
                    let elapsed = self
                        .scan_started
                        .map_or(0.0, |start| start.elapsed().as_secs_f32());
                    self.status = format!(
                        "Scan finished in {elapsed:.1}s: {} analyzed, {} unchanged, {} failure(s){}",
                        scan.analyzed,
                        scan.unchanged,
                        scan.failed,
                        ai.as_ref().map_or_else(String::new, |report| {
                            format!(", {} AI embeddings", report.embedded)
                        })
                    );
                    if let Some(failure) = scan.failures.first() {
                        self.status.push_str(&format!(
                            " — {}: {}",
                            failure.path.display(),
                            failure.error
                        ));
                    }
                    append_status_warning(&mut self.status, wpaperd_warning);
                    self.scan_receiver = None;
                    self.scan_started = None;
                    self.scan_total = None;
                    self.reload(database, paths)?;
                    break;
                }
                Ok(BackgroundResult::Failed(error)) => {
                    self.status = format!("Background scan failed: {error}");
                    self.scan_receiver = None;
                    self.scan_started = None;
                    self.scan_total = None;
                    break;
                }
                Err(TryRecvError::Empty) => {
                    if self.scan_total.is_none()
                        && let Some(started) = self.scan_started
                    {
                        self.status = format!(
                            "Scanning in background… {:.1}s",
                            started.elapsed().as_secs_f32()
                        );
                    }
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    self.status = "Background scan stopped unexpectedly".into();
                    self.scan_receiver = None;
                    self.scan_started = None;
                    self.scan_total = None;
                    break;
                }
            }
        }
        Ok(())
    }

    fn start_command(&mut self, display: String, arguments: Vec<String>) -> Result<()> {
        if self.scan_receiver.is_some() {
            anyhow::bail!("wait for the background scan to finish before running a command");
        }
        if self.semantic_receiver.is_some() {
            anyhow::bail!("wait for the semantic filter refresh before running a command");
        }
        if self.wpaperd_receiver.is_some() {
            anyhow::bail!("wait for the wpaperd worker before running a command");
        }
        if self.wpaperd_refresh_queued {
            anyhow::bail!("wait for the queued wpaperd refresh before running a command");
        }
        if self.command_receiver.is_some() {
            anyhow::bail!("another command is already running");
        }
        if self.move_receiver.is_some() {
            anyhow::bail!("wait for the background move before running a command");
        }
        let executable = std::env::current_exe().context("failed to locate the bgm executable")?;
        let (sender, receiver) = unbounded();
        thread::spawn(move || {
            let result = Command::new(executable)
                .args(arguments)
                .env("NO_COLOR", "1")
                .stdin(Stdio::null())
                .output()
                .map(|output| CommandBackgroundResult::Finished {
                    success: output.status.success(),
                    stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                })
                .unwrap_or_else(|error| CommandBackgroundResult::Failed(format!("{error:#}")));
            let _ = sender.send(result);
        });
        if self.command_history.last() != Some(&display) {
            self.command_history.push(display.clone());
            if self.command_history.len() > 100 {
                self.command_history.remove(0);
            }
        }
        self.command_receiver = Some(receiver);
        self.command_started = Some(Instant::now());
        self.running_command = Some(display.clone());
        if let Mode::CommandPalette(palette) = &mut self.mode {
            palette.running = true;
            palette.error = None;
        }
        self.status = format!("Running `{display}`…");
        Ok(())
    }

    fn poll_command(&mut self, database: &Database, paths: &AppPaths, config: &mut Config) {
        let Some(receiver) = self.command_receiver.clone() else {
            if matches!(self.mode, Mode::Browse)
                && let Some(output) = self.pending_command_output.take()
            {
                self.mode = Mode::CommandOutput(output);
            }
            return;
        };
        match receiver.try_recv() {
            Ok(CommandBackgroundResult::Finished {
                success,
                stdout,
                mut stderr,
            }) => {
                let command = self.running_command.take().unwrap_or_default();
                self.command_receiver = None;
                let elapsed = self
                    .command_started
                    .take()
                    .map_or(0.0, |started| started.elapsed().as_secs_f32());
                if let Err(error) = self.reload(database, paths) {
                    append_command_note(
                        &mut stderr,
                        &format!("TUI refresh failed after the command: {error:#}"),
                    );
                }
                match Config::load(&paths.config_file) {
                    Ok(updated) => *config = updated,
                    Err(error) => append_command_note(
                        &mut stderr,
                        &format!("Configuration reload failed: {error:#}"),
                    ),
                }
                self.status = format!(
                    "Command {} in {elapsed:.1}s: {command}",
                    if success { "finished" } else { "failed" }
                );
                self.present_command_output(CommandOutput {
                    command,
                    success,
                    stdout,
                    stderr,
                    scroll: 0,
                    viewport_height: 1,
                });
            }
            Ok(CommandBackgroundResult::Failed(error)) => {
                let command = self.running_command.take().unwrap_or_default();
                self.command_receiver = None;
                self.command_started = None;
                self.status = format!("Command failed to start: {command}");
                self.present_command_output(CommandOutput {
                    command,
                    success: false,
                    stdout: String::new(),
                    stderr: error,
                    scroll: 0,
                    viewport_height: 1,
                });
            }
            Err(TryRecvError::Empty) => {
                if let (Some(command), Some(started)) =
                    (self.running_command.as_deref(), self.command_started)
                {
                    self.status = format!(
                        "Running `{command}`… {:.1}s",
                        started.elapsed().as_secs_f32()
                    );
                }
            }
            Err(TryRecvError::Disconnected) => {
                let command = self.running_command.take().unwrap_or_default();
                self.command_receiver = None;
                self.command_started = None;
                self.status = format!("Command worker stopped unexpectedly: {command}");
                self.present_command_output(CommandOutput {
                    command,
                    success: false,
                    stdout: String::new(),
                    stderr: "the background command worker disconnected without a result".into(),
                    scroll: 0,
                    viewport_height: 1,
                });
            }
        }
    }

    fn present_command_output(&mut self, output: CommandOutput) {
        if matches!(self.mode, Mode::Browse | Mode::CommandPalette(_)) {
            self.mode = Mode::CommandOutput(output);
        } else {
            self.pending_command_output = Some(output);
            self.status
                .push_str(" — output will open after the current dialog");
        }
    }

    fn start_move(&mut self, database: &Database, paths: &AppPaths, plan: MovePlan) -> Result<()> {
        self.ensure_catalog_mutation_idle()?;
        let operation_id = plan.id;
        let database_path = database.path().to_owned();
        let paths = paths.clone();
        let (sender, receiver) = unbounded();
        thread::spawn(move || {
            let result = Database::open(database_path)
                .and_then(|database| apply_move(&database, &paths, plan))
                .map(MoveBackgroundResult::Finished)
                .unwrap_or_else(|error| MoveBackgroundResult::Failed(format!("{error:#}")));
            let _ = sender.send(result);
        });
        self.move_receiver = Some(receiver);
        self.move_started = Some(Instant::now());
        self.status = format!("Moving files in background — operation {operation_id}…");
        Ok(())
    }

    fn poll_move(&mut self, database: &Database, paths: &AppPaths) {
        let Some(receiver) = self.move_receiver.clone() else {
            return;
        };
        match receiver.try_recv() {
            Ok(MoveBackgroundResult::Finished(result)) => {
                let elapsed = self
                    .move_started
                    .take()
                    .map_or(0.0, |started| started.elapsed().as_secs_f32());
                self.move_receiver = None;
                if let Err(error) = self.reload(database, paths) {
                    self.status = format!(
                        "Moved {} file(s) in {elapsed:.1}s; TUI refresh failed: {error:#}; undo ID {}",
                        result.moved, result.id
                    );
                } else {
                    self.status = format!(
                        "Moved {} file(s) in {elapsed:.1}s; undo ID {}",
                        result.moved, result.id
                    );
                }
                self.start_wpaperd_refresh(database, paths);
            }
            Ok(MoveBackgroundResult::Failed(error)) => {
                self.move_receiver = None;
                self.move_started = None;
                self.status = format!("Background move failed: {error}");
                if let Err(reload_error) = self.reload(database, paths) {
                    self.status
                        .push_str(&format!(" — TUI refresh failed: {reload_error:#}"));
                }
                self.start_wpaperd_refresh(database, paths);
            }
            Err(TryRecvError::Empty) => {
                if let Some(started) = self.move_started {
                    self.status = format!(
                        "Moving files in background… {:.1}s",
                        started.elapsed().as_secs_f32()
                    );
                }
            }
            Err(TryRecvError::Disconnected) => {
                self.move_receiver = None;
                self.move_started = None;
                self.status =
                    "Background move stopped unexpectedly; inspect `bgm move undo` state".into();
            }
        }
    }
}

fn append_command_note(output: &mut String, note: &str) {
    if !output.is_empty() && !output.ends_with('\n') {
        output.push('\n');
    }
    output.push_str(note);
    output.push('\n');
}

pub fn run(database: &Database, paths: &AppPaths, config: &Config) -> Result<()> {
    let mut app = App::new(database, paths)?;
    let mut config = config.clone();
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    if let Err(error) = execute!(stdout, EnterAlternateScreen, EnableBracketedPaste) {
        let _ = disable_raw_mode();
        let _ = execute!(stdout, LeaveAlternateScreen, DisableBracketedPaste, Show);
        return Err(error.into());
    }
    let cleanup = TerminalCleanup;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let result = run_loop(&mut terminal, &mut app, database, paths, &mut config);

    drop(terminal);
    drop(cleanup);
    result
}

struct TerminalCleanup;

impl Drop for TerminalCleanup {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            std::io::stdout(),
            LeaveAlternateScreen,
            DisableBracketedPaste,
            Show
        );
    }
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    database: &Database,
    paths: &AppPaths,
    config: &mut Config,
) -> Result<()> {
    while !app.should_quit {
        if let Err(error) = app.poll_scan(database, paths) {
            app.status = format!("Background result error: {error:#}");
        }
        app.poll_semantic_reload();
        app.poll_move(database, paths);
        app.poll_wpaperd(database, paths);
        app.poll_command(database, paths, config);
        app.poll_preview();
        terminal.draw(|frame| draw(frame, app))?;
        if event::poll(Duration::from_millis(100))? {
            let result = match event::read()? {
                Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
                    handle_key(app, key, database, paths, config)
                }
                Event::Paste(value) => {
                    handle_paste(app, &value, database);
                    Ok(())
                }
                _ => Ok(()),
            };
            if let Err(error) = result {
                app.mode = Mode::Browse;
                app.status = format!("Error: {error:#}");
            }
        }
    }
    Ok(())
}

fn handle_key(
    app: &mut App,
    key: KeyEvent,
    database: &Database,
    paths: &AppPaths,
    config: &Config,
) -> Result<()> {
    if matches!(&app.mode, Mode::FilterEditor(_)) {
        return handle_filter_editor_key(app, key, database, paths);
    }
    if matches!(&app.mode, Mode::Collections(_)) {
        return handle_collections_key(app, key, database, paths);
    }
    if matches!(&app.mode, Mode::CommandPalette(_)) {
        return handle_command_palette_key(app, key, database);
    }
    if matches!(&app.mode, Mode::Input { .. }) {
        return handle_input_key(app, key, database, paths);
    }
    match &mut app.mode {
        Mode::Browse => handle_browse_key(app, key, database, paths, config),
        Mode::FilterEditor(_) => unreachable!("filter editor handled above"),
        Mode::Collections(_) => unreachable!("collections manager handled above"),
        Mode::CommandPalette(_) => unreachable!("command palette handled above"),
        Mode::CommandOutput(output) => {
            match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => app.mode = Mode::Browse,
                KeyCode::Char(':') => open_command_palette(app, database),
                KeyCode::Up | KeyCode::Char('k') => output.move_scroll(-1),
                KeyCode::Down | KeyCode::Char('j') => output.move_scroll(1),
                KeyCode::PageUp => output.move_scroll(
                    -isize::try_from(output.viewport_height.saturating_sub(1).max(1))
                        .unwrap_or(isize::MAX),
                ),
                KeyCode::PageDown => output.move_scroll(
                    isize::try_from(output.viewport_height.saturating_sub(1).max(1))
                        .unwrap_or(isize::MAX),
                ),
                KeyCode::Home | KeyCode::Char('g') => output.scroll = 0,
                KeyCode::End | KeyCode::Char('G') => output.scroll = output.max_scroll(),
                _ => {}
            }
            Ok(())
        }
        Mode::Help => {
            app.mode = Mode::Browse;
            Ok(())
        }
        Mode::Input { .. } => unreachable!("input dialog handled above"),
        Mode::ConfirmMove(plan) => match key.code {
            KeyCode::Char('y' | 'Y') => {
                let plan = plan.clone();
                app.mode = Mode::Browse;
                app.start_move(database, paths, plan)
            }
            KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                app.mode = Mode::Browse;
                app.status = "Move cancelled".into();
                Ok(())
            }
            _ => Ok(()),
        },
    }
}

fn handle_input_key(
    app: &mut App,
    key: KeyEvent,
    database: &Database,
    paths: &AppPaths,
) -> Result<()> {
    match key.code {
        KeyCode::Esc => app.mode = Mode::Browse,
        KeyCode::Backspace => {
            if let Mode::Input { value, error, .. } = &mut app.mode {
                value.pop();
                *error = None;
            }
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Mode::Input { value, error, .. } = &mut app.mode {
                value.push(character);
                *error = None;
            }
        }
        KeyCode::Enter => {
            let (action, value) = match &app.mode {
                Mode::Input { action, value, .. } => (*action, value.trim().to_owned()),
                _ => return Ok(()),
            };
            match submit_input(app, action, value, database, paths) {
                Ok(()) => {
                    if matches!(app.mode, Mode::Input { .. }) {
                        app.mode = Mode::Browse;
                    }
                }
                Err(error) => {
                    if let Mode::Input {
                        error: dialog_error,
                        ..
                    } = &mut app.mode
                    {
                        *dialog_error = Some(format!("{error:#}"));
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn handle_command_palette_key(app: &mut App, key: KeyEvent, database: &Database) -> Result<()> {
    let action = match &mut app.mode {
        Mode::CommandPalette(palette) => palette.handle_key(key, &app.command_history, database),
        _ => return Ok(()),
    };
    match action {
        CommandPaletteAction::Continue => {}
        CommandPaletteAction::Cancel => {
            let running = app.command_receiver.is_some();
            app.mode = Mode::Browse;
            if running {
                app.status =
                    "Command continues in the background; its output will open when finished"
                        .into();
            }
        }
        CommandPaletteAction::Submit(value) => {
            let display = value.trim().to_owned();
            match parse_tui_command_line(&display) {
                Ok(arguments) => {
                    if let Err(error) = app.start_command(display, arguments)
                        && let Mode::CommandPalette(palette) = &mut app.mode
                    {
                        palette.error = Some(format!("{error:#}"));
                    }
                }
                Err(error) => {
                    if let Mode::CommandPalette(palette) = &mut app.mode {
                        palette.error = Some(format!("{error:#}"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn handle_filter_editor_key(
    app: &mut App,
    key: KeyEvent,
    database: &Database,
    paths: &AppPaths,
) -> Result<()> {
    let command = match &mut app.mode {
        Mode::FilterEditor(editor) => editor.handle_key(key),
        _ => return Ok(()),
    };
    match command {
        FilterEditorCommand::Continue => {}
        FilterEditorCommand::Cancel => app.mode = Mode::Browse,
        FilterEditorCommand::Reset => {
            let value = serde_json::to_string_pretty(&FilterSpecV1::default())?;
            if let Mode::FilterEditor(editor) = &mut app.mode {
                editor.replace_document(value);
                editor.focus = FilterEditorFocus::Document;
                editor.notice = Some("Reset to all wallpapers; Ctrl+S applies it.".into());
            }
        }
        FilterEditorCommand::LoadPreset { name, filter } => {
            let value = serde_json::to_string_pretty(&filter)?;
            if let Mode::FilterEditor(editor) = &mut app.mode {
                editor.replace_document(value);
                editor.focus = FilterEditorFocus::Document;
                editor.notice = Some(format!("Loaded {name}; Ctrl+S applies it."));
            }
        }
        FilterEditorCommand::SavePreset(name) => {
            app.ensure_catalog_mutation_idle()?;
            let value = match &app.mode {
                Mode::FilterEditor(editor) => editor.value(),
                _ => return Ok(()),
            };
            let result = (|| -> Result<(SavedCollection, Vec<SavedCollection>)> {
                let filter = parse_filter(&value)?;
                let saved = save_collection(database, &name, &filter)?;
                let presets = list_collections(database)?;
                Ok((saved, presets))
            })();
            match result {
                Ok((saved, presets)) => {
                    if let Mode::FilterEditor(editor) = &mut app.mode {
                        editor.refresh_saved_presets(presets, &saved.name);
                        editor.focus = FilterEditorFocus::Presets;
                        editor.notice = Some(format!("Saved preset {}.", saved.name));
                        editor.error = None;
                    }
                    app.status = format!("Saved filter preset {}", saved.name);
                    app.start_wpaperd_refresh(database, paths);
                }
                Err(error) => {
                    if let Mode::FilterEditor(editor) = &mut app.mode {
                        editor.focus = FilterEditorFocus::Document;
                        editor.error = Some(format!("{error:#}"));
                    }
                }
            }
        }
        FilterEditorCommand::Apply => {
            let value = match &app.mode {
                Mode::FilterEditor(editor) => editor.value(),
                _ => return Ok(()),
            };
            match apply_filter(app, &value, database, paths) {
                Ok(()) => app.mode = Mode::Browse,
                Err(error) => {
                    if let Mode::FilterEditor(editor) = &mut app.mode {
                        editor.error = Some(format!("{error:#}"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn handle_collections_key(
    app: &mut App,
    key: KeyEvent,
    database: &Database,
    paths: &AppPaths,
) -> Result<()> {
    let command = match &mut app.mode {
        Mode::Collections(manager) => manager.handle_key(key),
        _ => return Ok(()),
    };
    match command {
        CollectionsManagerCommand::Continue => {}
        CollectionsManagerCommand::Close => app.mode = Mode::Browse,
        CollectionsManagerCommand::Load(collection) => {
            if app.filter == collection.filter {
                app.mode = Mode::Browse;
                app.status = format!("Collection {} is already current", collection.name);
                return Ok(());
            }
            let result = (|| -> Result<()> {
                app.ensure_catalog_mutation_idle()?;
                let previous_filter = std::mem::replace(&mut app.filter, collection.filter.clone());
                if let Err(error) = app.reload(database, paths) {
                    app.filter = previous_filter;
                    return Err(error);
                }
                if app.semantic_receiver.is_some() {
                    app.semantic_previous_filter = Some(previous_filter);
                } else {
                    app.status = format!("Loaded collection {}", collection.name);
                }
                Ok(())
            })();
            match result {
                Ok(()) => app.mode = Mode::Browse,
                Err(error) => set_collections_error(app, &error),
            }
        }
        CollectionsManagerCommand::Save(name) => {
            let name = name.trim().to_owned();
            if name.is_empty() {
                set_collections_error_message(app, "Collection name cannot be empty.");
                return Ok(());
            }
            if name.chars().any(char::is_control) {
                set_collections_error_message(
                    app,
                    "Collection name cannot contain control characters.",
                );
                return Ok(());
            }
            let awaiting_overwrite = matches!(
                &app.mode,
                Mode::Collections(manager)
                    if matches!(
                        manager.prompt,
                        Some(CollectionPrompt::Confirm(ref confirmation))
                            if matches!(
                                confirmation.as_ref(),
                                CollectionConfirmation::Overwrite { .. }
                            )
                    )
            );
            if !awaiting_overwrite {
                let existing = match crate::collection::get_collection(database, &name) {
                    Ok(existing) => existing,
                    Err(error) => {
                        set_collections_error(app, &error);
                        return Ok(());
                    }
                };
                if let Some(existing) = existing {
                    if let Mode::Collections(manager) = &mut app.mode {
                        manager.prompt = Some(CollectionPrompt::Confirm(Box::new(
                            CollectionConfirmation::Overwrite {
                                name,
                                existing_name: existing.name,
                            },
                        )));
                        manager.error = None;
                        manager.notice = None;
                    }
                    return Ok(());
                }
            }
            let result =
                (|| -> Result<(SavedCollection, Vec<SavedCollection>, Vec<wpaperd::Binding>)> {
                    app.ensure_catalog_mutation_idle()?;
                    let saved = save_collection(database, &name, &app.filter)?;
                    let collections = list_collections(database)?;
                    let bindings = wpaperd::list_bindings(database)?;
                    Ok((saved, collections, bindings))
                })();
            match result {
                Ok((saved, collections, bindings)) => {
                    if let Mode::Collections(manager) = &mut app.mode {
                        manager.replace_store(collections, bindings, Some(saved.id), 0);
                        manager.prompt = None;
                        manager.error = None;
                        manager.notice = Some(format!("Saved collection {}.", saved.name));
                    }
                    app.status = format!("Saved collection {}", saved.name);
                    app.start_wpaperd_refresh(database, paths);
                }
                Err(error) => set_collections_error(app, &error),
            }
        }
        CollectionsManagerCommand::Update(collection) => {
            let result =
                (|| -> Result<(SavedCollection, Vec<SavedCollection>, Vec<wpaperd::Binding>)> {
                    app.ensure_catalog_mutation_idle()?;
                    let saved = save_collection(database, &collection.name, &app.filter)?;
                    let collections = list_collections(database)?;
                    let bindings = wpaperd::list_bindings(database)?;
                    Ok((saved, collections, bindings))
                })();
            match result {
                Ok((saved, collections, bindings)) => {
                    if let Mode::Collections(manager) = &mut app.mode {
                        manager.replace_store(collections, bindings, Some(saved.id), 0);
                        manager.prompt = None;
                        manager.error = None;
                        manager.notice = Some(format!(
                            "Updated {} from the current browser filter.",
                            saved.name
                        ));
                    }
                    app.status = format!("Updated collection {}", saved.name);
                    app.start_wpaperd_refresh(database, paths);
                }
                Err(error) => set_collections_error(app, &error),
            }
        }
        CollectionsManagerCommand::Delete(collection) => {
            let fallback_selection = match &app.mode {
                Mode::Collections(manager) => manager.selected,
                _ => 0,
            };
            let result = (|| -> Result<(Vec<SavedCollection>, Vec<wpaperd::Binding>)> {
                app.ensure_catalog_mutation_idle()?;
                anyhow::ensure!(
                    delete_collection(database, &collection.name)?,
                    "collection not found: {}",
                    collection.name
                );
                Ok((
                    list_collections(database)?,
                    wpaperd::list_bindings(database)?,
                ))
            })();
            match result {
                Ok((collections, bindings)) => {
                    if let Mode::Collections(manager) = &mut app.mode {
                        manager.replace_store(collections, bindings, None, fallback_selection);
                        manager.prompt = None;
                        manager.error = None;
                        manager.notice = Some(format!("Deleted collection {}.", collection.name));
                    }
                    app.status = format!("Deleted collection {}", collection.name);
                    app.start_wpaperd_refresh(database, paths);
                }
                Err(error) => set_collections_error(app, &error),
            }
        }
    }
    Ok(())
}

fn set_collections_error(app: &mut App, error: &anyhow::Error) {
    set_collections_error_message(app, &format!("{error:#}"));
}

fn set_collections_error_message(app: &mut App, message: &str) {
    if let Mode::Collections(manager) = &mut app.mode {
        manager.error = Some(message.to_owned());
        manager.notice = None;
    }
}

fn handle_paste(app: &mut App, value: &str, database: &Database) {
    match &mut app.mode {
        Mode::FilterEditor(editor) => editor.paste(value),
        Mode::Collections(manager) => manager.paste(value),
        Mode::CommandPalette(palette) => palette.paste(value, database),
        Mode::Input {
            value: input_value,
            error,
            ..
        } => {
            input_value.extend(
                value.chars().filter(|character| {
                    !matches!(character, '\r' | '\n') && !character.is_control()
                }),
            );
            *error = None;
        }
        _ => {}
    }
}

fn handle_browse_key(
    app: &mut App,
    key: KeyEvent,
    database: &Database,
    paths: &AppPaths,
    config: &Config,
) -> Result<()> {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => {
            if app.move_receiver.is_some() {
                app.status = "Wait for the background move to finish before quitting".into();
            } else {
                app.should_quit = true;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
        KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
        KeyCode::PageDown => app.move_selection(10),
        KeyCode::PageUp => app.move_selection(-10),
        KeyCode::Home | KeyCode::Char('g') => {
            app.selected = 0;
            app.load_preview();
        }
        KeyCode::End | KeyCode::Char('G') => {
            app.selected = app.images.len().saturating_sub(1);
            app.load_preview();
        }
        KeyCode::Char('?') => app.mode = Mode::Help,
        KeyCode::Char('a') => {
            app.mode = Mode::Input {
                action: InputAction::Source,
                value: String::new(),
                error: None,
            };
        }
        KeyCode::Char(':') => {
            if app.command_receiver.is_some() {
                app.status = "A command is already running in the background".into();
            } else {
                open_command_palette(app, database);
            }
        }
        KeyCode::Char('/') => {
            app.mode = Mode::FilterEditor(FilterEditor::new(
                serde_json::to_string_pretty(&app.filter)?,
                list_collections(database)?,
            ));
        }
        KeyCode::Char('r') => reset_filter(app, database, paths)?,
        KeyCode::Char('t') => {
            if app.selected().is_some() {
                app.mode = Mode::Input {
                    action: InputAction::Tag,
                    value: String::new(),
                    error: None,
                };
            } else {
                app.status = selection_required_status(app, "tag");
            }
        }
        KeyCode::Char('c') => {
            app.mode =
                Mode::Collections(Box::new(CollectionsManager::load(database, &app.filter)?));
        }
        KeyCode::Char('m') => {
            if app.selected().is_some() {
                app.mode = Mode::Input {
                    action: InputAction::Move,
                    value: String::new(),
                    error: None,
                };
            } else {
                app.status = selection_required_status(app, "move");
            }
        }
        KeyCode::Char('w') => {
            app.mode = Mode::Input {
                action: InputAction::Bind,
                value: "any ".into(),
                error: None,
            };
        }
        KeyCode::Char('f') => {
            if let Some(image) = app.selected() {
                app.ensure_catalog_mutation_idle()?;
                let (id, favorite) = (image.id, !image.favorite);
                set_favorite(database, &[id], favorite)?;
                app.reload(database, paths)?;
                app.status = if favorite {
                    "Marked favorite".into()
                } else {
                    "Removed favorite".into()
                };
                app.start_wpaperd_refresh(database, paths);
            } else {
                app.status = selection_required_status(app, "favorite");
            }
        }
        KeyCode::Char('o') | KeyCode::Enter => {
            if let Some(image) = app.selected() {
                Command::new("xdg-open")
                    .arg(&image.path)
                    .spawn()
                    .context("failed to start xdg-open")?;
                app.status = format!("Opened {}", image.path.display());
            } else {
                app.status = selection_required_status(app, "open");
            }
        }
        KeyCode::Char('s') => app.start_scan(database, paths, config),
        _ => {}
    }
    Ok(())
}

fn open_command_palette(app: &mut App, database: &Database) {
    let palette = CommandPalette::new(database, app.selected());
    app.mode = Mode::CommandPalette(palette);
}

fn reset_filter(app: &mut App, database: &Database, paths: &AppPaths) -> Result<()> {
    if app.filter == FilterSpecV1::default() {
        app.status = if app.catalog.ready_images == 0 {
            initial_status(app.catalog)
        } else {
            "Already showing all wallpapers".into()
        };
        return Ok(());
    }
    app.ensure_catalog_mutation_idle()?;
    let previous_filter = std::mem::take(&mut app.filter);
    if let Err(error) = app.reload(database, paths) {
        app.filter = previous_filter;
        return Err(error);
    }
    app.status = format!("Filter cleared — showing {} wallpaper(s)", app.images.len());
    Ok(())
}

fn selection_required_status(app: &App, action: &str) -> String {
    match app.empty_state() {
        Some(EmptyState::NoSources) => {
            format!("Nothing to {action} yet — press a to add a wallpaper folder")
        }
        Some(EmptyState::NeedsScan) => {
            format!("Nothing to {action} yet — press s to scan registered sources")
        }
        Some(EmptyState::NoReadyImages) => {
            format!("Nothing to {action} — review failures with `: scan --no-ai`")
        }
        Some(EmptyState::NoMatches) => {
            format!("No matching wallpaper to {action} — press r to clear the filter")
        }
        None => format!("Select a wallpaper to {action}"),
    }
}

fn submit_input(
    app: &mut App,
    action: InputAction,
    value: String,
    database: &Database,
    paths: &AppPaths,
) -> Result<()> {
    match action {
        InputAction::Source => {
            anyhow::ensure!(
                !value.is_empty(),
                "type the directory containing your wallpapers"
            );
            app.ensure_catalog_mutation_idle()?;
            let source = database.add_source(&expand_home_path(&value))?;
            app.catalog = database.catalog_summary()?;
            app.status = format!("Registered {} — press s to scan it", source.path.display());
        }
        InputAction::Tag => {
            anyhow::ensure!(!value.is_empty(), "type a tag, or `remove TAG`");
            let image_id = app
                .selected()
                .map(|image| image.id)
                .context("select a wallpaper before changing tags")?;
            app.ensure_catalog_mutation_idle()?;
            if let Some(tag) = value.strip_prefix("remove ").map(str::trim) {
                anyhow::ensure!(!tag.is_empty(), "type the tag to remove after `remove`");
                let changed = remove_tag(database, &[image_id], tag)?;
                app.reload(database, paths)?;
                app.status = if changed == 0 {
                    format!("Tag {tag} was not set on this wallpaper")
                } else {
                    format!("Removed tag {tag}")
                };
            } else {
                add_tag(database, &[image_id], &value)?;
                app.reload(database, paths)?;
                app.status = format!("Added tag {value}");
            }
            app.start_wpaperd_refresh(database, paths);
        }
        InputAction::Move => {
            anyhow::ensure!(!value.is_empty(), "type a destination directory");
            let image = app
                .selected()
                .context("select a wallpaper before moving it")?;
            let plan = plan_move(std::slice::from_ref(image), &expand_home_path(&value))?;
            app.mode = Mode::ConfirmMove(plan);
        }
        InputAction::Bind => {
            if let Some(display) = value.strip_prefix("unbind ").map(str::trim) {
                app.ensure_catalog_mutation_idle()?;
                let result = wpaperd::unbind(database, paths, display)?;
                app.status = if result.restored {
                    format!("Unbound {display} and restored its previous path")
                } else {
                    format!("Unbound {display}; externally edited path was preserved")
                };
            } else {
                let (display, collection) = value
                    .split_once(char::is_whitespace)
                    .context("binding input must be DISPLAY COLLECTION or `unbind DISPLAY`")?;
                app.start_wpaperd_bind(database, paths, display, collection.trim())?;
            }
        }
    }
    Ok(())
}

fn expand_home_path(value: &str) -> PathBuf {
    let Some(rest) = value.strip_prefix('~') else {
        return PathBuf::from(value);
    };
    if !rest.is_empty() && !rest.starts_with('/') {
        return PathBuf::from(value);
    }
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from(value),
        |home| {
            let mut path = PathBuf::from(home);
            if let Some(rest) = rest.strip_prefix('/')
                && !rest.is_empty()
            {
                path.push(rest);
            }
            path
        },
    )
}

fn refresh_wpaperd_warning(database: &Database, paths: &AppPaths) -> Option<String> {
    match model::without_interactive_install(|| wpaperd::refresh(database, paths, None)) {
        Ok(report) => report
            .failure_summary()
            .map(|failures| format!("wpaperd refresh warning: {failures}")),
        Err(error) => Some(format!("wpaperd refresh failed: {error:#}")),
    }
}

fn append_status_warning(status: &mut String, warning: Option<String>) {
    if let Some(warning) = warning {
        status.push_str(" — ");
        status.push_str(&warning);
    }
}

fn apply_filter(app: &mut App, value: &str, database: &Database, paths: &AppPaths) -> Result<()> {
    app.ensure_catalog_mutation_idle()?;
    let filter = parse_filter(value)?;

    let previous_filter = std::mem::replace(&mut app.filter, filter);
    if let Err(error) = app.reload(database, paths) {
        app.filter = previous_filter;
        return Err(error);
    }
    if app.semantic_receiver.is_some() {
        app.semantic_previous_filter = Some(previous_filter);
    } else {
        app.status = format!("Filter applied — {} result(s)", app.images.len());
    }
    Ok(())
}

fn parse_filter(value: &str) -> Result<FilterSpecV1> {
    let filter = if value.trim().is_empty() {
        FilterSpecV1::default()
    } else {
        serde_json::from_str::<FilterSpecV1>(value)
            .context("filter must be valid FilterSpecV1 JSON")?
    };
    filter.validate()?;
    Ok(filter)
}

fn draw(frame: &mut Frame<'_>, app: &mut App) {
    let areas = render_base(frame, app);
    if let Some(preview) = app.preview.as_mut() {
        frame.render_stateful_widget(
            StatefulImage::new().resize(Resize::Fit(None)),
            areas.preview,
            preview,
        );
    } else {
        let message = match app.empty_state() {
            Some(EmptyState::NoSources) => concat!(
                "Welcome to bgm\n\n",
                "1. Press a to add a wallpaper folder\n",
                "2. Press s to scan it\n\n",
                "Your source images are never modified by a scan."
            )
            .into(),
            Some(EmptyState::NeedsScan) => format!(
                "Ready to build your catalog\n\nPress s to scan {} registered source{}\n\nThe scan runs in the background.",
                app.catalog.sources,
                if app.catalog.sources == 1 { "" } else { "s" }
            ),
            Some(EmptyState::NoReadyImages) => concat!(
                "No searchable wallpapers\n\n",
                "The scan found files, but none are ready.\n",
                "Run : scan --no-ai to review failures, and check import bounds."
            )
            .into(),
            Some(EmptyState::NoMatches) => concat!(
                "No wallpaper matches this filter\n\n",
                "Press / to edit the filter\n",
                "Press r to show all wallpapers"
            )
            .into(),
            None if app.preview_requested_id == app.selected().map(|image| image.id) => {
                "Loading preview…".into()
            }
            None => "Preview unavailable\n\nPress Enter or o to open the original".into(),
        };
        frame.render_widget(
            Paragraph::new(message)
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" Preview ")),
            areas.preview,
        );
    }
    render_modal(frame, app);
}

struct ScreenAreas {
    preview: Rect,
}

fn render_base(frame: &mut Frame<'_>, app: &App) -> ScreenAreas {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(frame.area());
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(40),
            Constraint::Percentage(26),
        ])
        .split(vertical[1]);

    let result_count = if app.filter == FilterSpecV1::default() {
        format!(
            "{} wallpaper{}",
            app.images.len(),
            if app.images.len() == 1 { "" } else { "s" }
        )
    } else {
        format!(
            "{} of {} wallpaper{}",
            app.images.len(),
            app.catalog.ready_images,
            if app.catalog.ready_images == 1 {
                ""
            } else {
                "s"
            }
        )
    };
    let title = format!(" bgm — {result_count} — filter v{} ", app.filter.version);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "Background Manager",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  native catalog • collections • wpaperd"),
        ]))
        .block(Block::default().borders(Borders::ALL).title(title)),
        vertical[0],
    );

    if app.images.is_empty() {
        frame.render_widget(
            Paragraph::new(empty_results_text(app))
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" Results ")),
            body[0],
        );
    } else {
        let items: Vec<_> = app
            .images
            .iter()
            .map(|image| {
                let favorite = if image.favorite { "★" } else { " " };
                let dimensions = match (image.width, image.height) {
                    (Some(width), Some(height)) => format!("{width}×{height}"),
                    _ => "?×?".into(),
                };
                let name = image.path.file_name().map_or_else(
                    || image.path.display().to_string(),
                    |name| name.to_string_lossy().into(),
                );
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{favorite} "), Style::default().fg(Color::Yellow)),
                    Span::raw(format!("{name}  ")),
                    Span::styled(dimensions, Style::default().fg(Color::DarkGray)),
                ]))
            })
            .collect();
        let mut list_state = ListState::default().with_selected(Some(app.selected));
        frame.render_stateful_widget(
            List::new(items)
                .highlight_symbol("▸ ")
                .highlight_style(
                    Style::default()
                        .bg(Color::Rgb(35, 48, 65))
                        .add_modifier(Modifier::BOLD),
                )
                .block(Block::default().borders(Borders::ALL).title(" Results ")),
            body[0],
            &mut list_state,
        );
    }

    let metadata = app
        .selected()
        .map_or_else(|| empty_metadata_text(app), metadata_text);
    frame.render_widget(
        Paragraph::new(metadata).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Metadata • palette • AI estimates "),
        ),
        body[2],
    );

    let footer = if vertical[2].width >= 108 {
        "↑/↓ nav  / filter  r reset  a add  s scan  : cmd  f fave  t tag  c collections  m move  w bind  o open  ? help  q quit"
    } else {
        "↑/↓ nav  / filter  a add  s scan  : cmd  ? help  q quit"
    };
    frame.render_widget(
        Paragraph::new(Span::styled(footer, Style::default().fg(Color::Cyan)))
            .block(Block::default().borders(Borders::TOP)),
        vertical[2],
    );
    let status_area = Rect {
        x: vertical[0].x + 2,
        y: vertical[0].y + 1,
        width: vertical[0].width.saturating_sub(4),
        height: 1,
    };
    frame.render_widget(
        Paragraph::new(app.status.as_str()).alignment(Alignment::Right),
        status_area,
    );
    ScreenAreas { preview: body[1] }
}

fn empty_results_text(app: &App) -> Text<'static> {
    let lines = match app.empty_state() {
        Some(EmptyState::NoSources) => vec![
            Line::from(Span::styled(
                "No sources yet",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Press a to add a folder"),
            Line::from("then s to scan"),
        ],
        Some(EmptyState::NeedsScan) => vec![
            Line::from(Span::styled(
                "Catalog is empty",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Press s to scan"),
        ],
        Some(EmptyState::NoReadyImages) => vec![
            Line::from(Span::styled(
                "No ready images",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Run : scan --no-ai"),
            Line::from("and review scan warnings"),
        ],
        Some(EmptyState::NoMatches) => vec![
            Line::from(Span::styled(
                "No matches",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("/ edit filter"),
            Line::from("r show all"),
        ],
        None => Vec::new(),
    };
    Text::from(lines)
}

fn empty_metadata_text(app: &App) -> Text<'static> {
    let message = match app.empty_state() {
        Some(EmptyState::NoSources) => {
            "Setup\n\nAdd a source folder with a.\nIts images stay untouched."
        }
        Some(EmptyState::NeedsScan) => {
            "Setup\n\nA source is registered.\nPress s to discover and analyze its images."
        }
        Some(EmptyState::NoReadyImages) => {
            "Catalog status\n\nFiles were discovered, but none are ready to browse.\nCheck import bounds and scan errors."
        }
        Some(EmptyState::NoMatches) => {
            "Filter status\n\nThe catalog has images, but the active filter matched none."
        }
        None => "No wallpaper selected",
    };
    Text::from(message)
}

fn metadata_text(image: &ImageRecord) -> Text<'static> {
    let mut lines = vec![
        Line::from(Span::styled(
            image.path.display().to_string(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(format!(
            "{} × {}   ratio {}   {}",
            image
                .width
                .map_or_else(|| "?".into(), |value| value.to_string()),
            image
                .height
                .map_or_else(|| "?".into(), |value| value.to_string()),
            image.common_ratio.as_deref().unwrap_or("custom"),
            image.orientation.as_deref().unwrap_or("unknown")
        )),
        Line::from(format!(
            "dominant {} {}   {}",
            image.dominant_hex.as_deref().unwrap_or("—"),
            image.dominant_name.as_deref().unwrap_or(""),
            image.light_dark.as_deref().unwrap_or("unknown")
        )),
        Line::from(format!(
            "luminance {:.3}  saturation {:.3}  contrast {:.3}",
            image.luminance.unwrap_or_default(),
            image.saturation.unwrap_or_default(),
            image.contrast.unwrap_or_default()
        )),
        Line::from(""),
        Line::from(Span::styled("Palette", Style::default().fg(Color::Cyan))),
    ];
    for colour in &image.palette {
        let terminal_colour = parse_terminal_colour(&colour.hex).unwrap_or(Color::Reset);
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default().bg(terminal_colour)),
            Span::raw(format!(
                " {} {:>5.1}%",
                colour.hex,
                colour.proportion * 100.0
            )),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "AI estimates",
        Style::default().fg(Color::Magenta),
    )));
    if image.ai_estimates.is_empty() {
        lines.push(Line::from("not analyzed"));
    } else {
        for estimate in image
            .ai_estimates
            .iter()
            .filter(|estimate| estimate.score >= 0.05)
            .take(8)
        {
            lines.push(Line::from(format!(
                "{} ≈ {} ({:.0}%)",
                estimate.pack,
                estimate.label,
                estimate.score * 100.0
            )));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(format!(
        "tags: {}",
        if image.tags.is_empty() {
            "—".into()
        } else {
            image.tags.join(", ")
        }
    )));
    Text::from(lines)
}

fn collection_detail_lines(manager: &CollectionsManager) -> Vec<Line<'static>> {
    let Some(collection) = manager.selected() else {
        return vec![
            Line::from(Span::styled(
                "No saved collections",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Press s to save the current browser filter."),
            Line::from("The new collection will also be available to the CLI and wpaperd."),
        ];
    };
    let displays = manager.bound_displays(collection.id);
    let mut lines = vec![
        Line::from(Span::styled(
            collection.name.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(format!(
            "Created: {}",
            format_collection_timestamp(collection.created_at)
        )),
        Line::from(format!(
            "Updated: {}",
            format_collection_timestamp(collection.updated_at)
        )),
        Line::from(format!(
            "Current browser filter: {}",
            if collection.filter == manager.current_filter {
                "exact match"
            } else {
                "different"
            }
        )),
        Line::from(format!(
            "wpaperd displays: {}",
            if displays.is_empty() {
                "none".into()
            } else {
                displays.join(", ")
            }
        )),
        Line::from(""),
    ];
    match manager.view {
        CollectionDetailView::Summary => {
            lines.push(Line::from(Span::styled(
                "Filter facets",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            lines.extend(
                readable_filter_summary(&collection.filter)
                    .into_iter()
                    .map(Line::from),
            );
        }
        CollectionDetailView::Json => {
            lines.push(Line::from(Span::styled(
                "FilterSpecV1 JSON",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            let json = serde_json::to_string_pretty(&collection.filter)
                .unwrap_or_else(|error| format!("Could not serialize filter: {error}"));
            lines.extend(json.lines().map(|line| Line::from(line.to_owned())));
        }
    }
    lines
}

fn readable_filter_summary(filter: &FilterSpecV1) -> Vec<String> {
    let default = FilterSpecV1::default();
    let mut lines = Vec::new();
    if !filter.source_ids.is_empty() {
        lines.push(format!(
            "Source IDs (any): {}",
            join_values(&filter.source_ids)
        ));
    }
    if !filter.paths.is_empty() {
        lines.push(format!("Path contains (any): {}", filter.paths.join(", ")));
    }
    if let Some(value) = filter.min_width {
        lines.push(format!("Minimum width: {value} px"));
    }
    if let Some(value) = filter.max_width {
        lines.push(format!("Maximum width: {value} px"));
    }
    if let Some(value) = filter.min_height {
        lines.push(format!("Minimum height: {value} px"));
    }
    if let Some(value) = filter.max_height {
        lines.push(format!("Maximum height: {value} px"));
    }
    if !filter.orientations.is_empty() {
        lines.push(format!(
            "Orientations (any): {}",
            filter
                .orientations
                .iter()
                .map(|orientation| orientation.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !filter.aspect_ratios.is_empty() {
        lines.push(format!(
            "Aspect ratios (any): {}",
            filter
                .aspect_ratios
                .iter()
                .map(|ratio| concise_decimal(*ratio))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if filter.aspect_tolerance != default.aspect_tolerance {
        lines.push(format!(
            "Aspect tolerance: {}%",
            concise_decimal(filter.aspect_tolerance * 100.0)
        ));
    }
    if !filter.light_dark.is_empty() {
        lines.push(format!(
            "Light/dark (any): {}",
            filter
                .light_dark
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(value) = filter.min_luminance {
        lines.push(format!("Minimum luminance: {}", concise_decimal(value)));
    }
    if let Some(value) = filter.max_luminance {
        lines.push(format!("Maximum luminance: {}", concise_decimal(value)));
    }
    for colour in &filter.dominant_colours {
        lines.push(format!(
            "Dominant colour (any): {} within Oklab distance {}",
            colour.hex,
            concise_decimal(colour.max_distance)
        ));
    }
    for colour in &filter.palette_colours {
        lines.push(format!(
            "Palette colour (any): {} within Oklab distance {}",
            colour.hex,
            concise_decimal(colour.max_distance)
        ));
    }
    for label in &filter.ai_labels {
        lines.push(format!(
            "AI label (any): {} / {} at least {}",
            label.pack,
            label.label,
            concise_decimal(label.min_score)
        ));
    }
    if let Some(text) = &filter.semantic_text {
        lines.push(format!("Semantic text: {text}"));
    }
    if let Some(value) = filter.semantic_min_score {
        lines.push(format!(
            "Minimum semantic score: {}",
            concise_decimal(value)
        ));
    }
    if !filter.tags.is_empty() {
        lines.push(format!("Tags (any): {}", filter.tags.join(", ")));
    }
    if let Some(favorite) = filter.favorite {
        lines.push(format!(
            "Favourite: {}",
            if favorite { "yes" } else { "no" }
        ));
    }
    if lines.is_empty() {
        lines.push("No filter facets — includes every ready wallpaper.".into());
    }
    lines
}

fn join_values<T: ToString>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn concise_decimal(value: impl Into<f64>) -> String {
    let value = value.into();
    let formatted = format!("{value:.3}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn format_collection_timestamp(milliseconds: i64) -> String {
    DateTime::<Utc>::from_timestamp_millis(milliseconds).map_or_else(
        || format!("invalid timestamp ({milliseconds})"),
        |timestamp| timestamp.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
    )
}

fn render_modal(frame: &mut Frame<'_>, app: &mut App) {
    match &mut app.mode {
        Mode::Browse => {}
        Mode::FilterEditor(editor) => render_filter_editor(frame, editor),
        Mode::Collections(manager) => render_collections_manager(frame, manager),
        Mode::CommandPalette(palette) => render_command_palette(frame, palette),
        Mode::CommandOutput(output) => render_command_output(frame, output),
        Mode::Help => {
            let area = centered_rect(96, 96, frame.area());
            frame.render_widget(Clear, area);
            let block = Block::default().borders(Borders::ALL).title(" Help ");
            let inner = block.inner(area);
            frame.render_widget(block, area);
            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(inner);
            frame.render_widget(
                Paragraph::new(concat!(
                    "Browse\n",
                    "↑/↓, j/k    navigate\n",
                    "a / s       add source / scan\n",
                    "/ / r       edit / reset filter\n",
                    "f / t       favorite / tags\n",
                    "c / m / w   collections / move / bind\n",
                    ":           command palette\n",
                    "o / Enter   open; q quits\n\n",
                    "Collections manager\n",
                    "↑/↓         select / scroll\n",
                    "Tab, ←/→    switch pane\n",
                    "Enter       load selected\n",
                    "s / u / d   save / update / delete\n",
                    "v           readable / JSON\n",
                    "Esc / c     cancel / close",
                ))
                .wrap(Wrap { trim: false }),
                columns[0],
            );
            frame.render_widget(
                Paragraph::new(concat!(
                    "Command palette\n",
                    "Tab / ↑/↓    complete / choose\n",
                    "Ctrl+P/N     command history\n",
                    "←/→ Home End edit\n",
                    "Ctrl+W       delete a word\n",
                    "Enter runs; Esc closes\n\n",
                    "Filter editor\n",
                    "Ctrl+Space   JSON IntelliSense\n",
                    "Tab / ↑/↓    accept / choose\n",
                    "Ctrl+P       save named preset\n",
                    "Ctrl+S/R     apply / reset\n",
                    "Esc          cancel\n\n",
                    "Press any key to close help.",
                ))
                .wrap(Wrap { trim: false }),
                columns[1],
            );
        }
        Mode::Input {
            action,
            value,
            error,
        } => {
            let (title, hint) = match action {
                InputAction::Source => (
                    " Add wallpaper source ",
                    "directory to scan; ~/ paths are supported",
                ),
                InputAction::Tag => (" Change tags ", "TAG to add, or: remove TAG"),
                InputAction::Move => (" Move preview ", "destination directory"),
                InputAction::Bind => (
                    " Manage wpaperd binding ",
                    "DISPLAY COLLECTION, or: unbind DISPLAY",
                ),
            };
            let area = centered_fixed_height(70, 9, frame.area());
            frame.render_widget(Clear, area);
            let feedback = error.as_ref().map_or_else(
                || Line::from(""),
                |error| {
                    Line::from(Span::styled(
                        format!("Error: {error}"),
                        Style::default().fg(Color::Red),
                    ))
                },
            );
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(hint),
                    Line::from(""),
                    Line::from(Span::styled(
                        format!("> {value}█"),
                        Style::default().fg(Color::Cyan),
                    )),
                    Line::from(""),
                    feedback,
                    Line::from("Enter applies • Esc cancels"),
                ])
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(title)),
                area,
            );
        }
        Mode::ConfirmMove(plan) => {
            let area = centered_rect(78, 46, frame.area());
            frame.render_widget(Clear, area);
            let mut lines = vec![
                Line::from(Span::styled(
                    "Filesystem move preview",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(format!("Destination: {}", plan.destination_root.display())),
                Line::from(""),
            ];
            for item in plan.items.iter().take(8) {
                lines.push(Line::from(format!(
                    "{} → {}",
                    item.original_path.display(),
                    item.destination.display()
                )));
            }
            lines.push(Line::from(""));
            lines.push(Line::from("Apply this move? y = apply • n/Esc = cancel"));
            frame.render_widget(
                Paragraph::new(lines).wrap(Wrap { trim: false }).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Confirm safe move "),
                ),
                area,
            );
        }
    }
}

fn render_collections_manager(frame: &mut Frame<'_>, manager: &mut CollectionsManager) {
    let area = centered_rect(94, 88, frame.area());
    frame.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Collections manager ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(4)])
        .split(inner);
    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(sections[0]);

    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_style(
            Style::default().fg(if manager.focus == CollectionsFocus::List {
                Color::Cyan
            } else {
                Color::DarkGray
            }),
        )
        .title(" Saved collections ");
    if manager.collections.is_empty() {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "No saved collections",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from("Press s to save the"),
                Line::from("current browser filter."),
            ])
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: false })
            .block(list_block),
            panes[0],
        );
    } else {
        let items = manager
            .collections
            .iter()
            .map(|collection| {
                let current = if collection.filter == manager.current_filter {
                    "●"
                } else {
                    " "
                };
                let bound = if manager.bound_displays(collection.id).is_empty() {
                    " "
                } else {
                    "W"
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{current} "), Style::default().fg(Color::Cyan)),
                    Span::styled(format!("{bound} "), Style::default().fg(Color::Magenta)),
                    Span::raw(collection.name.clone()),
                ]))
            })
            .collect::<Vec<_>>();
        let mut state = ListState::default().with_selected(Some(manager.selected));
        frame.render_stateful_widget(
            List::new(items)
                .highlight_symbol("▸ ")
                .highlight_style(
                    Style::default()
                        .bg(Color::Rgb(35, 48, 65))
                        .add_modifier(Modifier::BOLD),
                )
                .block(list_block),
            panes[0],
            &mut state,
        );
    }

    let detail_title = match manager.view {
        CollectionDetailView::Summary => " Details — readable ",
        CollectionDetailView::Json => " Details — JSON ",
    };
    let detail_block = Block::default()
        .borders(Borders::ALL)
        .border_style(
            Style::default().fg(if manager.focus == CollectionsFocus::Details {
                Color::Cyan
            } else {
                Color::DarkGray
            }),
        )
        .title(detail_title);
    let detail_inner = detail_block.inner(panes[1]);
    manager.detail_viewport_height = usize::from(detail_inner.height).max(1);
    manager.detail_scroll = manager.detail_scroll.min(manager.max_detail_scroll());
    let details = collection_detail_lines(manager);
    frame.render_widget(
        Paragraph::new(details)
            .scroll((u16::try_from(manager.detail_scroll).unwrap_or(u16::MAX), 0))
            .wrap(Wrap { trim: false })
            .block(detail_block),
        panes[1],
    );

    let feedback = if let Some(error) = &manager.error {
        Span::styled(format!("Error: {error}"), Style::default().fg(Color::Red))
    } else if let Some(notice) = &manager.notice {
        Span::styled(notice.clone(), Style::default().fg(Color::Green))
    } else {
        Span::styled(
            "● current filter • W active wpaperd binding",
            Style::default().fg(Color::DarkGray),
        )
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from("Enter load • s save new • u update • d delete • v readable/JSON"),
            Line::from("Tab or ←/→ switches pane • ↑/↓, Page Up/Down, Home/End navigate"),
            Line::from("Esc closes or cancels a prompt first • c closes"),
            Line::from(feedback),
        ])
        .wrap(Wrap { trim: false }),
        sections[1],
    );

    if manager.prompt.is_some() {
        render_collection_prompt(frame, manager, area);
    }
}

fn render_collection_prompt(frame: &mut Frame<'_>, manager: &CollectionsManager, parent: Rect) {
    match manager.prompt.as_ref() {
        Some(CollectionPrompt::Name(entry)) => {
            let area = centered_fixed_height(70, 10, parent);
            frame.render_widget(Clear, area);
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Save current filter as a new collection ");
            let inner = block.inner(area);
            frame.render_widget(block, area);
            let sections = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Length(3),
                    Constraint::Min(1),
                ])
                .split(inner);
            frame.render_widget(
                Paragraph::new("Enter a collection name; matching names require confirmation."),
                sections[0],
            );
            let input_block = Block::default().borders(Borders::ALL).title(" Name ");
            let input_inner = input_block.inner(sections[1]);
            frame.render_widget(input_block, sections[1]);
            let content_width = usize::from(input_inner.width.saturating_sub(2)).max(1);
            let mut scroll = 0;
            while display_width(&entry.value, scroll, entry.cursor) >= content_width
                && scroll < entry.cursor
            {
                scroll += 1;
            }
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("> ", Style::default().fg(Color::Cyan)),
                    Span::raw(visible_text(&entry.value, scroll, content_width)),
                ])),
                input_inner,
            );
            let feedback = manager.error.as_ref().map_or_else(
                || {
                    Line::from(
                        "←/→ Home End move • Delete/Backspace edit • Enter saves • Esc cancels",
                    )
                },
                |error| {
                    Line::from(Span::styled(
                        format!("Error: {error}"),
                        Style::default().fg(Color::Red),
                    ))
                },
            );
            frame.render_widget(
                Paragraph::new(feedback).wrap(Wrap { trim: false }),
                sections[2],
            );
            let cursor_x = u16::try_from(display_width(&entry.value, scroll, entry.cursor))
                .unwrap_or(u16::MAX)
                .min(input_inner.width.saturating_sub(3));
            frame.set_cursor_position((input_inner.x + 2 + cursor_x, input_inner.y));
        }
        Some(CollectionPrompt::Confirm(confirmation)) => {
            let (title, message) = match confirmation.as_ref() {
                CollectionConfirmation::Overwrite { existing_name, .. } => (
                    " Confirm overwrite ",
                    format!(
                        "A collection named {existing_name} already exists. Replace its filter with the current browser filter?"
                    ),
                ),
                CollectionConfirmation::Update(collection) => (
                    " Confirm update ",
                    format!(
                        "Replace the filter saved in {} with the current browser filter?",
                        collection.name
                    ),
                ),
                CollectionConfirmation::Delete(collection) => (
                    " Confirm deletion ",
                    format!(
                        "Delete collection {}? This cannot be undone.",
                        collection.name
                    ),
                ),
            };
            let area = centered_fixed_height(70, 9, parent);
            frame.render_widget(Clear, area);
            let feedback = manager.error.as_ref().map_or_else(
                || Line::from("Enter or y confirms • n or Esc cancels"),
                |error| {
                    Line::from(Span::styled(
                        format!("Error: {error}"),
                        Style::default().fg(Color::Red),
                    ))
                },
            );
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(Span::styled(
                        message,
                        Style::default().add_modifier(Modifier::BOLD),
                    )),
                    Line::from(""),
                    feedback,
                ])
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Yellow))
                        .title(title),
                ),
                area,
            );
        }
        None => {}
    }
}

fn render_command_palette(frame: &mut Frame<'_>, palette: &CommandPalette) {
    let area = centered_rect(88, 62, frame.area());
    frame.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Command palette — bgm commands ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(4),
        ])
        .split(inner);

    frame.render_widget(
        Paragraph::new("Type a command without the leading `bgm` • quotes and ~/ paths work")
            .style(Style::default().fg(Color::DarkGray)),
        sections[0],
    );
    let input_block = Block::default().borders(Borders::ALL).title(" Command ");
    let input_inner = input_block.inner(sections[1]);
    frame.render_widget(input_block, sections[1]);
    let content_width = usize::from(input_inner.width.saturating_sub(2));
    let mut scroll = 0;
    while display_width(&palette.value, scroll, palette.cursor) >= content_width.max(1)
        && scroll < palette.cursor
    {
        scroll += 1;
    }
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("> ", Style::default().fg(Color::Cyan)),
            Span::raw(visible_text(&palette.value, scroll, content_width)),
        ])),
        input_inner,
    );

    let items = if palette.suggestions.is_empty() {
        vec![ListItem::new(Span::styled(
            "No completions — continue typing or press Enter to validate",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        palette
            .suggestions
            .iter()
            .map(|suggestion| {
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<24}", suggestion.label),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::raw(suggestion.description.clone()),
                ]))
            })
            .collect()
    };
    let mut state = ListState::default()
        .with_selected((!palette.suggestions.is_empty()).then_some(palette.selected_suggestion));
    frame.render_stateful_widget(
        List::new(items)
            .highlight_symbol("▸ ")
            .highlight_style(
                Style::default()
                    .bg(Color::Rgb(35, 48, 65))
                    .add_modifier(Modifier::BOLD),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" IntelliSense "),
            ),
        sections[2],
        &mut state,
    );

    let feedback = if palette.running {
        Span::styled(
            "Running in the background… Esc hides this box; output opens when ready.",
            Style::default().fg(Color::Yellow),
        )
    } else if let Some(error) = &palette.error {
        Span::styled(format!("Error: {error}"), Style::default().fg(Color::Red))
    } else if let Some(suggestion) = palette.suggestions.get(palette.selected_suggestion) {
        Span::styled(
            suggestion.description.clone(),
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::raw("")
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from("Tab completes • ↑/↓ selects • Ctrl+P/N history • Enter runs • Esc closes"),
            Line::from(
                "Only bgm commands run; shell operators and $VARIABLE expansion are not used.",
            ),
            Line::from(feedback),
        ])
        .wrap(Wrap { trim: false }),
        sections[3],
    );

    if !palette.running && input_inner.width > 2 {
        let cursor_x = u16::try_from(display_width(&palette.value, scroll, palette.cursor))
            .unwrap_or(u16::MAX)
            .min(input_inner.width.saturating_sub(3));
        frame.set_cursor_position((input_inner.x + 2 + cursor_x, input_inner.y));
    }
}

fn render_command_output(frame: &mut Frame<'_>, output: &mut CommandOutput) {
    let area = centered_rect(92, 82, frame.area());
    frame.render_widget(Clear, area);
    let colour = if output.success {
        Color::Green
    } else {
        Color::Red
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(colour))
        .title(if output.success {
            " Command finished "
        } else {
            " Command failed "
        });
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(inner);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled("bgm ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    output.command.clone(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(if output.success {
                "exit: success"
            } else {
                "exit: failure"
            }),
        ]),
        sections[0],
    );

    let mut lines = Vec::new();
    lines.extend(
        output
            .stdout
            .lines()
            .map(|line| Line::from(line.to_owned())),
    );
    if !output.stderr.is_empty() {
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            "stderr / diagnostics",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )));
        lines.extend(output.stderr.lines().map(|line| {
            Line::from(Span::styled(
                line.to_owned(),
                Style::default().fg(Color::LightRed),
            ))
        }));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "(command produced no output)",
            Style::default().fg(Color::DarkGray),
        )));
    }
    output.viewport_height = usize::from(sections[1].height);
    output.scroll = output.scroll.min(output.max_scroll());
    frame.render_widget(
        Paragraph::new(lines)
            .scroll((u16::try_from(output.scroll).unwrap_or(u16::MAX), 0))
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title(" Output ")),
        sections[1],
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "line {} of {}",
                output.scroll.saturating_add(1),
                output.line_count().max(1)
            )),
            Line::from("↑/↓ or Page Up/Down scroll • Enter/Esc closes • : runs another command"),
        ])
        .style(Style::default().fg(Color::DarkGray)),
        sections[2],
    );
}

fn render_filter_editor(frame: &mut Frame<'_>, editor: &mut FilterEditor) {
    let area = centered_rect(96, 90, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Filter editor — examples, saved presets, and FilterSpecV1 JSON ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(4),
        ])
        .split(inner);

    frame.render_widget(
        Paragraph::new(
            "Load an example or saved collection, then edit JSON • facets AND; arrays OR",
        )
        .style(Style::default().fg(Color::DarkGray)),
        sections[0],
    );

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Min(30)])
        .split(sections[1]);

    let preset_border = if editor.focus == FilterEditorFocus::Presets {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let preset_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(preset_border))
        .title(" Filter presets ");
    let preset_inner = preset_block.inner(main[0]);
    frame.render_widget(preset_block, main[0]);
    let preset_sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(4)])
        .split(preset_inner);
    let preset_items = editor
        .presets
        .iter()
        .map(|preset| {
            let (marker, colour) = match preset.source {
                FilterPresetSource::Example => ("E", Color::Cyan),
                FilterPresetSource::Saved => ("S", Color::Magenta),
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("{marker} "), Style::default().fg(colour)),
                Span::raw(preset.name.clone()),
            ]))
        })
        .collect::<Vec<_>>();
    let mut preset_state = ListState::default().with_selected(Some(editor.selected_preset));
    frame.render_stateful_widget(
        List::new(preset_items)
            .highlight_symbol("▸ ")
            .highlight_style(
                Style::default()
                    .bg(Color::Rgb(35, 48, 65))
                    .add_modifier(Modifier::BOLD),
            ),
        preset_sections[0],
        &mut preset_state,
    );
    let preset_description = editor.presets.get(editor.selected_preset).map_or_else(
        || Text::from("No presets"),
        |preset| {
            Text::from(vec![
                Line::from(Span::styled(
                    preset.source.label(),
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(preset.description.clone()),
            ])
        },
    );
    frame.render_widget(
        Paragraph::new(preset_description).wrap(Wrap { trim: false }),
        preset_sections[1],
    );

    let document_border = if editor.focus == FilterEditorFocus::Document {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let document_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(document_border))
        .title(" JSON document ");
    let editor_area = document_block.inner(main[1]);
    frame.render_widget(document_block, main[1]);
    let line_number_width = editor.lines.len().to_string().len();
    let gutter_width = u16::try_from(line_number_width + 3)
        .unwrap_or(u16::MAX)
        .min(editor_area.width);
    let content_width = editor_area.width.saturating_sub(gutter_width);
    editor.ensure_cursor_visible(usize::from(content_width), usize::from(editor_area.height));

    let lines = editor
        .lines
        .iter()
        .enumerate()
        .skip(editor.scroll_line)
        .take(usize::from(editor_area.height))
        .map(|(index, line)| {
            let number = format!("{:>line_number_width$} │ ", index + 1);
            Line::from(vec![
                Span::styled(number, Style::default().fg(Color::DarkGray)),
                Span::raw(visible_text(
                    line,
                    editor.scroll_column,
                    usize::from(content_width),
                )),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(lines), editor_area);

    let visible_end =
        (editor.scroll_line + usize::from(editor_area.height)).min(editor.lines.len());
    let location = format!(
        "Ln {}, Col {} • showing lines {}–{} of {}",
        editor.cursor_line + 1,
        editor.cursor_column + 1,
        editor.scroll_line + 1,
        visible_end,
        editor.lines.len()
    );
    let feedback = if let Some(error) = &editor.error {
        Span::styled(format!("Error: {error}"), Style::default().fg(Color::Red))
    } else if let Some(notice) = &editor.notice {
        Span::styled(notice.clone(), Style::default().fg(Color::Green))
    } else {
        Span::styled(location, Style::default().fg(Color::DarkGray))
    };
    let focus_help = if editor.completions.is_some() {
        "IntelliSense: ↑/↓ choose • Tab/Enter accept • Ctrl+Space refresh • Esc closes"
    } else {
        match editor.focus {
            FilterEditorFocus::Document => {
                "JSON: arrows move • Enter new line • Ctrl+Space IntelliSense • Tab presets"
            }
            FilterEditorFocus::Presets => {
                "Presets: ↑/↓ choose • Enter loads • s saves • Tab edits JSON"
            }
        }
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(focus_help),
            Line::from("Ctrl+P save preset • Ctrl+S apply • Ctrl+R reset • Esc cancel"),
            Line::from("E = built-in example • S = saved collection"),
            Line::from(feedback),
        ]),
        sections[2],
    );

    let cursor_position = if editor.focus == FilterEditorFocus::Document
        && editor.save_name.is_none()
        && content_width > 0
        && editor_area.height > 0
    {
        let cursor_x = display_width(
            &editor.lines[editor.cursor_line],
            editor.scroll_column,
            editor.cursor_column,
        );
        let cursor_x = u16::try_from(cursor_x)
            .unwrap_or(u16::MAX)
            .min(content_width.saturating_sub(1));
        let cursor_y = u16::try_from(editor.cursor_line.saturating_sub(editor.scroll_line))
            .unwrap_or(u16::MAX)
            .min(editor_area.height.saturating_sub(1));
        frame.set_cursor_position((
            editor_area.x + gutter_width + cursor_x,
            editor_area.y + cursor_y,
        ));
        Some((
            editor_area.x + gutter_width + cursor_x,
            editor_area.y + cursor_y,
        ))
    } else {
        None
    };

    if let (Some(completions), Some(cursor)) = (&editor.completions, cursor_position) {
        render_filter_completions(frame, completions, cursor, editor_area);
    }

    if let Some(name) = &editor.save_name {
        let prompt_area = centered_rect(64, 24, area);
        frame.render_widget(Clear, prompt_area);
        let prompt_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Save filter preset ");
        let prompt_inner = prompt_block.inner(prompt_area);
        frame.render_widget(prompt_block, prompt_area);
        frame.render_widget(
            Paragraph::new(vec![
                Line::from("Saved presets are also available as collections."),
                Line::from(""),
                Line::from(vec![
                    Span::styled("> ", Style::default().fg(Color::Cyan)),
                    Span::raw(name.clone()),
                    Span::styled("█", Style::default().fg(Color::Cyan)),
                ]),
                Line::from("Enter saves or updates • Esc cancels"),
            ]),
            prompt_inner,
        );
        let cursor_offset =
            u16::try_from(display_width(name, 0, name.chars().count())).unwrap_or(u16::MAX);
        frame.set_cursor_position((
            prompt_inner.x + 2 + cursor_offset.min(prompt_inner.width.saturating_sub(3)),
            prompt_inner.y + 2,
        ));
    }
}

fn render_filter_completions(
    frame: &mut Frame<'_>,
    completions: &FilterCompletionState,
    cursor: (u16, u16),
    editor_area: Rect,
) {
    if editor_area.width < 4 || editor_area.height < 3 {
        return;
    }
    let width = 54_u16.min(editor_area.width);
    let height = u16::try_from(completions.items.len().min(6) + 2)
        .unwrap_or(u16::MAX)
        .min(editor_area.height)
        .max(3);
    let x = cursor
        .0
        .min(editor_area.right().saturating_sub(width))
        .max(editor_area.x);
    let below = cursor.1.saturating_add(1);
    let y = if below.saturating_add(height) <= editor_area.bottom() {
        below
    } else {
        cursor.1.saturating_sub(height).max(editor_area.y)
    };
    let area = Rect::new(x, y, width, height);
    frame.render_widget(Clear, area);
    let items = completions
        .items
        .iter()
        .map(|completion| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<20}", completion.label),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(completion.description.clone()),
            ]))
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(completions.selected));
    frame.render_stateful_widget(
        List::new(items)
            .highlight_symbol("▸ ")
            .highlight_style(
                Style::default()
                    .bg(Color::Rgb(35, 48, 65))
                    .add_modifier(Modifier::BOLD),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan))
                    .title(" Filter IntelliSense "),
            ),
        area,
        &mut state,
    );
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

fn centered_fixed_height(percent_x: u16, height: u16, area: Rect) -> Rect {
    let width = area.width.saturating_mul(percent_x).saturating_div(100);
    let width = width.max(1).min(area.width);
    let height = height.max(1).min(area.height);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn parse_terminal_colour(hex: &str) -> Option<Color> {
    let value = hex.strip_prefix('#')?;
    if value.len() != 6 {
        return None;
    }
    Some(Color::Rgb(
        u8::from_str_radix(&value[0..2], 16).ok()?,
        u8::from_str_radix(&value[2..4], 16).ok()?,
        u8::from_str_radix(&value[4..6], 16).ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use ratatui::{Terminal, backend::TestBackend};

    use crate::{collection::get_collection, db::ImageStatus};

    use super::*;

    fn mock_image() -> ImageRecord {
        ImageRecord {
            id: 1,
            source_id: 1,
            path: "/wallpapers/mountain.jpg".into(),
            size: 100,
            modified_ns: 0,
            hash: Some("abc".into()),
            status: ImageStatus::Ready,
            error: None,
            width: Some(1920),
            height: Some(1080),
            ratio: Some(16.0 / 9.0),
            orientation: Some("landscape".into()),
            common_ratio: Some("16:9".into()),
            dominant_hex: Some("#204080".into()),
            dominant_name: Some("blue".into()),
            luminance: Some(0.3),
            saturation: Some(0.6),
            contrast: Some(0.2),
            light_dark: Some("dark".into()),
            thumbnail_path: None,
            palette: Vec::new(),
            ai_estimates: Vec::new(),
            favorite: true,
            tags: vec!["desktop".into()],
        }
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        (0..area.height)
            .map(|y| {
                (0..area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_owned()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn mock_app(mode: Mode) -> App {
        App {
            images: vec![mock_image()],
            selected: 0,
            filter: FilterSpecV1::default(),
            catalog: CatalogSummary {
                sources: 1,
                images: 1,
                ready_images: 1,
            },
            mode,
            status: "Ready — ? for help".into(),
            picker: Picker::halfblocks(),
            preview: None,
            preview_id: None,
            preview_requested_id: None,
            preview_worker: PreviewWorker::new(),
            scan_receiver: None,
            scan_started: None,
            scan_total: None,
            semantic_receiver: None,
            semantic_selected_id: None,
            semantic_previous_filter: None,
            semantic_started: None,
            wpaperd_receiver: None,
            wpaperd_refresh_queued: false,
            move_receiver: None,
            move_started: None,
            command_receiver: None,
            command_started: None,
            running_command: None,
            command_history: Vec::new(),
            pending_command_output: None,
            should_quit: false,
        }
    }

    #[test]
    fn semantic_reload_applies_results_and_restores_a_failed_filter() {
        let mut app = mock_app(Mode::Browse);
        let selected_id = app.images[0].id;
        let mut other = app.images[0].clone();
        other.id += 1;
        other.path = PathBuf::from("/wallpapers/other.png");
        let (sender, receiver) = unbounded();
        app.semantic_receiver = Some(receiver);
        app.semantic_selected_id = Some(selected_id);
        app.semantic_started = Some(Instant::now());
        sender
            .send(SemanticReloadResult::Finished(vec![
                other,
                app.images[0].clone(),
            ]))
            .expect("semantic result");

        app.poll_semantic_reload();

        assert_eq!(app.images.len(), 2);
        assert_eq!(app.selected, 1);
        assert!(app.semantic_receiver.is_none());
        assert!(app.status.starts_with("Semantic filter refreshed"));

        let previous_filter = app.filter.clone();
        app.filter.semantic_text = Some("mountains".into());
        let (sender, receiver) = unbounded();
        app.semantic_receiver = Some(receiver);
        app.semantic_previous_filter = Some(previous_filter.clone());
        sender
            .send(SemanticReloadResult::Failed("GPU unavailable".into()))
            .expect("semantic failure");

        app.poll_semantic_reload();

        assert_eq!(app.filter, previous_filter);
        assert!(app.semantic_receiver.is_none());
        assert_eq!(app.status, "Semantic filter failed: GPU unavailable");
    }

    #[test]
    fn preview_worker_results_cannot_replace_the_current_selection() {
        let mut app = mock_app(Mode::Browse);
        let selected_id = app.images[0].id;
        let (request_sender, _request_receiver) = unbounded();
        let (result_sender, result_receiver) = unbounded();
        app.preview_worker = PreviewWorker {
            request_sender,
            result_receiver,
        };
        app.preview_requested_id = Some(selected_id);

        result_sender
            .send(PreviewResult {
                image_id: selected_id + 1,
                image: Err("stale failure".into()),
            })
            .expect("stale result");
        app.poll_preview();
        assert_eq!(app.preview_requested_id, Some(selected_id));
        assert_eq!(app.status, "Ready — ? for help");

        result_sender
            .send(PreviewResult {
                image_id: selected_id,
                image: Err("selected failure".into()),
            })
            .expect("selected result");
        app.poll_preview();
        assert_eq!(app.preview_requested_id, None);
        assert_eq!(app.preview_id, Some(selected_id));
        assert_eq!(app.status, "Preview unavailable: selected failure");
    }

    #[test]
    fn wpaperd_worker_coalesces_refreshes_and_reports_bind_completion() {
        let (directory, paths, database, mut app) = empty_runtime();
        let (sender, receiver) = unbounded();
        app.wpaperd_receiver = Some(receiver);
        app.start_wpaperd_refresh(&database, &paths);
        assert!(app.wpaperd_refresh_queued);
        let (_semantic_sender, semantic_receiver) = unbounded();
        app.semantic_receiver = Some(semantic_receiver);
        sender
            .send(WpaperdBackgroundResult::Refreshed(None))
            .expect("refresh result");

        app.poll_wpaperd(&database, &paths);

        assert!(app.wpaperd_receiver.is_none());
        assert!(app.wpaperd_refresh_queued);

        app.semantic_receiver = None;
        app.wpaperd_refresh_queued = false;
        let (sender, receiver) = unbounded();
        app.wpaperd_receiver = Some(receiver);
        sender
            .send(WpaperdBackgroundResult::Bound("Bound DP-1 to all".into()))
            .expect("bind result");

        app.poll_wpaperd(&database, &paths);

        assert_eq!(app.status, "Bound DP-1 to all");
        drop(directory);
    }

    #[test]
    fn background_command_blocks_catalog_mutations_and_scans() {
        let (_directory, paths, database, mut app) = empty_runtime();
        let (_sender, receiver) = unbounded();
        app.command_receiver = Some(receiver);

        assert!(
            format!(
                "{:#}",
                app.ensure_catalog_mutation_idle().expect_err("busy")
            )
            .contains("background command")
        );
        app.start_scan(&database, &paths, &Config::default());
        assert!(app.scan_receiver.is_none());
        assert_eq!(
            app.status,
            "Wait for the background command before scanning"
        );
    }

    #[test]
    fn background_move_blocks_mutations_scans_and_early_exit() {
        let (_directory, paths, database, mut app) = empty_runtime();
        let (_sender, receiver) = unbounded();
        app.move_receiver = Some(receiver);
        app.move_started = Some(Instant::now());

        assert!(
            format!(
                "{:#}",
                app.ensure_catalog_mutation_idle().expect_err("busy")
            )
            .contains("background move")
        );
        app.start_scan(&database, &paths, &Config::default());
        assert!(app.scan_receiver.is_none());
        assert_eq!(app.status, "Wait for the background move before scanning");

        handle_browse_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE),
            &database,
            &paths,
            &Config::default(),
        )
        .expect("quit key");
        assert!(!app.should_quit);
        assert_eq!(
            app.status,
            "Wait for the background move to finish before quitting"
        );
    }

    fn press(editor: &mut FilterEditor, code: KeyCode) {
        assert!(matches!(
            editor.handle_key(KeyEvent::new(code, KeyModifiers::NONE)),
            FilterEditorCommand::Continue
        ));
    }

    fn press_control(editor: &mut FilterEditor, code: KeyCode) -> FilterEditorCommand {
        editor.handle_key(KeyEvent::new(code, KeyModifiers::CONTROL))
    }

    fn empty_runtime() -> (tempfile::TempDir, AppPaths, Database, App) {
        let directory = tempfile::tempdir().expect("tempdir");
        let paths = AppPaths::from_xdg_roots(
            directory.path().join("config"),
            directory.path().join("data"),
            directory.path().join("cache"),
            directory.path().join("state"),
        );
        paths.ensure_owned_dirs().expect("paths");
        let database = Database::open(&paths.database).expect("database");
        let app = App::new(&database, &paths).expect("app");
        (directory, paths, database, app)
    }

    fn runtime_with_image() -> (tempfile::TempDir, AppPaths, Database, App) {
        let (directory, paths, database, _) = empty_runtime();
        let source = directory.path().join("wallpapers");
        std::fs::create_dir(&source).expect("source");
        image::RgbImage::from_pixel(32, 18, image::Rgb([20, 60, 120]))
            .save(source.join("wall.png"))
            .expect("image");
        database.add_source(&source).expect("add source");
        crate::scan::scan_catalog(
            &database,
            &paths,
            &Config::default(),
            ScanOptions {
                full: false,
                no_ai: true,
            },
        )
        .expect("scan");
        let app = App::new(&database, &paths).expect("app");
        (directory, paths, database, app)
    }

    fn collection_key(app: &mut App, code: KeyCode, database: &Database, paths: &AppPaths) {
        handle_collections_key(
            app,
            KeyEvent::new(code, KeyModifiers::NONE),
            database,
            paths,
        )
        .expect("handle collections key");
    }

    fn type_collection_name(app: &mut App, value: &str, database: &Database, paths: &AppPaths) {
        for character in value.chars() {
            collection_key(app, KeyCode::Char(character), database, paths);
        }
    }

    fn saved_collection(id: i64, name: &str, filter: FilterSpecV1) -> SavedCollection {
        SavedCollection {
            id,
            name: name.into(),
            filter,
            created_at: 1_700_000_000_000,
            updated_at: 1_700_003_661_000,
        }
    }

    fn populated_collections_manager() -> CollectionsManager {
        let current_filter = FilterSpecV1 {
            min_width: Some(2560),
            orientations: vec![Orientation::Landscape],
            aspect_ratios: vec![16.0 / 9.0, 21.0 / 9.0],
            aspect_tolerance: 0.05,
            light_dark: vec![LightDark::Dark],
            palette_colours: vec![ColourFilter {
                hex: "#D08040".into(),
                max_distance: 0.1,
            }],
            tags: vec!["desktop".into(), "warm".into()],
            favorite: Some(true),
            ..FilterSpecV1::default()
        };
        CollectionsManager {
            collections: vec![
                saved_collection(1, "All wallpapers", FilterSpecV1::default()),
                saved_collection(2, "Warm widescreen", current_filter.clone()),
            ],
            bindings: vec![wpaperd::Binding {
                display: "DP-1".into(),
                collection_id: 2,
                collection_name: "Warm widescreen".into(),
                pool_path: "/tmp/bgm-test-pool".into(),
                displaced_path: None,
                active: true,
                refreshed_at: Some(1_700_003_661_000),
            }],
            current_filter,
            selected: 1,
            focus: CollectionsFocus::List,
            view: CollectionDetailView::Summary,
            detail_scroll: 0,
            detail_viewport_height: 1,
            prompt: None,
            error: None,
            notice: None,
        }
    }

    #[test]
    fn first_run_screen_explains_how_to_build_the_catalog() {
        let (_directory, _paths, _database, mut app) = empty_runtime();
        let backend = TestBackend::new(100, 28);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal.draw(|frame| draw(frame, &mut app)).expect("draw");

        let screen = buffer_text(&terminal);
        assert!(screen.contains("No sources yet"));
        assert!(screen.contains("Welcome to bgm"));
        assert!(screen.contains("Press a to add"));
        assert!(screen.contains("Press s to scan"));
    }

    #[test]
    fn help_content_fits_a_standard_small_terminal() {
        let mut app = mock_app(Mode::Help);
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal.draw(|frame| draw(frame, &mut app)).expect("draw");

        let screen = buffer_text(&terminal);
        assert!(screen.contains("Browse"));
        assert!(screen.contains("Command palette"));
        assert!(screen.contains("Filter editor"));
        assert!(screen.contains("Press any key to close help"));
    }

    #[test]
    fn source_input_registers_a_folder_and_points_to_scan() {
        let (directory, paths, database, mut app) = empty_runtime();
        let source = directory.path().join("wallpapers");
        std::fs::create_dir(&source).expect("source");

        submit_input(
            &mut app,
            InputAction::Source,
            source.display().to_string(),
            &database,
            &paths,
        )
        .expect("add source");

        assert_eq!(app.catalog.sources, 1);
        assert_eq!(database.list_sources().expect("sources").len(), 1);
        assert!(app.status.contains("press s to scan it"));
    }

    #[test]
    fn input_errors_stay_open_and_clear_when_edited() {
        let (_directory, paths, database, mut app) = empty_runtime();
        app.mode = Mode::Input {
            action: InputAction::Source,
            value: String::new(),
            error: None,
        };

        handle_input_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &database,
            &paths,
        )
        .expect("submit invalid input");
        assert!(matches!(
            &app.mode,
            Mode::Input {
                error: Some(error),
                ..
            } if error.contains("type the directory")
        ));

        handle_input_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE),
            &database,
            &paths,
        )
        .expect("edit input");
        assert!(matches!(
            &app.mode,
            Mode::Input { value, error: None, .. } if value == "/"
        ));
    }

    #[test]
    fn reset_filter_recovers_from_an_empty_result() {
        let (_directory, paths, database, mut app) = runtime_with_image();
        app.filter.favorite = Some(true);
        app.reload(&database, &paths).expect("filtered reload");
        assert!(app.images.is_empty());
        assert_eq!(app.empty_state(), Some(EmptyState::NoMatches));

        reset_filter(&mut app, &database, &paths).expect("reset filter");

        assert_eq!(app.filter, FilterSpecV1::default());
        assert_eq!(app.images.len(), 1);
        assert!(app.status.contains("Filter cleared"));
    }

    #[test]
    fn tag_input_can_remove_an_existing_tag() {
        let (_directory, paths, database, mut app) = runtime_with_image();
        let image_id = app.selected().expect("image").id;
        add_tag(&database, &[image_id], "desktop").expect("tag");
        app.reload(&database, &paths).expect("tagged reload");

        submit_input(
            &mut app,
            InputAction::Tag,
            "remove desktop".into(),
            &database,
            &paths,
        )
        .expect("remove tag");

        assert!(app.selected().expect("image").tags.is_empty());
        assert_eq!(app.status, "Removed tag desktop");
    }

    #[test]
    fn collections_manager_selects_the_current_filter_and_navigates_both_panes() {
        let (_directory, paths, database, mut app) = empty_runtime();
        let portrait = FilterSpecV1 {
            orientations: vec![Orientation::Portrait],
            ..FilterSpecV1::default()
        };
        save_collection(&database, "All", &FilterSpecV1::default()).expect("save all");
        save_collection(&database, "Portrait", &portrait).expect("save portrait");
        save_collection(
            &database,
            "Wide",
            &FilterSpecV1 {
                min_width: Some(2560),
                ..FilterSpecV1::default()
            },
        )
        .expect("save wide");
        app.filter = portrait;
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));

        let Mode::Collections(manager) = &app.mode else {
            panic!("manager did not open");
        };
        assert_eq!(manager.selected().expect("selection").name, "Portrait");

        collection_key(&mut app, KeyCode::Up, &database, &paths);
        collection_key(&mut app, KeyCode::Right, &database, &paths);
        collection_key(&mut app, KeyCode::Char('v'), &database, &paths);
        if let Mode::Collections(manager) = &mut app.mode {
            manager.detail_viewport_height = 3;
        }
        collection_key(&mut app, KeyCode::PageDown, &database, &paths);

        let Mode::Collections(manager) = &app.mode else {
            panic!("manager closed while navigating");
        };
        assert_eq!(manager.selected().expect("selection").name, "All");
        assert_eq!(manager.focus, CollectionsFocus::Details);
        assert_eq!(manager.view, CollectionDetailView::Json);
        assert!(manager.detail_scroll > 0);
        collection_key(&mut app, KeyCode::Tab, &database, &paths);
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager) if manager.focus == CollectionsFocus::List
        ));
    }

    #[test]
    fn empty_collections_manager_explains_save_and_escape_priority() {
        let (_directory, paths, database, mut app) = empty_runtime();
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));

        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if manager.error.as_deref().is_some_and(|error| error.contains("press s"))
        ));
        collection_key(&mut app, KeyCode::Char('s'), &database, &paths);
        collection_key(&mut app, KeyCode::Esc, &database, &paths);
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager) if manager.prompt.is_none()
        ));
        collection_key(&mut app, KeyCode::Esc, &database, &paths);
        assert!(matches!(app.mode, Mode::Browse));
    }

    #[test]
    fn collections_manager_loads_filters_and_keeps_model_errors_open() {
        let (_directory, paths, database, mut app) = runtime_with_image();
        let favorite = FilterSpecV1 {
            favorite: Some(true),
            ..FilterSpecV1::default()
        };
        save_collection(&database, "Favourites", &favorite).expect("save favorites");
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));

        collection_key(&mut app, KeyCode::Enter, &database, &paths);

        assert!(matches!(app.mode, Mode::Browse));
        assert_eq!(app.filter, favorite);
        assert!(app.images.is_empty());
        assert_eq!(app.status, "Loaded collection Favourites");

        save_collection(&database, "Current", &favorite).expect("save current");
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));
        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        assert!(matches!(app.mode, Mode::Browse));
        assert_eq!(app.status, "Collection Current is already current");

        let semantic = FilterSpecV1 {
            semantic_text: Some("misty forest".into()),
            ..FilterSpecV1::default()
        };
        save_collection(&database, "Semantic", &semantic).expect("save semantic");
        app.filter = FilterSpecV1::default();
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));
        if let Mode::Collections(manager) = &mut app.mode {
            manager.selected = manager
                .collections
                .iter()
                .position(|collection| collection.name == "Semantic")
                .expect("semantic selection");
        }
        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        assert_eq!(app.filter, FilterSpecV1::default());
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if manager.error.as_deref().is_some_and(|error| error.contains("model installed"))
        ));
    }

    #[test]
    fn collections_manager_saves_and_confirms_case_insensitive_overwrites() {
        let (_directory, paths, database, mut app) = empty_runtime();
        save_collection(&database, "Night", &FilterSpecV1::default()).expect("save existing");
        app.filter.favorite = Some(true);
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));

        collection_key(&mut app, KeyCode::Char('s'), &database, &paths);
        type_collection_name(&mut app, "night", &database, &paths);
        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if matches!(
                    manager.prompt,
                    Some(CollectionPrompt::Confirm(ref confirmation))
                        if matches!(
                            confirmation.as_ref(),
                            CollectionConfirmation::Overwrite { existing_name, .. }
                                if existing_name == "Night"
                        )
                )
        ));
        collection_key(&mut app, KeyCode::Esc, &database, &paths);
        assert_eq!(
            get_collection(&database, "Night")
                .expect("read")
                .expect("existing")
                .filter
                .favorite,
            None
        );

        collection_key(&mut app, KeyCode::Char('s'), &database, &paths);
        type_collection_name(&mut app, "NIGHT", &database, &paths);
        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        collection_key(&mut app, KeyCode::Char('y'), &database, &paths);

        let saved = get_collection(&database, "night")
            .expect("read")
            .expect("overwritten");
        assert_eq!(saved.name, "Night");
        assert_eq!(saved.filter.favorite, Some(true));
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if manager.selected().is_some_and(|collection| collection.id == saved.id)
                    && manager.notice.as_deref() == Some("Saved collection Night.")
        ));
        assert_eq!(app.status, "Saved collection Night");
    }

    #[test]
    fn collections_manager_updates_only_after_confirmation() {
        let (_directory, paths, database, mut app) = empty_runtime();
        save_collection(&database, "Desktop", &FilterSpecV1::default()).expect("save");
        app.filter.tags = vec!["desktop".into()];
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));

        collection_key(&mut app, KeyCode::Char('u'), &database, &paths);
        collection_key(&mut app, KeyCode::Esc, &database, &paths);
        assert!(
            get_collection(&database, "Desktop")
                .expect("read")
                .expect("collection")
                .filter
                .tags
                .is_empty()
        );

        collection_key(&mut app, KeyCode::Char('u'), &database, &paths);
        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        assert_eq!(
            get_collection(&database, "Desktop")
                .expect("read")
                .expect("updated")
                .filter
                .tags,
            ["desktop"]
        );
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if manager.notice.as_deref().is_some_and(|notice| notice.contains("Updated Desktop"))
        ));
    }

    #[test]
    fn collections_manager_cancels_or_confirms_deletion_and_keeps_nearest_selection() {
        let (_directory, paths, database, mut app) = empty_runtime();
        for name in ["Alpha", "Bravo", "Charlie"] {
            save_collection(&database, name, &FilterSpecV1::default()).expect("save");
        }
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));
        if let Mode::Collections(manager) = &mut app.mode {
            manager.selected = 1;
        }

        collection_key(&mut app, KeyCode::Char('d'), &database, &paths);
        collection_key(&mut app, KeyCode::Char('n'), &database, &paths);
        assert!(get_collection(&database, "Bravo").expect("read").is_some());

        collection_key(&mut app, KeyCode::Char('d'), &database, &paths);
        collection_key(&mut app, KeyCode::Char('y'), &database, &paths);

        assert!(get_collection(&database, "Bravo").expect("read").is_none());
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if manager.selected().is_some_and(|collection| collection.name == "Charlie")
                    && manager.notice.as_deref() == Some("Deleted collection Bravo.")
        ));
        assert_eq!(app.status, "Deleted collection Bravo");
    }

    #[test]
    fn collections_manager_refuses_to_delete_bound_collections_and_lists_displays() {
        let (_directory, paths, database, mut app) = runtime_with_image();
        save_collection(&database, "Displayed", &FilterSpecV1::default()).expect("save");
        wpaperd::bind(&database, &paths, "any", "Displayed").expect("bind any");
        wpaperd::bind(&database, &paths, "DP-1", "Displayed").expect("bind DP-1");
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));

        collection_key(&mut app, KeyCode::Char('d'), &database, &paths);

        assert!(
            get_collection(&database, "Displayed")
                .expect("read")
                .is_some()
        );
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if manager.prompt.is_none()
                    && manager.error.as_deref().is_some_and(|error| {
                        error.contains("any, DP-1") && error.contains("unbind first")
                    })
        ));
    }

    #[test]
    fn collection_name_entry_supports_editing_paste_and_persists_after_errors() {
        let (_directory, paths, database, mut app) = empty_runtime();
        app.mode = Mode::Collections(Box::new(
            CollectionsManager::load(&database, &app.filter).expect("manager"),
        ));
        collection_key(&mut app, KeyCode::Char('s'), &database, &paths);
        type_collection_name(&mut app, "ac", &database, &paths);
        collection_key(&mut app, KeyCode::Left, &database, &paths);
        handle_paste(&mut app, "β\n", &database);
        collection_key(&mut app, KeyCode::Home, &database, &paths);
        collection_key(&mut app, KeyCode::Delete, &database, &paths);
        collection_key(&mut app, KeyCode::End, &database, &paths);
        collection_key(&mut app, KeyCode::Backspace, &database, &paths);
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if matches!(
                    manager.prompt,
                    Some(CollectionPrompt::Name(CollectionNameEntry { ref value, cursor: 1 }))
                        if value == "β"
                )
        ));
        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        assert!(get_collection(&database, "β").expect("read").is_some());

        app.wpaperd_receiver = None;
        app.wpaperd_refresh_queued = false;
        collection_key(&mut app, KeyCode::Char('s'), &database, &paths);
        type_collection_name(&mut app, "   ", &database, &paths);
        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if matches!(manager.prompt, Some(CollectionPrompt::Name(_)))
                    && manager.error.as_deref() == Some("Collection name cannot be empty.")
        ));
        collection_key(&mut app, KeyCode::Esc, &database, &paths);

        let (_sender, receiver) = unbounded();
        app.wpaperd_receiver = Some(receiver);
        collection_key(&mut app, KeyCode::Char('s'), &database, &paths);
        type_collection_name(&mut app, "Busy", &database, &paths);
        collection_key(&mut app, KeyCode::Enter, &database, &paths);
        assert!(matches!(
            &app.mode,
            Mode::Collections(manager)
                if matches!(manager.prompt, Some(CollectionPrompt::Name(_)))
                    && manager.error.as_deref().is_some_and(|error| error.contains("wpaperd worker"))
        ));
        assert!(get_collection(&database, "Busy").expect("read").is_none());
    }

    #[test]
    fn readable_collection_summary_includes_every_filter_facet() {
        let filter = FilterSpecV1 {
            source_ids: vec![1, 2],
            paths: vec!["mountain".into()],
            min_width: Some(100),
            max_width: Some(200),
            min_height: Some(300),
            max_height: Some(400),
            orientations: vec![Orientation::Landscape],
            aspect_ratios: vec![16.0 / 9.0],
            aspect_tolerance: 0.05,
            light_dark: vec![LightDark::Dark],
            min_luminance: Some(0.1),
            max_luminance: Some(0.9),
            dominant_colours: vec![ColourFilter {
                hex: "#112233".into(),
                max_distance: 0.1,
            }],
            palette_colours: vec![ColourFilter {
                hex: "#445566".into(),
                max_distance: 0.2,
            }],
            ai_labels: vec![crate::filter::AiLabelFilter {
                pack: "mood".into(),
                label: "calm".into(),
                min_score: 0.7,
            }],
            semantic_text: Some("mist".into()),
            semantic_min_score: Some(0.25),
            tags: vec!["desktop".into()],
            favorite: Some(false),
            ..FilterSpecV1::default()
        };
        let summary = readable_filter_summary(&filter).join("\n");

        for label in [
            "Source IDs",
            "Path contains",
            "Minimum width",
            "Maximum width",
            "Minimum height",
            "Maximum height",
            "Orientations",
            "Aspect ratios",
            "Aspect tolerance",
            "Light/dark",
            "Minimum luminance",
            "Maximum luminance",
            "Dominant colour",
            "Palette colour",
            "AI label",
            "Semantic text",
            "Minimum semantic score",
            "Tags",
            "Favourite: no",
        ] {
            assert!(summary.contains(label), "missing {label} from {summary}");
        }
    }

    #[test]
    fn filter_editor_navigates_and_edits_across_lines() {
        let mut editor = FilterEditor::new("abcdef\nx\nabcdef".into(), Vec::new());

        press(&mut editor, KeyCode::End);
        press(&mut editor, KeyCode::Down);
        assert_eq!((editor.cursor_line, editor.cursor_column), (1, 1));
        press(&mut editor, KeyCode::Down);
        assert_eq!((editor.cursor_line, editor.cursor_column), (2, 6));

        press(&mut editor, KeyCode::Home);
        press(&mut editor, KeyCode::Backspace);
        assert_eq!(editor.value(), "abcdef\nxabcdef");
        assert_eq!((editor.cursor_line, editor.cursor_column), (1, 1));

        press(&mut editor, KeyCode::Delete);
        press(&mut editor, KeyCode::Char('界'));
        assert_eq!(editor.value(), "abcdef\nx界bcdef");
        press(&mut editor, KeyCode::Backspace);
        assert_eq!(editor.value(), "abcdef\nxbcdef");
    }

    #[test]
    fn filter_editor_accepts_multiline_paste() {
        let mut editor = FilterEditor::new(String::new(), Vec::new());
        editor.insert_text("{\r\n\t\"tags\": [\"café\"]\r\n}");

        assert_eq!(editor.value(), "{\n  \"tags\": [\"café\"]\n}");
        assert_eq!((editor.cursor_line, editor.cursor_column), (2, 1));
    }

    #[test]
    fn filter_editor_completes_contextual_enum_values() {
        let mut editor = FilterEditor::new(r#"{"orientations": []}"#.into(), Vec::new());
        editor.cursor_column = editor.lines[0]
            .chars()
            .position(|character| character == '[')
            .expect("array")
            + 1;

        assert!(matches!(
            press_control(&mut editor, KeyCode::Char(' ')),
            FilterEditorCommand::Continue
        ));
        let completions = editor.completions.as_ref().expect("completion popup");
        assert_eq!(
            completions
                .items
                .iter()
                .map(|completion| completion.label.as_str())
                .collect::<Vec<_>>(),
            ["landscape", "portrait", "square"]
        );

        press(&mut editor, KeyCode::Down);
        press(&mut editor, KeyCode::Enter);

        assert_eq!(editor.value(), r#"{"orientations": ["portrait"]}"#);
        assert!(editor.completions.is_none());
        assert_eq!(editor.notice.as_deref(), Some("Inserted portrait."));
    }

    #[test]
    fn filter_editor_filters_completions_as_the_user_types() {
        let mut editor = FilterEditor::new(r#"{"light_dark": [""]}"#.into(), Vec::new());
        editor.cursor_column = editor.lines[0].rfind('"').expect("closing string");

        press(&mut editor, KeyCode::Char('d'));

        let completions = editor.completions.as_ref().expect("filtered popup");
        assert_eq!(completions.items.len(), 1);
        assert_eq!(completions.items[0].label, "dark");
        press(&mut editor, KeyCode::Tab);
        assert_eq!(editor.value(), r#"{"light_dark": ["dark"]}"#);
        assert_eq!(editor.focus, FilterEditorFocus::Document);
    }

    #[test]
    fn escape_closes_filter_intellisense_before_the_editor() {
        let mut editor = FilterEditor::new(r#"{"favorite": null}"#.into(), Vec::new());
        editor.cursor_column = editor.lines[0].find("null").expect("null");
        assert!(matches!(
            press_control(&mut editor, KeyCode::Char(' ')),
            FilterEditorCommand::Continue
        ));

        assert!(matches!(
            editor.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            FilterEditorCommand::Continue
        ));
        assert!(editor.completions.is_none());
        assert!(matches!(
            editor.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            FilterEditorCommand::Cancel
        ));
    }

    #[test]
    fn json_delimiters_do_not_make_enter_accept_an_unrequested_completion() {
        let mut editor = FilterEditor::new(r#"{"favorite": true}"#.into(), Vec::new());
        editor.cursor_column = editor.lines[0].chars().count() - 1;

        press(&mut editor, KeyCode::Char(','));
        assert!(editor.completions.is_none());
        press(&mut editor, KeyCode::Enter);

        assert_eq!(editor.value(), "{\"favorite\": true,\n}");
    }

    #[test]
    fn filter_editor_scrolls_to_keep_cursor_visible() {
        let value = (0..20)
            .map(|index| format!("line {index:02}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut editor = FilterEditor::new(value, Vec::new());
        assert!(matches!(
            press_control(&mut editor, KeyCode::End),
            FilterEditorCommand::Continue
        ));

        editor.ensure_cursor_visible(5, 4);

        assert_eq!(editor.scroll_line, 16);
        assert_eq!(editor.scroll_column, 3);
        assert_eq!(
            display_width(
                &editor.lines[editor.cursor_line],
                editor.scroll_column,
                editor.cursor_column,
            ),
            4
        );
        assert_eq!(visible_text("a界b", 0, 3), "a界");
    }

    #[test]
    fn filter_editor_commands_do_not_modify_the_document() {
        let mut editor = FilterEditor::new("{}".into(), Vec::new());

        assert!(matches!(
            press_control(&mut editor, KeyCode::Char('s')),
            FilterEditorCommand::Apply
        ));
        assert!(matches!(
            press_control(&mut editor, KeyCode::Char('r')),
            FilterEditorCommand::Reset
        ));
        assert!(matches!(
            editor.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            FilterEditorCommand::Cancel
        ));
        assert_eq!(editor.value(), "{}");
    }

    #[test]
    fn invalid_filter_stays_open_for_correction() {
        let (_directory, paths, database, mut app) = empty_runtime();
        app.mode = Mode::FilterEditor(FilterEditor::new("{".into(), Vec::new()));

        handle_filter_editor_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL),
            &database,
            &paths,
        )
        .expect("handle key");

        let Mode::FilterEditor(editor) = &app.mode else {
            panic!("invalid filter closed the editor");
        };
        assert!(
            editor
                .error
                .as_deref()
                .is_some_and(|error| { error.contains("filter must be valid FilterSpecV1 JSON") })
        );
    }

    #[test]
    fn valid_partial_filter_applies_and_closes_the_editor() {
        let (_directory, paths, database, mut app) = empty_runtime();
        app.mode = Mode::FilterEditor(FilterEditor::new(
            r#"{"min_width":2560,"orientations":["landscape"]}"#.into(),
            Vec::new(),
        ));

        handle_filter_editor_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL),
            &database,
            &paths,
        )
        .expect("handle key");

        assert!(matches!(app.mode, Mode::Browse));
        assert_eq!(app.filter.min_width, Some(2560));
        assert_eq!(app.filter.orientations.len(), 1);
        assert_eq!(app.status, "Filter applied — 0 result(s)");
    }

    #[test]
    fn built_in_filter_presets_are_valid_examples() {
        let presets = filter_presets(Vec::new());

        assert_eq!(presets.len(), 6);
        assert!(
            presets
                .iter()
                .all(|preset| preset.source == FilterPresetSource::Example)
        );
        for preset in presets {
            preset.filter.validate().expect("valid example preset");
        }
    }

    #[test]
    fn filter_editor_loads_a_selected_example_into_the_document() {
        let (_directory, paths, database, mut app) = empty_runtime();
        app.mode = Mode::FilterEditor(FilterEditor::new("{}".into(), Vec::new()));

        for key in [KeyCode::Tab, KeyCode::Down, KeyCode::Enter] {
            handle_filter_editor_key(
                &mut app,
                KeyEvent::new(key, KeyModifiers::NONE),
                &database,
                &paths,
            )
            .expect("handle preset key");
        }

        let Mode::FilterEditor(editor) = &app.mode else {
            panic!("loading a preset closed the editor");
        };
        let loaded = parse_filter(&editor.value()).expect("loaded preset JSON");
        assert_eq!(loaded.orientations, vec![Orientation::Landscape]);
        assert_eq!(loaded.light_dark, vec![LightDark::Dark]);
        assert_eq!(editor.focus, FilterEditorFocus::Document);
        assert_eq!(
            editor.notice.as_deref(),
            Some("Loaded Dark landscapes; Ctrl+S applies it.")
        );
    }

    #[test]
    fn filter_editor_saves_the_document_as_a_selectable_preset() {
        let (_directory, paths, database, mut app) = empty_runtime();
        let filter = FilterSpecV1 {
            favorite: Some(true),
            ..FilterSpecV1::default()
        };
        app.mode = Mode::FilterEditor(FilterEditor::new(
            serde_json::to_string_pretty(&filter).expect("serialize"),
            Vec::new(),
        ));

        handle_filter_editor_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
            &database,
            &paths,
        )
        .expect("open save prompt");
        for character in "My favourites".chars() {
            handle_filter_editor_key(
                &mut app,
                KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE),
                &database,
                &paths,
            )
            .expect("type preset name");
        }
        handle_filter_editor_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &database,
            &paths,
        )
        .expect("save preset");

        let saved = get_collection(&database, "my favourites")
            .expect("read preset")
            .expect("saved preset");
        assert_eq!(saved.filter.favorite, Some(true));
        let Mode::FilterEditor(editor) = &app.mode else {
            panic!("saving a preset closed the editor");
        };
        assert_eq!(editor.focus, FilterEditorFocus::Presets);
        assert!(editor.presets.iter().any(|preset| {
            preset.source == FilterPresetSource::Saved && preset.name == "My favourites"
        }));
        assert_eq!(app.status, "Saved filter preset My favourites");
    }

    #[test]
    fn filter_editor_renders_the_full_document_with_scrolling() {
        let value = serde_json::to_string_pretty(&FilterSpecV1::default()).expect("serialize");
        let mut app = mock_app(Mode::FilterEditor(FilterEditor::new(value, Vec::new())));
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| render_modal(frame, &mut app))
            .expect("draw top");
        let top = buffer_text(&terminal);
        assert!(top.contains("\"version\": 1"));
        assert!(top.contains("All wallpapers"));
        assert!(top.contains("Dark landscapes"));
        assert!(top.contains("Ctrl+P save preset"));
        assert!(top.contains("Ctrl+S apply"));
        insta::assert_snapshot!("filter_editor", top);

        if let Mode::FilterEditor(editor) = &mut app.mode {
            assert!(matches!(
                press_control(editor, KeyCode::End),
                FilterEditorCommand::Continue
            ));
        }
        terminal
            .draw(|frame| render_modal(frame, &mut app))
            .expect("draw bottom");
        let bottom = buffer_text(&terminal);
        assert!(bottom.contains("\"favorite\": null"));
        assert!(bottom.contains("showing lines"));
    }

    #[test]
    fn filter_editor_renders_intellisense_suggestions() {
        let mut editor = FilterEditor::new(r#"{"orientations": []}"#.into(), Vec::new());
        editor.cursor_column = editor.lines[0]
            .chars()
            .position(|character| character == '[')
            .expect("array")
            + 1;
        assert!(matches!(
            press_control(&mut editor, KeyCode::Char(' ')),
            FilterEditorCommand::Continue
        ));
        let mut app = mock_app(Mode::FilterEditor(editor));
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| render_modal(frame, &mut app))
            .expect("draw IntelliSense");
        let screen = buffer_text(&terminal);

        assert!(screen.contains("Filter IntelliSense"));
        assert!(screen.contains("landscape"));
        assert!(screen.contains("portrait"));
        assert!(screen.contains("Ctrl+Space refresh"));
        insta::assert_snapshot!("filter_editor_intellisense", screen);
    }

    #[test]
    fn command_palette_completes_commands_and_clap_subcommands() {
        let (_directory, _paths, database, _app) = empty_runtime();
        let image = mock_image();
        let mut palette = CommandPalette::new(&database, Some(&image));

        for character in "col".chars() {
            assert!(matches!(
                palette.handle_key(
                    KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE),
                    &[],
                    &database,
                ),
                CommandPaletteAction::Continue
            ));
        }
        assert_eq!(palette.suggestions.len(), 1);
        assert_eq!(palette.suggestions[0].label, "collection");
        palette.handle_key(
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &[],
            &database,
        );

        assert_eq!(palette.value, "collection ");
        assert!(
            palette
                .suggestions
                .iter()
                .any(|suggestion| suggestion.label == "save")
        );
    }

    #[test]
    fn command_palette_completes_live_collection_names_with_quoting() {
        let (_directory, _paths, database, _app) = empty_runtime();
        save_collection(&database, "Night skies", &FilterSpecV1::default())
            .expect("save collection");
        let image = mock_image();
        let mut palette = CommandPalette::new(&database, Some(&image));
        palette.value = "collection show Ni".into();
        palette.cursor = palette.value.chars().count();
        palette.refresh_suggestions(&database);
        palette.selected_suggestion = palette
            .suggestions
            .iter()
            .position(|suggestion| suggestion.label == "Night skies")
            .expect("live collection completion");

        palette.accept_suggestion(&database);

        assert_eq!(palette.value, "collection show 'Night skies' ");
    }

    #[test]
    fn command_palette_completes_catalog_tags() {
        let (_directory, _paths, database, _app) = empty_runtime();
        database
            .with_connection(|connection| {
                connection.execute("INSERT INTO tags(name) VALUES ('desktop')", [])?;
                Ok(())
            })
            .expect("insert tag");
        let mut palette = CommandPalette::new(&database, None);
        palette.value = "search --tag des".into();
        palette.cursor = palette.value.chars().count();
        palette.refresh_suggestions(&database);

        assert!(
            palette
                .suggestions
                .iter()
                .any(|suggestion| suggestion.label == "desktop")
        );
    }

    #[test]
    fn command_palette_rejects_a_nested_tui_without_closing() {
        let (_directory, _paths, database, mut app) = empty_runtime();
        let mut palette = CommandPalette::new(&database, None);
        palette.value = "tui".into();
        palette.cursor = 3;
        palette.refresh_suggestions(&database);
        app.mode = Mode::CommandPalette(palette);

        handle_command_palette_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &database,
        )
        .expect("handle command");

        let Mode::CommandPalette(palette) = &app.mode else {
            panic!("invalid command closed the palette");
        };
        assert!(
            palette
                .error
                .as_deref()
                .is_some_and(|error| error.contains("already inside the TUI"))
        );
    }

    #[test]
    fn completed_command_does_not_discard_an_open_dialog() {
        let (_directory, paths, database, mut app) = empty_runtime();
        app.mode = Mode::Help;
        app.present_command_output(CommandOutput {
            command: "doctor".into(),
            success: true,
            stdout: "ok".into(),
            stderr: String::new(),
            scroll: 0,
            viewport_height: 1,
        });

        assert!(matches!(app.mode, Mode::Help));
        assert!(app.pending_command_output.is_some());

        app.mode = Mode::Browse;
        let mut config = Config::default();
        app.poll_command(&database, &paths, &mut config);
        assert!(matches!(app.mode, Mode::CommandOutput(_)));
        assert!(app.pending_command_output.is_none());
    }

    #[test]
    fn command_palette_renders_intellisense() {
        let (_directory, _paths, database, _app) = empty_runtime();
        let mut palette = CommandPalette::new(&database, Some(&mock_image()));
        palette.value = "collection ".into();
        palette.cursor = palette.value.chars().count();
        palette.refresh_suggestions(&database);
        let mut app = mock_app(Mode::CommandPalette(palette));
        let backend = TestBackend::new(110, 32);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| render_modal(frame, &mut app))
            .expect("draw palette");
        let screen = buffer_text(&terminal);

        assert!(screen.contains("Command palette"));
        assert!(screen.contains("IntelliSense"));
        assert!(screen.contains("save"));
        assert!(screen.contains("Ctrl+P/N history"));
        insta::assert_snapshot!("command_palette", screen);
    }

    #[test]
    fn populated_collections_manager_snapshot() {
        let manager = populated_collections_manager();
        let mut app = mock_app(Mode::Collections(Box::new(manager)));
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| render_modal(frame, &mut app))
            .expect("draw collections manager");
        let screen = buffer_text(&terminal);

        assert!(screen.contains("Collections manager"));
        assert!(screen.contains("Warm widescreen"));
        assert!(screen.contains("exact match"));
        assert!(screen.contains("wpaperd displays: DP-1"));
        assert!(screen.contains("Minimum width: 2560 px"));
        assert!(screen.contains("Enter load"));
        assert!(screen.contains("● current filter"));
        insta::assert_snapshot!("collections_manager_populated", screen);
    }

    #[test]
    fn empty_collections_manager_snapshot() {
        let manager = CollectionsManager {
            collections: Vec::new(),
            bindings: Vec::new(),
            current_filter: FilterSpecV1::default(),
            selected: 0,
            focus: CollectionsFocus::List,
            view: CollectionDetailView::Summary,
            detail_scroll: 0,
            detail_viewport_height: 1,
            prompt: None,
            error: None,
            notice: None,
        };
        let mut app = mock_app(Mode::Collections(Box::new(manager)));
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| render_modal(frame, &mut app))
            .expect("draw empty collections manager");
        let screen = buffer_text(&terminal);

        assert!(screen.contains("No saved collections"));
        assert!(screen.contains("Press s to save the"));
        assert!(screen.contains("current browser filter"));
        insta::assert_snapshot!("collections_manager_empty", screen);
    }

    #[test]
    fn collections_manager_confirmation_prompt_snapshot() {
        let mut manager = populated_collections_manager();
        manager.prompt = Some(CollectionPrompt::Confirm(Box::new(
            CollectionConfirmation::Overwrite {
                name: "warm widescreen".into(),
                existing_name: "Warm widescreen".into(),
            },
        )));
        let mut app = mock_app(Mode::Collections(Box::new(manager)));
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| render_modal(frame, &mut app))
            .expect("draw collection confirmation");
        let screen = buffer_text(&terminal);

        assert!(screen.contains("Confirm overwrite"));
        assert!(screen.contains("Warm widescreen already exists"));
        assert!(screen.contains("Enter or y confirms"));
        insta::assert_snapshot!("collections_manager_confirmation", screen);
    }

    #[test]
    fn browser_screen_snapshot() {
        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let app = mock_app(Mode::Browse);
        terminal
            .draw(|frame| {
                let areas = render_base(frame, &app);
                frame.render_widget(
                    Paragraph::new("[mock image]")
                        .alignment(Alignment::Center)
                        .block(Block::default().borders(Borders::ALL).title(" Preview ")),
                    areas.preview,
                );
            })
            .expect("draw");
        let snapshot = buffer_text(&terminal);
        insta::assert_snapshot!("browser", snapshot);
    }
}
