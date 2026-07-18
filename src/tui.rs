use std::{
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, TryRecvError, unbounded};
use crossterm::{
    cursor::Show,
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyEvent, KeyModifiers,
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
        SavedCollection, add_tag, delete_collection, get_collection, list_collections,
        save_collection, search_resolved, set_favorite,
    },
    config::Config,
    db::{Database, ImageRecord},
    filter::{ColourFilter, FilterSpecV1},
    model,
    move_files::{MovePlan, apply_move, plan_move},
    scan::{ScanEvent, ScanOptions, ScanReport, scan_catalog_with_progress},
    wpaperd,
};

#[derive(Clone, Copy)]
enum InputAction {
    Tag,
    Collection,
    Move,
    Bind,
}

enum Mode {
    Browse,
    FilterEditor(FilterEditor),
    Input { action: InputAction, value: String },
    CommandPalette(CommandPalette),
    CommandOutput(CommandOutput),
    ConfirmMove(MovePlan),
    Help,
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
    }

    fn handle_key(&mut self, key: KeyEvent) -> FilterEditorCommand {
        if self.save_name.is_some() {
            return self.handle_save_name_key(key);
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => FilterEditorCommand::Cancel,
            KeyCode::Char('s' | 'S') if control => FilterEditorCommand::Apply,
            KeyCode::Enter if control => FilterEditorCommand::Apply,
            KeyCode::Char('r' | 'R') if control => FilterEditorCommand::Reset,
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
                FilterEditorCommand::Continue
            }
            _ if self.focus == FilterEditorFocus::Presets => self.handle_preset_key(key),
            KeyCode::Home if control => {
                self.cursor_line = 0;
                self.cursor_column = 0;
                self.preferred_column = None;
                FilterEditorCommand::Continue
            }
            KeyCode::End if control => {
                self.cursor_line = self.lines.len() - 1;
                self.cursor_column = self.current_line_len();
                self.preferred_column = None;
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
                FilterEditorCommand::Continue
            }
            KeyCode::End => {
                self.cursor_column = self.current_line_len();
                self.preferred_column = None;
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
    }

    fn insert_newline(&mut self) {
        let byte_index = char_to_byte_index(&self.lines[self.cursor_line], self.cursor_column);
        let remainder = self.lines[self.cursor_line].split_off(byte_index);
        self.cursor_line += 1;
        self.lines.insert(self.cursor_line, remainder);
        self.cursor_column = 0;
        self.changed();
    }

    fn backspace(&mut self) {
        if self.cursor_column > 0 {
            self.cursor_column -= 1;
            let byte_index = char_to_byte_index(&self.lines[self.cursor_line], self.cursor_column);
            self.lines[self.cursor_line].remove(byte_index);
            self.changed();
        } else if self.cursor_line > 0 {
            let current = self.lines.remove(self.cursor_line);
            self.cursor_line -= 1;
            self.cursor_column = self.current_line_len();
            self.lines[self.cursor_line].push_str(&current);
            self.changed();
        }
    }

    fn delete(&mut self) {
        if self.cursor_column < self.current_line_len() {
            let byte_index = char_to_byte_index(&self.lines[self.cursor_line], self.cursor_column);
            self.lines[self.cursor_line].remove(byte_index);
            self.changed();
        } else if self.cursor_line + 1 < self.lines.len() {
            let next = self.lines.remove(self.cursor_line + 1);
            self.lines[self.cursor_line].push_str(&next);
            self.changed();
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
    }

    fn move_right(&mut self) {
        if self.cursor_column < self.current_line_len() {
            self.cursor_column += 1;
        } else if self.cursor_line + 1 < self.lines.len() {
            self.cursor_line += 1;
            self.cursor_column = 0;
        }
        self.preferred_column = None;
    }

    fn move_vertical(&mut self, amount: isize) {
        let preferred = self.preferred_column.unwrap_or(self.cursor_column);
        self.cursor_line = self
            .cursor_line
            .saturating_add_signed(amount)
            .min(self.lines.len() - 1);
        self.cursor_column = preferred.min(self.current_line_len());
        self.preferred_column = Some(preferred);
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

struct App {
    images: Vec<ImageRecord>,
    selected: usize,
    filter: FilterSpecV1,
    mode: Mode,
    status: String,
    picker: Picker,
    preview: Option<StatefulProtocol>,
    preview_id: Option<i64>,
    scan_receiver: Option<Receiver<BackgroundResult>>,
    scan_started: Option<Instant>,
    scan_total: Option<usize>,
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
            mode: Mode::Browse,
            status: "Ready — ? for help".into(),
            picker,
            preview: None,
            preview_id: None,
            scan_receiver: None,
            scan_started: None,
            scan_total: None,
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

    fn reload(&mut self, database: &Database, paths: &AppPaths) -> Result<()> {
        let selected_id = self.selected().map(|image| image.id);
        if self.filter.semantic_text.is_some() && !model::status(paths, false).verified {
            anyhow::bail!(
                "semantic TUI filters need the model installed first; run `bgm model install --yes`"
            );
        }
        self.images = search_resolved(database, paths, &self.filter)?
            .into_iter()
            .map(|result| result.image)
            .collect();
        self.selected = selected_id
            .and_then(|id| self.images.iter().position(|image| image.id == id))
            .unwrap_or(0)
            .min(self.images.len().saturating_sub(1));
        self.preview_id = None;
        self.load_preview();
        Ok(())
    }

    fn load_preview(&mut self) {
        let Some(image) = self.selected() else {
            self.preview = None;
            self.preview_id = None;
            return;
        };
        if self.preview_id == Some(image.id) {
            return;
        }
        let id = image.id;
        let path = image.thumbnail_path.as_ref().unwrap_or(&image.path).clone();
        match image::open(&path) {
            Ok(image) => {
                self.preview = Some(self.picker.new_resize_protocol(image));
                self.preview_id = Some(id);
            }
            Err(error) => {
                self.preview = None;
                self.preview_id = Some(id);
                self.status = format!("Preview unavailable: {error}");
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
                    Some(model::analyze_missing(&database, &paths)?)
                } else {
                    None
                };
                let _ = wpaperd::refresh(&database, &paths, None);
                Ok(BackgroundResult::Finished { scan, ai })
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
                Ok(BackgroundResult::Finished { scan, ai }) => {
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
        if self.command_receiver.is_some() {
            anyhow::bail!("another command is already running");
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
    if let Err(error) = execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    ) {
        let _ = disable_raw_mode();
        let _ = execute!(
            stdout,
            LeaveAlternateScreen,
            DisableBracketedPaste,
            DisableMouseCapture,
            Show
        );
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
            DisableMouseCapture,
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
        app.poll_command(database, paths, config);
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
    if matches!(&app.mode, Mode::CommandPalette(_)) {
        return handle_command_palette_key(app, key, database);
    }
    match &mut app.mode {
        Mode::Browse => handle_browse_key(app, key, database, paths, config),
        Mode::FilterEditor(_) => unreachable!("filter editor handled above"),
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
        Mode::Input { action, value } => match key.code {
            KeyCode::Esc => {
                app.mode = Mode::Browse;
                Ok(())
            }
            KeyCode::Backspace => {
                value.pop();
                Ok(())
            }
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                value.push(character);
                Ok(())
            }
            KeyCode::Enter => {
                let value = value.trim().to_owned();
                let action = *action;
                app.mode = Mode::Browse;
                submit_input(app, action, value, database, paths)
            }
            _ => Ok(()),
        },
        Mode::ConfirmMove(plan) => match key.code {
            KeyCode::Char('y' | 'Y') => {
                let plan = plan.clone();
                app.mode = Mode::Browse;
                let result = apply_move(database, paths, plan)?;
                app.status = format!("Moved {} file(s); undo ID {}", result.moved, result.id);
                let _ = wpaperd::refresh(database, paths, None);
                app.reload(database, paths)
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
                    let _ = wpaperd::refresh(database, paths, None);
                    if let Mode::FilterEditor(editor) = &mut app.mode {
                        editor.refresh_saved_presets(presets, &saved.name);
                        editor.focus = FilterEditorFocus::Presets;
                        editor.notice = Some(format!("Saved preset {}.", saved.name));
                        editor.error = None;
                    }
                    app.status = format!("Saved filter preset {}", saved.name);
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

fn handle_paste(app: &mut App, value: &str, database: &Database) {
    match &mut app.mode {
        Mode::FilterEditor(editor) => editor.paste(value),
        Mode::CommandPalette(palette) => palette.paste(value, database),
        Mode::Input {
            value: input_value, ..
        } => input_value.extend(
            value
                .chars()
                .filter(|character| !matches!(character, '\r' | '\n') && !character.is_control()),
        ),
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
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
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
        KeyCode::Char('t') => {
            app.mode = Mode::Input {
                action: InputAction::Tag,
                value: String::new(),
            };
        }
        KeyCode::Char('c') => {
            app.mode = Mode::Input {
                action: InputAction::Collection,
                value: String::new(),
            };
        }
        KeyCode::Char('m') => {
            app.mode = Mode::Input {
                action: InputAction::Move,
                value: String::new(),
            };
        }
        KeyCode::Char('w') => {
            app.mode = Mode::Input {
                action: InputAction::Bind,
                value: "any ".into(),
            };
        }
        KeyCode::Char('f') => {
            if let Some(image) = app.selected() {
                let (id, favorite) = (image.id, !image.favorite);
                set_favorite(database, &[id], favorite)?;
                let _ = wpaperd::refresh(database, paths, None);
                app.reload(database, paths)?;
                app.status = if favorite {
                    "Marked favorite".into()
                } else {
                    "Removed favorite".into()
                };
            }
        }
        KeyCode::Char('o') | KeyCode::Enter => {
            if let Some(image) = app.selected() {
                Command::new("xdg-open")
                    .arg(&image.path)
                    .spawn()
                    .context("failed to start xdg-open")?;
                app.status = format!("Opened {}", image.path.display());
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

fn submit_input(
    app: &mut App,
    action: InputAction,
    value: String,
    database: &Database,
    paths: &AppPaths,
) -> Result<()> {
    match action {
        InputAction::Tag => {
            if !value.is_empty()
                && let Some(image) = app.selected()
            {
                add_tag(database, &[image.id], &value)?;
                let _ = wpaperd::refresh(database, paths, None);
                app.reload(database, paths)?;
                app.status = format!("Added tag {value}");
            }
        }
        InputAction::Collection => {
            if value.eq_ignore_ascii_case("list") {
                let names = list_collections(database)?
                    .into_iter()
                    .map(|collection| collection.name)
                    .collect::<Vec<_>>();
                app.status = if names.is_empty() {
                    "No saved collections".into()
                } else {
                    format!("Collections: {}", names.join(", "))
                };
            } else if let Some(name) = value.strip_prefix("load ").map(str::trim) {
                let collection = get_collection(database, name)?
                    .with_context(|| format!("collection not found: {name}"))?;
                app.filter = collection.filter;
                app.reload(database, paths)?;
                app.status = format!("Loaded collection {}", collection.name);
            } else if let Some(name) = value.strip_prefix("delete ").map(str::trim) {
                if !delete_collection(database, name)? {
                    anyhow::bail!("collection not found: {name}");
                }
                let _ = wpaperd::refresh(database, paths, None);
                app.status = format!("Deleted collection {name}");
            } else if !value.is_empty() {
                let name = value
                    .strip_prefix("save ")
                    .map_or(value.as_str(), str::trim);
                let collection = save_collection(database, name, &app.filter)?;
                let _ = wpaperd::refresh(database, paths, None);
                app.status = format!("Saved collection {}", collection.name);
            }
        }
        InputAction::Move => {
            if !value.is_empty()
                && let Some(image) = app.selected()
            {
                let plan = plan_move(std::slice::from_ref(image), &PathBuf::from(value))?;
                app.mode = Mode::ConfirmMove(plan);
            }
        }
        InputAction::Bind => {
            if let Some(display) = value.strip_prefix("unbind ").map(str::trim) {
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
                let binding = wpaperd::bind(database, paths, display, collection.trim())?;
                app.status = format!("Bound {} to {}", binding.display, binding.collection_name);
            }
        }
    }
    Ok(())
}

fn apply_filter(app: &mut App, value: &str, database: &Database, paths: &AppPaths) -> Result<()> {
    let filter = parse_filter(value)?;

    let previous_filter = std::mem::replace(&mut app.filter, filter);
    if let Err(error) = app.reload(database, paths) {
        app.filter = previous_filter;
        return Err(error);
    }
    app.status = format!("Filter applied — {} result(s)", app.images.len());
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
        frame.render_widget(
            Paragraph::new("No preview\n\nPress Enter or o to use xdg-open")
                .alignment(Alignment::Center)
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

    let title = format!(
        " bgm — {} wallpaper{} — filter v{} ",
        app.images.len(),
        if app.images.len() == 1 { "" } else { "s" },
        app.filter.version
    );
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
    let mut list_state =
        ListState::default().with_selected((!app.images.is_empty()).then_some(app.selected));
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

    let metadata = app
        .selected()
        .map_or_else(|| Text::from("No matching images"), metadata_text);
    frame.render_widget(
        Paragraph::new(metadata).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Metadata • palette • AI estimates "),
        ),
        body[2],
    );

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("↑/↓", Style::default().fg(Color::Cyan)),
            Span::raw(" navigate  "),
            Span::styled("/", Style::default().fg(Color::Cyan)),
            Span::raw(" filter  : commands  f favorite  t tag  c collection  m move  w bind  s scan  o open  ? help  q quit"),
        ]))
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

fn render_modal(frame: &mut Frame<'_>, app: &mut App) {
    match &mut app.mode {
        Mode::Browse => {}
        Mode::FilterEditor(editor) => render_filter_editor(frame, editor),
        Mode::CommandPalette(palette) => render_command_palette(frame, palette),
        Mode::CommandOutput(output) => render_command_output(frame, output),
        Mode::Help => {
            let area = centered_rect(76, 86, frame.area());
            frame.render_widget(Clear, area);
            frame.render_widget(
                Paragraph::new(
                    "Keyboard\n\n↑/↓, j/k    navigate\n:            run any bgm CLI command\n/            filter examples, presets, and JSON\nf/t/c        favorite, tag, collections\nm/w/s        move, wpaperd binding, background scan\no or Enter   open with xdg-open\nq            quit\n\nCommand palette\nTab / ↑/↓    complete / choose suggestion\nCtrl+P/N     previous / next command in history\n←/→ Home End edit; Ctrl+W deletes a word\nEnter runs in background; Esc closes\n\nFilter editor\nTab / ↑/↓    switch pane / choose preset\nEnter loads preset or inserts a JSON line\nCtrl+P       save JSON as a named preset\nCtrl+S/R     apply / reset filter\nEsc          cancel\n\nPress any key to close.",
                )
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" Help ")),
                area,
            );
        }
        Mode::Input { action, value } => {
            let (title, hint) = match action {
                InputAction::Tag => (" Add tag ", "tag for selected image"),
                InputAction::Collection => (
                    " Manage collections ",
                    "NAME/save NAME, load NAME, delete NAME, or list",
                ),
                InputAction::Move => (" Move preview ", "destination directory"),
                InputAction::Bind => (
                    " Manage wpaperd binding ",
                    "DISPLAY COLLECTION, or: unbind DISPLAY",
                ),
            };
            let area = centered_rect(70, 22, frame.area());
            frame.render_widget(Clear, area);
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(hint),
                    Line::from(""),
                    Line::from(Span::styled(
                        format!("> {value}█"),
                        Style::default().fg(Color::Cyan),
                    )),
                    Line::from(""),
                    Line::from("Enter applies • Esc cancels"),
                ])
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
    let focus_help = match editor.focus {
        FilterEditorFocus::Document => {
            "JSON: arrows move • Enter inserts a line • Tab selects presets"
        }
        FilterEditorFocus::Presets => {
            "Presets: ↑/↓ choose • Enter loads • s saves • Tab edits JSON"
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

    if editor.focus == FilterEditorFocus::Document
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

    use crate::db::ImageStatus;

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
            mode,
            status: "Ready — ? for help".into(),
            picker: Picker::halfblocks(),
            preview: None,
            preview_id: None,
            scan_receiver: None,
            scan_started: None,
            scan_total: None,
            command_receiver: None,
            command_started: None,
            running_command: None,
            command_history: Vec::new(),
            pending_command_output: None,
            should_quit: false,
        }
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
