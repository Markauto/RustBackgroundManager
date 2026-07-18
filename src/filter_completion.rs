#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FilterJsonCompletion {
    pub label: String,
    pub replacement: String,
    pub description: String,
    pub replace_start: usize,
    pub replace_end: usize,
    pub cursor_after: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObjectKind {
    Root,
    Colour,
    AiLabel,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObjectPhase {
    KeyOrEnd,
    Colon,
    Value,
    CommaOrEnd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArrayPhase {
    ValueOrEnd,
    CommaOrEnd,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Container {
    Object {
        kind: ObjectKind,
        phase: ObjectPhase,
        current_key: Option<String>,
    },
    Array {
        owner_field: Option<String>,
        phase: ArrayPhase,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum StringRole {
    Key(ObjectKind),
    Value(Option<String>),
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ActiveString {
    start: usize,
    value: String,
    role: StringRole,
    escaped: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CursorSlot {
    Key {
        kind: ObjectKind,
        quoted: bool,
        prefix: String,
        replace_start: usize,
        replace_end: usize,
    },
    Value {
        field: String,
        quoted: bool,
        prefix: String,
        replace_start: usize,
        replace_end: usize,
    },
}

#[derive(Clone, Copy)]
struct FieldSpec {
    name: &'static str,
    default: &'static str,
    description: &'static str,
}

const ROOT_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "version",
        default: "1",
        description: "Filter schema version; must be 1",
    },
    FieldSpec {
        name: "source_ids",
        default: "[]",
        description: "Source IDs; repeated values are ORed",
    },
    FieldSpec {
        name: "paths",
        default: "[]",
        description: "Case-insensitive path fragments",
    },
    FieldSpec {
        name: "min_width",
        default: "null",
        description: "Minimum image width in pixels or null",
    },
    FieldSpec {
        name: "max_width",
        default: "null",
        description: "Maximum image width in pixels or null",
    },
    FieldSpec {
        name: "min_height",
        default: "null",
        description: "Minimum image height in pixels or null",
    },
    FieldSpec {
        name: "max_height",
        default: "null",
        description: "Maximum image height in pixels or null",
    },
    FieldSpec {
        name: "orientations",
        default: "[]",
        description: "landscape, portrait, or square",
    },
    FieldSpec {
        name: "aspect_ratios",
        default: "[]",
        description: "Positive decimal aspect ratios",
    },
    FieldSpec {
        name: "aspect_tolerance",
        default: "0.03",
        description: "Ratio tolerance from 0 to 1",
    },
    FieldSpec {
        name: "light_dark",
        default: "[]",
        description: "light or dark analysis classes",
    },
    FieldSpec {
        name: "min_luminance",
        default: "null",
        description: "Minimum luminance from 0 to 1 or null",
    },
    FieldSpec {
        name: "max_luminance",
        default: "null",
        description: "Maximum luminance from 0 to 1 or null",
    },
    FieldSpec {
        name: "dominant_colours",
        default: "[]",
        description: "Dominant colour and Oklab distance filters",
    },
    FieldSpec {
        name: "palette_colours",
        default: "[]",
        description: "Palette colour and Oklab distance filters",
    },
    FieldSpec {
        name: "ai_labels",
        default: "[]",
        description: "CLIP label pack, label, and score filters",
    },
    FieldSpec {
        name: "semantic_text",
        default: "null",
        description: "Free-text CLIP query or null",
    },
    FieldSpec {
        name: "semantic_min_score",
        default: "null",
        description: "Minimum semantic score from -1 to 1",
    },
    FieldSpec {
        name: "tags",
        default: "[]",
        description: "Custom tags; repeated values are ORed",
    },
    FieldSpec {
        name: "favorite",
        default: "null",
        description: "true, false, or null for either",
    },
];

const COLOUR_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "hex",
        default: "\"#D08040\"",
        description: "Six-digit sRGB colour",
    },
    FieldSpec {
        name: "max_distance",
        default: "0.08",
        description: "Maximum Oklab colour distance",
    },
];

const AI_LABEL_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "pack",
        default: "\"mood\"",
        description: "Label-pack name",
    },
    FieldSpec {
        name: "label",
        default: "\"calm\"",
        description: "Label name in the selected pack",
    },
    FieldSpec {
        name: "min_score",
        default: "0.5",
        description: "Minimum CLIP estimate from 0 to 1",
    },
];

pub(crate) fn filter_json_completions(text: &str, cursor: usize) -> Vec<FilterJsonCompletion> {
    let cursor = cursor.min(text.chars().count());
    let Some(slot) = scan_cursor_slot(text, cursor) else {
        return Vec::new();
    };
    let sort_prefix = match &slot {
        CursorSlot::Key { prefix, .. } | CursorSlot::Value { prefix, .. } => {
            prefix.to_ascii_lowercase()
        }
    };
    let mut completions = match slot {
        CursorSlot::Key {
            kind,
            quoted,
            prefix,
            replace_start,
            replace_end,
        } => key_completions(kind, quoted, &prefix, replace_start, replace_end),
        CursorSlot::Value {
            field,
            quoted,
            prefix,
            replace_start,
            replace_end,
        } => value_completions(&field, quoted, &prefix, replace_start, replace_end),
    };
    completions.sort_by_key(|completion| {
        let label = completion.label.to_ascii_lowercase();
        (u8::from(!label.starts_with(&sort_prefix)), label)
    });
    completions
}

fn scan_cursor_slot(text: &str, cursor: usize) -> Option<CursorSlot> {
    let mut stack = Vec::new();
    let mut active_string: Option<ActiveString> = None;

    for (index, character) in text.chars().enumerate().take(cursor) {
        if let Some(active) = &mut active_string {
            if active.escaped {
                active.value.push(character);
                active.escaped = false;
            } else if character == '\\' {
                active.escaped = true;
            } else if character == '"' {
                let completed = active_string.take().expect("active string exists");
                finish_string(&mut stack, completed.role, completed.value);
            } else {
                active.value.push(character);
            }
            continue;
        }

        match character {
            '"' => {
                active_string = Some(ActiveString {
                    start: index,
                    value: String::new(),
                    role: string_role(&stack),
                    escaped: false,
                });
            }
            '{' => {
                let owner = begin_container_value(&mut stack);
                let kind = if stack.is_empty() {
                    ObjectKind::Root
                } else {
                    object_kind_for_owner(owner.as_deref())
                };
                stack.push(Container::Object {
                    kind,
                    phase: ObjectPhase::KeyOrEnd,
                    current_key: None,
                });
            }
            '[' => {
                let owner_field = begin_container_value(&mut stack);
                stack.push(Container::Array {
                    owner_field,
                    phase: ArrayPhase::ValueOrEnd,
                });
            }
            '}' | ']' => {
                stack.pop();
            }
            ':' => {
                if let Some(Container::Object { phase, .. }) = stack.last_mut()
                    && *phase == ObjectPhase::Colon
                {
                    *phase = ObjectPhase::Value;
                }
            }
            ',' => match stack.last_mut() {
                Some(Container::Object {
                    phase, current_key, ..
                }) => {
                    *phase = ObjectPhase::KeyOrEnd;
                    *current_key = None;
                }
                Some(Container::Array { phase, .. }) => *phase = ArrayPhase::ValueOrEnd,
                None => {}
            },
            _ => {}
        }
    }

    if let Some(active) = active_string {
        let replace_end = find_closing_quote(text, cursor, active.escaped);
        return match active.role {
            StringRole::Key(kind) => Some(CursorSlot::Key {
                kind,
                quoted: true,
                prefix: active.value,
                replace_start: active.start + 1,
                replace_end,
            }),
            StringRole::Value(Some(field)) => Some(CursorSlot::Value {
                field,
                quoted: true,
                prefix: active.value,
                replace_start: active.start + 1,
                replace_end,
            }),
            StringRole::Value(None) | StringRole::Unknown => None,
        };
    }

    let (replace_start, replace_end, prefix) = token_at_cursor(text, cursor);
    match stack.last()? {
        Container::Object {
            kind,
            phase: ObjectPhase::KeyOrEnd,
            ..
        } => Some(CursorSlot::Key {
            kind: *kind,
            quoted: false,
            prefix,
            replace_start,
            replace_end,
        }),
        Container::Object {
            phase: ObjectPhase::Value,
            current_key: Some(field),
            ..
        }
        | Container::Array {
            owner_field: Some(field),
            phase: ArrayPhase::ValueOrEnd,
        } => Some(CursorSlot::Value {
            field: field.clone(),
            quoted: false,
            prefix,
            replace_start,
            replace_end,
        }),
        _ => None,
    }
}

fn string_role(stack: &[Container]) -> StringRole {
    match stack.last() {
        Some(Container::Object {
            kind,
            phase: ObjectPhase::KeyOrEnd,
            ..
        }) => StringRole::Key(*kind),
        Some(Container::Object {
            phase: ObjectPhase::Value,
            current_key,
            ..
        }) => StringRole::Value(current_key.clone()),
        Some(Container::Array {
            owner_field,
            phase: ArrayPhase::ValueOrEnd,
        }) => StringRole::Value(owner_field.clone()),
        _ => StringRole::Unknown,
    }
}

fn finish_string(stack: &mut [Container], role: StringRole, value: String) {
    match role {
        StringRole::Key(_) => {
            if let Some(Container::Object {
                phase, current_key, ..
            }) = stack.last_mut()
            {
                *current_key = Some(value);
                *phase = ObjectPhase::Colon;
            }
        }
        StringRole::Value(_) => finish_scalar_value(stack),
        StringRole::Unknown => {}
    }
}

fn begin_container_value(stack: &mut [Container]) -> Option<String> {
    match stack.last_mut() {
        Some(Container::Object {
            phase, current_key, ..
        }) if *phase == ObjectPhase::Value => {
            *phase = ObjectPhase::CommaOrEnd;
            current_key.clone()
        }
        Some(Container::Array { owner_field, phase }) if *phase == ArrayPhase::ValueOrEnd => {
            *phase = ArrayPhase::CommaOrEnd;
            owner_field.clone()
        }
        _ => None,
    }
}

fn finish_scalar_value(stack: &mut [Container]) {
    match stack.last_mut() {
        Some(Container::Object { phase, .. }) if *phase == ObjectPhase::Value => {
            *phase = ObjectPhase::CommaOrEnd;
        }
        Some(Container::Array { phase, .. }) if *phase == ArrayPhase::ValueOrEnd => {
            *phase = ArrayPhase::CommaOrEnd;
        }
        _ => {}
    }
}

fn object_kind_for_owner(owner: Option<&str>) -> ObjectKind {
    match owner {
        Some("dominant_colours" | "palette_colours") => ObjectKind::Colour,
        Some("ai_labels") => ObjectKind::AiLabel,
        _ => ObjectKind::Unknown,
    }
}

fn find_closing_quote(text: &str, cursor: usize, mut escaped: bool) -> usize {
    for (index, character) in text.chars().enumerate().skip(cursor) {
        if escaped {
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == '"' {
            return index;
        }
    }
    cursor
}

fn token_at_cursor(text: &str, cursor: usize) -> (usize, usize, String) {
    let characters = text.chars().collect::<Vec<_>>();
    let mut start = cursor;
    while start > 0 && !is_token_delimiter(characters[start - 1]) {
        start -= 1;
    }
    let mut end = cursor;
    while end < characters.len() && !is_token_delimiter(characters[end]) {
        end += 1;
    }
    (start, end, characters[start..cursor].iter().collect())
}

fn is_token_delimiter(character: char) -> bool {
    character.is_whitespace() || matches!(character, ',' | ':' | '[' | ']' | '{' | '}' | '"')
}

fn key_completions(
    kind: ObjectKind,
    quoted: bool,
    prefix: &str,
    replace_start: usize,
    replace_end: usize,
) -> Vec<FilterJsonCompletion> {
    let fields = match kind {
        ObjectKind::Root => ROOT_FIELDS,
        ObjectKind::Colour => COLOUR_FIELDS,
        ObjectKind::AiLabel => AI_LABEL_FIELDS,
        ObjectKind::Unknown => return Vec::new(),
    };
    fields
        .iter()
        .filter(|field| matches_prefix(field.name, prefix))
        .map(|field| {
            let replacement = if quoted {
                field.name.to_owned()
            } else {
                format!("\"{}\": {}", field.name, field.default)
            };
            FilterJsonCompletion {
                label: field.name.into(),
                cursor_after: replacement.chars().count(),
                replacement,
                description: field.description.into(),
                replace_start,
                replace_end,
            }
        })
        .collect()
}

fn value_completions(
    field: &str,
    quoted: bool,
    prefix: &str,
    replace_start: usize,
    replace_end: usize,
) -> Vec<FilterJsonCompletion> {
    let mut candidates = Vec::new();
    let mut string = |value: &str, description: &str| {
        let replacement = if quoted {
            value.to_owned()
        } else {
            format!("\"{value}\"")
        };
        push_value(
            &mut candidates,
            value,
            replacement,
            description,
            prefix,
            replace_start,
            replace_end,
            None,
        );
    };
    if quoted {
        match field {
            "orientations" => {
                string("landscape", "Width is greater than height");
                string("portrait", "Height is greater than width");
                string("square", "Width and height are approximately equal");
            }
            "light_dark" => {
                string("light", "Luminance is at or above the configured threshold");
                string("dark", "Luminance is below the configured threshold");
            }
            "paths" => {
                string("mountain", "Case-insensitive path fragment");
                string("/Backgrounds/", "Case-insensitive path fragment");
            }
            "semantic_text" => string(
                "misty mountains at night",
                "Example free-text semantic query",
            ),
            "tags" => {
                string("desktop", "Example custom tag");
                string("archive", "Example custom tag");
            }
            "hex" => {
                string("#D08040", "Warm orange");
                string("#203060", "Dark blue");
            }
            "pack" => {
                string("mood", "Seeded mood label pack");
                string("subject", "Seeded subject label pack");
                string("style", "Seeded style label pack");
            }
            "label" => {
                string("calm", "Example mood label");
                string("nature", "Example subject label");
                string("digital art", "Example style label");
            }
            _ => {}
        }
        return candidates;
    }

    let mut token = |value: &str, description: &str| {
        push_value(
            &mut candidates,
            value,
            value.to_owned(),
            description,
            prefix,
            replace_start,
            replace_end,
            None,
        );
    };
    match field {
        "version" => token("1", "Filter schema version"),
        "source_ids" => token("1", "Example registered source ID"),
        "paths" => string_value_candidates(
            &mut candidates,
            prefix,
            replace_start,
            replace_end,
            &[
                ("mountain", "Case-insensitive path fragment"),
                ("/Backgrounds/", "Case-insensitive path fragment"),
            ],
        ),
        "min_width" | "max_width" => {
            token("null", "Disable this width bound");
            token("1920", "Full HD width");
            token("2560", "QHD width");
            token("3840", "4K UHD width");
        }
        "min_height" | "max_height" => {
            token("null", "Disable this height bound");
            token("1080", "Full HD height");
            token("1440", "QHD height");
            token("2160", "4K UHD height");
        }
        "orientations" => string_value_candidates(
            &mut candidates,
            prefix,
            replace_start,
            replace_end,
            &[
                ("landscape", "Width is greater than height"),
                ("portrait", "Height is greater than width"),
                ("square", "Width and height are approximately equal"),
            ],
        ),
        "aspect_ratios" => {
            token("1.7777777778", "16:9");
            token("1.6", "16:10");
            token("2.3333333333", "21:9");
        }
        "aspect_tolerance" => {
            token("0.03", "Default 3% ratio tolerance");
            token("0.05", "5% ratio tolerance");
        }
        "light_dark" => string_value_candidates(
            &mut candidates,
            prefix,
            replace_start,
            replace_end,
            &[
                ("light", "At or above the luminance threshold"),
                ("dark", "Below the luminance threshold"),
            ],
        ),
        "min_luminance" | "max_luminance" => {
            token("null", "Disable this luminance bound");
            token("0.25", "Quarter luminance");
            token("0.5", "Half luminance");
            token("0.75", "Three-quarter luminance");
        }
        "dominant_colours" | "palette_colours" => token(
            r##"{"hex":"#D08040","max_distance":0.08}"##,
            "Colour-filter object",
        ),
        "ai_labels" => token(
            r#"{"pack":"mood","label":"calm","min_score":0.5}"#,
            "AI-label filter object",
        ),
        "semantic_text" => {
            token("null", "Disable semantic filtering");
            push_value(
                &mut candidates,
                "misty mountains at night",
                "\"misty mountains at night\"".into(),
                "Example free-text semantic query",
                prefix,
                replace_start,
                replace_end,
                None,
            );
        }
        "semantic_min_score" => {
            token("null", "Use the default semantic threshold");
            token("0.2", "Example minimum cosine score");
            token("0.5", "Stricter minimum cosine score");
        }
        "tags" => string_value_candidates(
            &mut candidates,
            prefix,
            replace_start,
            replace_end,
            &[
                ("desktop", "Example custom tag"),
                ("archive", "Example custom tag"),
            ],
        ),
        "favorite" => {
            token("true", "Only favorite images");
            token("false", "Only non-favorite images");
            token("null", "Include either favorite state");
        }
        "hex" => string_value_candidates(
            &mut candidates,
            prefix,
            replace_start,
            replace_end,
            &[("#D08040", "Warm orange"), ("#203060", "Dark blue")],
        ),
        "max_distance" => {
            token("0.08", "Default Oklab colour distance");
            token("0.10", "Broader Oklab colour distance");
        }
        "pack" => string_value_candidates(
            &mut candidates,
            prefix,
            replace_start,
            replace_end,
            &[
                ("mood", "Seeded mood label pack"),
                ("subject", "Seeded subject label pack"),
                ("style", "Seeded style label pack"),
            ],
        ),
        "label" => string_value_candidates(
            &mut candidates,
            prefix,
            replace_start,
            replace_end,
            &[
                ("calm", "Example mood label"),
                ("nature", "Example subject label"),
                ("digital art", "Example style label"),
            ],
        ),
        "min_score" => {
            token("0.5", "50% minimum estimate");
            token("0.7", "70% minimum estimate");
        }
        _ => {}
    }
    candidates
}

#[allow(clippy::too_many_arguments)]
fn push_value(
    completions: &mut Vec<FilterJsonCompletion>,
    label: &str,
    replacement: String,
    description: &str,
    prefix: &str,
    replace_start: usize,
    replace_end: usize,
    cursor_after: Option<usize>,
) {
    if matches_prefix(label, prefix) || matches_prefix(&replacement, prefix) {
        completions.push(FilterJsonCompletion {
            label: label.into(),
            cursor_after: cursor_after.unwrap_or_else(|| replacement.chars().count()),
            replacement,
            description: description.into(),
            replace_start,
            replace_end,
        });
    }
}

fn string_value_candidates(
    completions: &mut Vec<FilterJsonCompletion>,
    prefix: &str,
    replace_start: usize,
    replace_end: usize,
    values: &[(&str, &str)],
) {
    for (value, description) in values {
        push_value(
            completions,
            value,
            format!("\"{value}\""),
            description,
            prefix,
            replace_start,
            replace_end,
            None,
        );
    }
}

fn matches_prefix(value: &str, prefix: &str) -> bool {
    let value = value.to_ascii_lowercase();
    let prefix = prefix.to_ascii_lowercase();
    prefix.is_empty() || value.starts_with(&prefix)
}

#[cfg(test)]
mod tests {
    use crate::filter::{AiLabelFilter, ColourFilter, FilterSpecV1};

    use super::*;

    fn cursor_after(text: &str, marker: &str) -> usize {
        let byte = text.rfind(marker).expect("marker") + marker.len();
        text[..byte].chars().count()
    }

    fn labels(text: &str, cursor: usize) -> Vec<String> {
        filter_json_completions(text, cursor)
            .into_iter()
            .map(|completion| completion.label)
            .collect()
    }

    #[test]
    fn completes_partial_root_keys_while_json_is_invalid() {
        let text = "{\n  \"ori\": []\n}";
        let completions = filter_json_completions(text, cursor_after(text, "ori"));

        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].label, "orientations");
        assert_eq!(completions[0].replacement, "orientations");
    }

    #[test]
    fn completes_orientation_values_inside_an_empty_array() {
        let text = r#"{"orientations": []}"#;
        let cursor = cursor_after(text, "[");

        assert_eq!(labels(text, cursor), ["landscape", "portrait", "square"]);
    }

    #[test]
    fn completes_partial_quoted_enum_values() {
        let text = r#"{"light_dark": ["da"]}"#;
        let completions = filter_json_completions(text, cursor_after(text, "da"));

        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].label, "dark");
        assert_eq!(completions[0].replacement, "dark");
    }

    #[test]
    fn completes_nested_colour_filter_keys() {
        let text = r#"{"palette_colours": [{"he": ""}]}"#;
        let completions = filter_json_completions(text, cursor_after(text, "he"));

        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].label, "hex");
    }

    #[test]
    fn scalar_completion_replaces_the_whole_current_token() {
        let text = r#"{"favorite": null}"#;
        let completions = filter_json_completions(text, cursor_after(text, "nu"));
        let favorite = completions
            .iter()
            .find(|completion| completion.label == "null")
            .expect("null completion");

        assert_eq!(favorite.replace_end - favorite.replace_start, 4);
    }

    #[test]
    fn completion_fields_cover_the_serialized_filter_schema() {
        fn object_keys(value: &impl serde::Serialize) -> Vec<String> {
            let mut keys = serde_json::to_value(value)
                .expect("serialize")
                .as_object()
                .expect("object")
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            keys.sort();
            keys
        }

        let mut root = ROOT_FIELDS
            .iter()
            .map(|field| field.name.to_owned())
            .collect::<Vec<_>>();
        root.sort();
        assert_eq!(root, object_keys(&FilterSpecV1::default()));

        let mut colour = COLOUR_FIELDS
            .iter()
            .map(|field| field.name.to_owned())
            .collect::<Vec<_>>();
        colour.sort();
        assert_eq!(
            colour,
            object_keys(&ColourFilter {
                hex: "#D08040".into(),
                max_distance: 0.08,
            })
        );

        let mut label = AI_LABEL_FIELDS
            .iter()
            .map(|field| field.name.to_owned())
            .collect::<Vec<_>>();
        label.sort();
        assert_eq!(
            label,
            object_keys(&AiLabelFilter {
                pack: "mood".into(),
                label: "calm".into(),
                min_score: 0.5,
            })
        );
    }

    #[test]
    fn nested_object_templates_are_valid_filter_values() {
        let colour_text = r#"{"palette_colours": []}"#;
        let colour_cursor = cursor_after(colour_text, "[");
        let colour = filter_json_completions(colour_text, colour_cursor)
            .into_iter()
            .next()
            .expect("colour template");
        serde_json::from_str::<ColourFilter>(&colour.replacement).expect("valid colour template");

        let label_text = r#"{"ai_labels": []}"#;
        let label_cursor = cursor_after(label_text, "[");
        let label = filter_json_completions(label_text, label_cursor)
            .into_iter()
            .next()
            .expect("AI template");
        serde_json::from_str::<AiLabelFilter>(&label.replacement).expect("valid AI template");
    }
}
