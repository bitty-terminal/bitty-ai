//! Minimal hand-rolled JSON helpers for newline-delimited JSON-RPC.
//!
//! `pub(crate)` only: the crate has no JSON dependency by design (zero new
//! external dependencies), so this module provides exactly the narrow shapes
//! the client needs — string escaping for outbound frames and scoped field
//! extraction for inbound frames. All parsing is bounded by the caller
//! (frames never exceed [`crate::frame::MAX_FRAME_BYTES`]) and operates on
//! `&str` slices without allocation except for unescaped output strings.

/// Append `value` to `out` with JSON string escaping (`"`, `\`, and
/// control bytes below `0x20` as `\u00XX`; named escapes for the common
/// whitespace controls).
pub fn escape_into(out: &mut String, value: &str) {
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            character if (character as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => out.push(character),
        }
    }
}

/// JSON-escape `value` into a new string.
#[must_use]
pub fn escaped(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    escape_into(&mut out, value);
    out
}

/// Find the byte offset of the value for a top-level `"key":` in `line`,
/// skipping whitespace between the colon and the value.
///
/// Tracks `{}`/`[]` nesting depth and string literals so keys inside nested
/// objects or arrays (`depth > 1`) and `"key":` substrings inside string
/// values never shadow top-level JSON object members.
fn value_offset(line: &str, key: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut index = 0_usize;
    let mut depth = 0_usize;
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                index += 1;
            }
            b'"' if depth == 1 => {
                let mut after = skip_json_string(bytes, index)?;
                while after < bytes.len() && bytes[after].is_ascii_whitespace() {
                    after += 1;
                }
                if after < bytes.len() && bytes[after] == b':' {
                    let mut val_start = after + 1;
                    while val_start < bytes.len() && bytes[val_start].is_ascii_whitespace() {
                        val_start += 1;
                    }
                    if let Some((parsed_key, _)) = parse_json_string(line, index)
                        && parsed_key == key
                    {
                        return (val_start <= bytes.len()).then_some(val_start);
                    }
                    index = val_start;
                } else {
                    index = after;
                }
            }
            b'"' => {
                index = skip_json_string(bytes, index)?;
            }
            _ => {
                index += 1;
            }
        }
    }
    None
}

fn skip_json_string(bytes: &[u8], quote_offset: usize) -> Option<usize> {
    if bytes.get(quote_offset) != Some(&b'"') {
        return None;
    }
    let mut index = quote_offset + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'"' => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}

fn parse_json_string(line: &str, quote_offset: usize) -> Option<(String, usize)> {
    let bytes = line.as_bytes();
    if bytes.get(quote_offset) != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut index = quote_offset + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => return Some((out, index + 1)),
            b'\\' => {
                index += 1;
                if index >= bytes.len() {
                    return None;
                }
                match bytes[index] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{08}'),
                    b'f' => out.push('\u{0C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if index + 4 >= bytes.len() {
                            return None;
                        }
                        let digits = line.get(index + 1..index + 5)?;
                        let code = u16::from_str_radix(digits, 16).ok()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            if index + 10 >= bytes.len()
                                || bytes[index + 5] != b'\\'
                                || bytes[index + 6] != b'u'
                            {
                                return None;
                            }
                            let low_digits = line.get(index + 7..index + 11)?;
                            let low = u16::from_str_radix(low_digits, 16).ok()?;
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return None;
                            }
                            let scalar = 0x10000
                                + (((u32::from(code) - 0xD800) << 10) | (u32::from(low) - 0xDC00));
                            out.push(char::from_u32(scalar)?);
                            index += 10;
                        } else if (0xDC00..=0xDFFF).contains(&code) {
                            return None;
                        } else {
                            out.push(char::from_u32(u32::from(code))?);
                            index += 4;
                        }
                    }
                    _ => return None,
                }
                index += 1;
            }
            _ => {
                let rest = line.get(index..)?;
                let character = rest.chars().next()?;
                out.push(character);
                index += character.len_utf8();
            }
        }
    }
    None
}

/// Extract a JSON string field, unescaping it. Returns `None` when the key
/// is absent or its value is not a string.
#[must_use]
pub fn find_string_field(line: &str, key: &str) -> Option<String> {
    let offset = value_offset(line, key)?;
    parse_json_string(line, offset).map(|(value, _)| value)
}

/// Extract the raw JSON token for a key: for a string value the inner text
/// (still escaped); for numbers, `true`/`false`/`null` the literal; for
/// objects/arrays the balanced substring. Returns `None` when absent.
#[must_use]
pub fn find_raw_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let offset = value_offset(line, key)?;
    let bytes = line.as_bytes();
    if offset >= bytes.len() {
        return None;
    }
    match bytes[offset] {
        b'"' => {
            let mut index = offset + 1;
            while index < bytes.len() {
                match bytes[index] {
                    b'\\' => index += 2,
                    b'"' => return Some(&line[offset + 1..index]),
                    _ => index += 1,
                }
                if index == offset + 1 {
                    break;
                }
            }
            None
        }
        b'{' | b'[' => {
            let open = bytes[offset];
            let close = if open == b'{' { b'}' } else { b']' };
            let end = balanced_end(line, offset, open, close)?;
            Some(&line[offset..=end])
        }
        _ => {
            let mut end = offset;
            while end < bytes.len() && !matches!(bytes[end], b',' | b'}' | b']') {
                end += 1;
            }
            let token = line[offset..end].trim();
            if token.is_empty() { None } else { Some(token) }
        }
    }
}

/// Byte index of the closing delimiter matching `line[start]`, respecting
/// strings and escapes. Returns `None` when unbalanced.
#[must_use]
pub fn balanced_end(line: &str, start: usize, open: u8, close: u8) -> Option<usize> {
    let bytes = line.as_bytes();
    if bytes.get(start) != Some(&open) {
        return None;
    }
    let mut depth = 0_usize;
    let mut index = start;
    let mut in_string = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            match byte {
                b'\\' => index += 1,
                b'"' => in_string = false,
                _ => {}
            }
        } else if byte == b'"' {
            in_string = true;
        } else if byte == open {
            depth += 1;
        } else if byte == close {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

/// Extract a balanced JSON object field for `key` (including braces).
#[must_use]
pub fn find_object_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let raw = find_raw_field(line, key)?;
    if raw.starts_with('{') {
        Some(raw)
    } else {
        None
    }
}

/// Extract a JSON boolean field. Returns `None` when absent or not a bool.
#[must_use]
pub fn find_bool_field(line: &str, key: &str) -> Option<bool> {
    match find_raw_field(line, key)? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Whether `line` looks like a JSON-RPC request or notification from the
/// server (carries a `method` string member).
#[must_use]
pub fn has_method(line: &str) -> bool {
    find_string_field(line, "method").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_round_trips_controls() {
        assert_eq!(escaped("a\"b\\c\nd\te"), "a\\\"b\\\\c\\nd\\te");
        assert_eq!(escaped("plain"), "plain");
        assert_eq!(escaped("tab\there"), "tab\\there");
    }

    #[test]
    fn string_field_parses_and_unescapes() {
        let line = r#"{"jsonrpc":"2.0","method":"roots/list","note":"a\"b"}"#;
        assert_eq!(
            find_string_field(line, "method"),
            Some("roots/list".to_owned())
        );
        assert_eq!(find_string_field(line, "note"), Some("a\"b".to_owned()));
        assert_eq!(find_string_field(line, "missing"), None);
        // Non-string values do not parse as strings.
        assert_eq!(find_string_field(r#"{"id":12}"#, "id"), None);
    }

    #[test]
    fn key_inside_a_value_does_not_match() {
        let line = r#"{"description":"mentions \"tools\":"}"#;
        assert_eq!(find_string_field(line, "tools"), None);
    }

    #[test]
    fn raw_field_echoes_ids_verbatim() {
        assert_eq!(find_raw_field(r#"{"id":12,"x":1}"#, "id"), Some("12"));
        assert_eq!(find_raw_field(r#"{"id":"abc","x":1}"#, "id"), Some("abc"));
        assert_eq!(
            find_bool_field(r#"{"isError":true}"#, "isError"),
            Some(true)
        );
    }

    #[test]
    fn object_field_extracts_balanced_braces() {
        let line = r#"{"result":{"tools":[],"nextCursor":"a}b"},"id":1}"#;
        let object = find_object_field(line, "result").expect("object");
        assert!(object.contains("\"tools\""));
        assert_eq!(
            find_string_field(object, "nextCursor"),
            Some("a}b".to_owned())
        );
        assert_eq!(find_object_field(line, "id"), None);
    }

    #[test]
    fn nested_keys_do_not_shadow_top_level_fields() {
        let response = r#"{"jsonrpc":"2.0","id":7,"result":{"structuredContent":{"method":"GET","error":{"code":-32000},"id":999}}}"#;
        assert!(!has_method(response));
        assert_eq!(find_raw_field(response, "id"), Some("7"));
        assert_eq!(find_object_field(response, "error"), None);

        let tool_obj = r#"{"inputSchema":{"type":"object","properties":{"name":{"type":"string"}}},"name":"read_file"}"#;
        assert_eq!(
            find_string_field(tool_obj, "name"),
            Some("read_file".to_owned())
        );
    }

    #[test]
    fn string_field_unescapes_utf16_surrogate_pairs_and_rejects_lone_surrogates() {
        let valid = r#"{"text":"smile \uD83D\uDE00 done"}"#;
        assert_eq!(
            find_string_field(valid, "text"),
            Some("smile 😀 done".to_owned())
        );

        let lone_high = r#"{"text":"\uD83Dabc"}"#;
        assert_eq!(find_string_field(lone_high, "text"), None);

        let lone_low = r#"{"text":"\uDE00abc"}"#;
        assert_eq!(find_string_field(lone_low, "text"), None);

        // Unrelated string values containing lone surrogates do not abort lookup of later fields.
        let unrelated_lone = r#"{"note":"\uD800","text":"kept"}"#;
        assert_eq!(
            find_string_field(unrelated_lone, "text"),
            Some("kept".to_owned())
        );
    }
}
