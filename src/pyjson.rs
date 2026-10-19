//! Python-compatible JSON rendering for prompt construction.
//!
//! The Python reference builds state text with `json.dumps(value, ensure_ascii=False)`,
//! whose default separators are `", "` and `": "` and whose dictionaries preserve
//! insertion order. `serde_json`'s default serializer is compact and, without the
//! `preserve_order` feature, sorts object keys. Both would change the prompt and thus
//! the model distribution. This module renders values byte-for-byte like CPython's
//! `json.dumps(..., ensure_ascii=False)`.

use serde_json::Value;

/// Render a JSON value the way CPython's `json.dumps(value, ensure_ascii=False)` does.
pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => {
            if number.is_f64() {
                out.push_str(&format_float(number.as_f64().unwrap_or(0.0)));
            } else {
                out.push_str(&number.to_string());
            }
        }
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(key, out);
                out.push_str(": ");
                write_value(item, out);
            }
            out.push('}');
        }
    }
}

fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Render a finite f64 with CPython's `repr` notation rules.
///
/// Python uses scientific notation when the decimal exponent is below -4 or at least 16,
/// always keeps a `.0` on integral fixed-notation values, and pads the scientific exponent
/// to two digits. Rust/ryu uses the same shortest round-trip digits but different notation,
/// so the exponent is normalised here.
pub fn format_float(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_owned();
    }
    if !value.is_finite() {
        // serde_json cannot represent these in a parsed document, but be explicit.
        return if value.is_nan() {
            "NaN".to_owned()
        } else if value.is_sign_negative() {
            "-Infinity".to_owned()
        } else {
            "Infinity".to_owned()
        };
    }

    // Rust's Debug formatting for f64 is shortest-round-trip (the same digit sequence
    // CPython uses) in either fixed or scientific notation.
    let debug = format!("{value:?}");
    let (mantissa, exponent) = match debug.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().unwrap_or(0)),
        None => (debug.as_str(), 0),
    };
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };

    let (integer, fraction) = match mantissa.split_once('.') {
        Some((integer, fraction)) => (integer, fraction),
        None => (mantissa, ""),
    };
    let raw_digits = format!("{integer}{fraction}");
    let leading_zeros = raw_digits.bytes().take_while(|byte| *byte == b'0').count();
    let digits = &raw_digits[leading_zeros..];
    if digits.is_empty() {
        return if sign == "-" { "-0.0" } else { "0.0" }.to_owned();
    }

    let decimal_exponent = exponent - fraction.len() as i32 + digits.len() as i32 - 1;

    let mut out = String::with_capacity(debug.len() + 4);
    out.push_str(sign);
    if !(-4..16).contains(&decimal_exponent) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        if decimal_exponent >= 0 {
            out.push('+');
        } else {
            out.push('-');
        }
        out.push_str(&format!("{:02}", decimal_exponent.abs()));
    } else {
        let point = decimal_exponent + 1;
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..(-point) {
                out.push('0');
            }
            out.push_str(digits);
        } else if (point as usize) >= digits.len() {
            out.push_str(digits);
            for _ in 0..(point as usize - digits.len()) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    }
    out
}

/// Python `str()` for the values that can appear in an OpenAI decisions body.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => py_repr(other),
    }
}

/// Python `repr()` for JSON-compatible values (insertion order preserved).
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) if number.is_f64() => format_float(number.as_f64().unwrap_or(0.0)),
        Value::Number(number) => number.to_string(),
        Value::String(text) => {
            let mut out = String::with_capacity(text.len() + 2);
            out.push('\'');
            for character in text.chars() {
                match character {
                    '\\' => out.push_str("\\\\"),
                    '\'' => out.push_str("\\'"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c => out.push(c),
                }
            }
            out.push('\'');
            out
        }
        Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", rendered.join(", "))
        }
        Value::Object(map) => {
            let rendered: Vec<String> = map
                .iter()
                .map(|(key, item)| {
                    format!(
                        "{}: {}",
                        py_repr(&Value::String(key.clone())),
                        py_repr(item)
                    )
                })
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
    }
}

/// Python truthiness for JSON values.
pub fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(boolean) => *boolean,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{dumps, format_float, py_repr, py_str, py_truthy};

    #[test]
    fn renders_like_python_with_order_spacing_and_unicode() {
        // Insertion order is deliberate: z before a, and 日本語 must stay raw.
        let value: serde_json::Value = serde_json::from_str(
            r#"{"z": 1, "a": [true, null, "日本語"], "nested": {"b": "\n", "a": 2.5}}"#,
        )
        .unwrap();
        assert_eq!(
            dumps(&value),
            r#"{"z": 1, "a": [true, null, "日本語"], "nested": {"b": "\n", "a": 2.5}}"#
        );
    }

    #[test]
    fn escapes_control_characters_and_quotes() {
        assert_eq!(dumps(&json!("a\"b\\c\n\u{0001}")), r#""a\"b\\c\n\u0001""#);
    }

    #[test]
    fn formats_floats_like_python() {
        assert_eq!(format_float(1.0), "1.0");
        assert_eq!(format_float(-0.0), "-0.0");
        assert_eq!(format_float(0.1), "0.1");
        assert_eq!(format_float(100.0), "100.0");
        assert_eq!(format_float(1e20), "1e+20");
        assert_eq!(format_float(1.5e20), "1.5e+20");
        assert_eq!(format_float(1e-7), "1e-07");
        assert_eq!(format_float(0.0001), "0.0001");
        assert_eq!(format_float(1234567890123456.0), "1234567890123456.0");
        assert_eq!(format_float(1e16), "1e+16");
        assert_eq!(format_float(1e15), "1000000000000000.0");
        assert_eq!(format_float(1.234e-5), "1.234e-05");
        assert_eq!(format_float(0.0001234), "0.0001234");
        assert_eq!(format_float(123.456), "123.456");
        assert_eq!(format_float(9_007_199_254_740_992.0), "9007199254740992.0");
        assert_eq!(format_float(-2.5e-8), "-2.5e-08");
        assert_eq!(
            format_float(1.7976931348623157e308),
            "1.7976931348623157e+308"
        );
        assert_eq!(format_float(5e-324), "5e-324");
    }

    #[test]
    fn python_str_and_truthiness() {
        assert_eq!(py_str(&json!("x")), "x");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_str(&json!({"a": [1, "b"]})), "{'a': [1, 'b']}");
        assert_eq!(py_repr(&json!("it's")), r"'it\'s'");
        assert!(py_truthy(&json!("x")));
        assert!(!py_truthy(&json!("")));
        assert!(!py_truthy(&json!(0)));
        assert!(!py_truthy(&json!([])));
        assert!(py_truthy(&json!([0])));
    }
}
