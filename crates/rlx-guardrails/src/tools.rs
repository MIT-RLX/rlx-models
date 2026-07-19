// RLX models — guardrails: tool-call validation.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Tool definitions, tool calls, and validation against OpenAI-style
//! function-calling schemas.
//!
//! Ported (and de-coupled) from `mesh-llm-guardrails`. Where the source
//! worked directly off raw `serde_json::Value` request contracts, this crate
//! exposes strongly-typed [`ToolDef`] / [`ToolCall`] structs that still keep
//! the dynamic JSON-Schema `parameters`/`arguments` payloads as
//! [`serde_json::Value`], so any OpenAI-compatible caller can round-trip them.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::error::GuardrailError;

/// Reserved tool name the server exposes for plain-text replies
/// (mesh's `_mesh_respond`).
pub const RLX_RESPOND: &str = "_rlx_respond";
/// Reserved tool name the server exposes for structured-output emission
/// (mesh's `_mesh_emit_structured`).
pub const RLX_EMIT_STRUCTURED: &str = "_rlx_emit_structured";

/// Default prefix that marks a tool name as reserved for RLX internals.
pub const RESERVED_TOOL_PREFIX: &str = "_rlx_";

/// The reserved names a caller-supplied tool set must never collide with.
pub const RESERVED_TOOL_NAMES: &[&str] = &[RLX_RESPOND, RLX_EMIT_STRUCTURED];

/// An OpenAI-style function-calling tool definition.
///
/// Mirrors the `{"type":"function","function":{...}}` wire shape. The
/// `parameters` field is a raw JSON-Schema object kept as a
/// [`serde_json::Value`] because it is dynamic per tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    /// Discriminator; defaults to `"function"`.
    #[serde(default = "function_type")]
    pub r#type: String,
    /// The function payload (name + JSON-Schema parameters).
    pub function: FunctionDef,
}

fn function_type() -> String {
    "function".to_string()
}

/// The `function` payload of a [`ToolDef`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Tool name the model must call.
    pub name: String,
    /// Human-readable description (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON-Schema for the tool arguments (dynamic; kept as raw JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

impl ToolDef {
    /// Build a function tool from a name and a JSON-Schema `parameters` value.
    pub fn function(name: impl Into<String>, parameters: Value) -> Self {
        Self {
            r#type: function_type(),
            function: FunctionDef {
                name: name.into(),
                description: None,
                parameters: Some(parameters),
            },
        }
    }

    /// The tool's name.
    pub fn name(&self) -> &str {
        &self.function.name
    }

    /// The tool's JSON-Schema `parameters`, if any.
    pub fn parameters(&self) -> Option<&Value> {
        self.function.parameters.as_ref()
    }
}

/// A tool call emitted by a model, in OpenAI function-calling shape.
///
/// `arguments` follows the OpenAI convention of a JSON *string* on the wire,
/// but this type also accepts an inline object (see [`ToolCall::args_object`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Called tool name.
    pub name: String,
    /// Raw arguments: either a JSON string or an inline object.
    #[serde(default)]
    pub arguments: Value,
}

impl ToolCall {
    /// Construct a tool call from a name and any arguments value.
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    /// Normalize `arguments` to an object map, decoding a JSON string if
    /// needed. Returns `None` for `null`, and an empty map for scalars —
    /// matching the source's `normalize_tool_arguments`.
    pub fn args_object(&self) -> Option<Map<String, Value>> {
        normalize_tool_arguments(&self.arguments)
    }
}

/// `true` when `name` is reserved for RLX internals (matches the default
/// `_rlx_` prefix).
pub fn is_reserved_tool_name(name: &str) -> bool {
    name.starts_with(RESERVED_TOOL_PREFIX)
}

/// `true` when `name` is reserved for the given `prefix`.
pub fn is_reserved_tool_name_with_prefix(name: &str, prefix: &str) -> bool {
    name.starts_with(prefix)
}

/// The `_rlx_respond` tool definition (plain-text reply escape hatch).
pub fn rlx_respond_tool_definition() -> ToolDef {
    ToolDef::function(
        RLX_RESPOND,
        json!({
            "type": "object",
            "properties": { "message": { "type": "string" } },
            "required": ["message"],
            "additionalProperties": false
        }),
    )
}

/// The `_rlx_emit_structured` tool definition for a given parameters schema.
pub fn rlx_emit_structured_tool_definition(parameters: Value) -> ToolDef {
    ToolDef::function(RLX_EMIT_STRUCTURED, parameters)
}

/// Validate a model's [`ToolCall`] against the provided [`ToolDef`]s.
///
/// Checks, in order:
/// 1. the called name is not a reserved RLX name colliding with a
///    caller-supplied tool (unless that tool set genuinely declares it),
/// 2. the called name exists in `tools`,
/// 3. all `required` parameters from the tool's JSON-Schema are present.
///
/// Returns `Ok(())` when the call is well-formed.
pub fn validate_tool_call(call: &ToolCall, tools: &[ToolDef]) -> Result<(), GuardrailError> {
    // A model must never invoke a reserved internal name unless a tool with
    // that exact name was explicitly provided in the catalog.
    let declared = tools.iter().any(|tool| tool.name() == call.name);
    if is_reserved_tool_name(&call.name) && !declared {
        return Err(GuardrailError::ReservedToolName {
            name: call.name.clone(),
        });
    }

    let Some(tool) = tools.iter().find(|tool| tool.name() == call.name) else {
        return Err(GuardrailError::UnknownTool {
            name: call.name.clone(),
        });
    };

    let arguments = call
        .args_object()
        .map(Value::Object)
        .unwrap_or_else(|| json!({}));

    if let Some(parameters) = tool.parameters() {
        ensure_required_arguments(&call.name, &arguments, parameters)?;
    }
    Ok(())
}

/// [`validate_tool_call`] returning an [`anyhow::Result`] for callers (like
/// `rlx-serve`) that thread errors through `anyhow`.
pub fn validate_tool_call_anyhow(call: &ToolCall, tools: &[ToolDef]) -> anyhow::Result<()> {
    validate_tool_call(call, tools).map_err(anyhow::Error::from)
}

/// `true` when a caller-supplied tool set collides with a reserved RLX name.
pub fn tools_collide_with_reserved(tools: &[ToolDef]) -> bool {
    tools
        .iter()
        .any(|tool| RESERVED_TOOL_NAMES.contains(&tool.name()))
}

/// Normalize dynamic arguments to an object map.
///
/// - object          → cloned map
/// - JSON string      → decoded object (or `None` if it is not an object)
/// - `null`           → `None`
/// - any other scalar → empty map
pub fn normalize_tool_arguments(arguments: &Value) -> Option<Map<String, Value>> {
    match arguments {
        Value::Object(arguments) => Some(arguments.clone()),
        Value::String(arguments) => serde_json::from_str::<Value>(arguments)
            .ok()?
            .as_object()
            .cloned(),
        Value::Null => None,
        _ => Some(Map::new()),
    }
}

/// Drop unknown/ill-typed arguments per the tool's JSON-Schema and then
/// verify that all required arguments survive. Returns the cleaned arguments.
///
/// Ported from the source's `sanitize_tool_arguments_for_tool`.
pub fn sanitize_tool_arguments(
    call: &ToolCall,
    tools: &[ToolDef],
) -> Result<Value, GuardrailError> {
    let mut arguments = normalize_tool_arguments(&call.arguments)
        .map(Value::Object)
        .unwrap_or_else(|| json!({}));

    let Some(tool) = tools.iter().find(|tool| tool.name() == call.name) else {
        return Ok(arguments);
    };
    let Some(parameters) = tool.parameters() else {
        return Ok(arguments);
    };

    sanitize_object_for_schema(&mut arguments, parameters);
    ensure_required_arguments(&call.name, &arguments, parameters)?;
    Ok(arguments)
}

/// Serialize `arguments` to the OpenAI wire form: a JSON *string* that is
/// always a valid object (falls back to `"{}"`).
pub fn tool_arguments_wire_string(arguments: &Value) -> String {
    match arguments {
        Value::String(value) => serde_json::from_str::<Value>(value)
            .ok()
            .filter(Value::is_object)
            .map_or_else(|| "{}".to_string(), |_| value.clone()),
        Value::Object(_) => serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
        _ => "{}".to_string(),
    }
}

fn sanitize_object_for_schema(arguments: &mut Value, schema: &Value) {
    let Some(arguments) = arguments.as_object_mut() else {
        return;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    let allow_additional = matches!(schema.get("additionalProperties"), Some(Value::Bool(true)))
        || schema
            .get("additionalProperties")
            .is_some_and(Value::is_object);

    arguments.retain(|key, value| {
        let Some(property_schema) = properties.get(key) else {
            return allow_additional;
        };
        argument_value_matches_schema(value, property_schema)
    });
}

fn argument_value_matches_schema(value: &Value, schema: &Value) -> bool {
    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array)
        && !enum_values.iter().any(|allowed| allowed == value)
    {
        return false;
    }

    let Some(schema_type) = schema.get("type") else {
        return true;
    };
    let types: Vec<&str> = match schema_type {
        Value::String(t) => vec![t.as_str()],
        Value::Array(types) => types.iter().filter_map(Value::as_str).collect(),
        _ => return true,
    };
    types
        .iter()
        .any(|schema_type| value_matches_type(value, schema_type))
}

fn value_matches_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        _ => true,
    }
}

fn ensure_required_arguments(
    tool_name: &str,
    arguments: &Value,
    schema: &Value,
) -> Result<(), GuardrailError> {
    let Some(required) = schema.get("required").and_then(Value::as_array) else {
        return Ok(());
    };
    let Some(arguments) = arguments.as_object() else {
        return Ok(());
    };
    let missing: Vec<String> = required
        .iter()
        .filter_map(Value::as_str)
        .filter(|field| !arguments.contains_key(*field))
        .map(str::to_string)
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(GuardrailError::MissingRequired {
            tool_name: tool_name.to_string(),
            fields: missing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_file_tools() -> Vec<ToolDef> {
        vec![ToolDef::function(
            "read_file",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
                "additionalProperties": false
            }),
        )]
    }

    #[test]
    fn accepts_a_valid_call() {
        let tools = read_file_tools();
        let call = ToolCall::new("read_file", json!({ "path": "README.md" }));
        assert!(validate_tool_call(&call, &tools).is_ok());
    }

    #[test]
    fn accepts_a_valid_call_with_string_arguments() {
        let tools = read_file_tools();
        let call = ToolCall::new("read_file", json!("{\"path\":\"README.md\"}"));
        assert!(validate_tool_call(&call, &tools).is_ok());
    }

    #[test]
    fn rejects_unknown_tool_name() {
        let tools = read_file_tools();
        let call = ToolCall::new("write_file", json!({ "path": "README.md" }));
        assert_eq!(
            validate_tool_call(&call, &tools),
            Err(GuardrailError::UnknownTool {
                name: "write_file".into()
            })
        );
    }

    #[test]
    fn rejects_missing_required_argument() {
        let tools = read_file_tools();
        let call = ToolCall::new("read_file", json!({ "mode": "r" }));
        assert_eq!(
            validate_tool_call(&call, &tools),
            Err(GuardrailError::MissingRequired {
                tool_name: "read_file".into(),
                fields: vec!["path".into()],
            })
        );
    }

    #[test]
    fn rejects_reserved_name_collision() {
        // A model tries to invoke a reserved internal name that no tool declares.
        let tools = read_file_tools();
        let call = ToolCall::new(RLX_RESPOND, json!({ "message": "hi" }));
        assert_eq!(
            validate_tool_call(&call, &tools),
            Err(GuardrailError::ReservedToolName {
                name: RLX_RESPOND.into()
            })
        );
    }

    #[test]
    fn allows_reserved_name_when_explicitly_declared() {
        let mut tools = read_file_tools();
        tools.push(rlx_respond_tool_definition());
        let call = ToolCall::new(RLX_RESPOND, json!({ "message": "hi" }));
        assert!(validate_tool_call(&call, &tools).is_ok());
    }

    #[test]
    fn reserved_name_helpers() {
        assert!(is_reserved_tool_name(RLX_RESPOND));
        assert!(is_reserved_tool_name(RLX_EMIT_STRUCTURED));
        assert!(!is_reserved_tool_name("read_file"));
        assert!(tools_collide_with_reserved(
            &[rlx_respond_tool_definition()]
        ));
        assert!(!tools_collide_with_reserved(&read_file_tools()));
    }

    #[test]
    fn normalize_tool_arguments_handles_null_string_and_primitive() {
        assert_eq!(
            normalize_tool_arguments(&json!({"path": "README.md"})).unwrap()["path"],
            "README.md"
        );
        assert_eq!(
            normalize_tool_arguments(&json!("{\"path\":\"README.md\"}")).unwrap()["path"],
            "README.md"
        );
        assert_eq!(normalize_tool_arguments(&Value::Null), None);
        assert_eq!(normalize_tool_arguments(&json!(42)), Some(Map::new()));
    }

    #[test]
    fn sanitize_removes_unknown_and_ill_typed_arguments() {
        let tools = vec![ToolDef::function(
            "exec",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "host": { "type": "string", "enum": ["gateway"] }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        )];
        let call = ToolCall::new(
            "exec",
            json!({ "command": "echo ok", "host": "sandbox", "extra": true }),
        );
        let cleaned = sanitize_tool_arguments(&call, &tools).unwrap();
        assert_eq!(cleaned, json!({ "command": "echo ok" }));
    }

    #[test]
    fn anyhow_variant_bridges_errors() {
        let tools = read_file_tools();
        let ok = ToolCall::new("read_file", json!({ "path": "README.md" }));
        assert!(validate_tool_call_anyhow(&ok, &tools).is_ok());
        let bad = ToolCall::new("nope", json!({}));
        assert!(validate_tool_call_anyhow(&bad, &tools).is_err());
    }

    #[test]
    fn tool_arguments_wire_string_always_object_json() {
        assert_eq!(tool_arguments_wire_string(&Value::Null), "{}");
        assert_eq!(tool_arguments_wire_string(&json!(42)), "{}");
        assert_eq!(
            tool_arguments_wire_string(&json!({ "path": "README.md" })),
            "{\"path\":\"README.md\"}"
        );
    }
}
