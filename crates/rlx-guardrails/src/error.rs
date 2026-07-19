// RLX models — guardrails: error types.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Error type returned by guardrail validation.

use std::fmt;

/// Why a guardrail check failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailError {
    /// The model called a tool that is not in the provided catalog.
    UnknownTool {
        /// The name the model tried to call.
        name: String,
    },
    /// The call is missing one or more required arguments.
    MissingRequired {
        /// The tool the call targeted.
        tool_name: String,
        /// The required argument names that were absent.
        fields: Vec<String>,
    },
    /// The model invoked a reserved RLX-internal tool name that no
    /// caller-supplied tool declared.
    ReservedToolName {
        /// The reserved name that collided.
        name: String,
    },
}

impl fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTool { name } => {
                write!(f, "unknown tool: {name:?} is not in the provided tool set")
            }
            Self::MissingRequired { tool_name, fields } => write!(
                f,
                "tool {tool_name:?} missing required argument(s): {}",
                fields.join(", ")
            ),
            Self::ReservedToolName { name } => {
                write!(f, "tool name {name:?} is reserved for RLX internals")
            }
        }
    }
}

impl std::error::Error for GuardrailError {}
