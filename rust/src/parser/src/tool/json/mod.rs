// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared parser core for JSON tool calls wrapped by text markers.

pub use granite4::Granite4ToolParser;
pub use hermes::HermesToolParser;
pub use internlm2::Internlm2ToolParser;
pub use llama::Llama3JsonToolParser;
pub use mistral::MistralToolParser;
pub use phi4mini::Phi4MiniJsonToolParser;
pub use qwen::Qwen3XmlToolParser;

mod granite4;
mod hermes;
mod internlm2;
mod llama;
mod mistral;
mod phi4mini;
mod qwen;

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::{AddContext, ModalResult, StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::literal;

use super::utils::{
    JsonObjectScanState, json_str, parse_buffered_event, safe_text_len, take_json_object,
};
use super::{Result, ToolCallDelta, ToolParserOutput};

pub(crate) type JsonToolInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy)]
pub(crate) struct JsonToolCallConfig {
    pub parser_name: &'static str,
    pub start_marker: &'static str,
    pub end_marker: &'static str,
    pub marker_whitespace: JsonToolCallWhitespace,
    pub delimiter: Option<&'static str>,
    pub name_key: &'static str,
    /// Candidate JSON keys naming the arguments payload, tried in order.
    /// Most parsers use a single key like `["arguments"]`, but some accept
    /// multiple (e.g. InternLM2 accepts `parameters` or `arguments`).
    pub arguments_key: &'static [&'static str],
    /// Accept one matching Markdown fence around the tool-call JSON.
    pub allow_markdown_fence: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum JsonToolCallWhitespace {
    Optional,
    Exact(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonToolCallMode {
    Text,
    Header,
    Arguments {
        json_scan: JsonObjectScanState,
        markdown_fenced: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JsonToolCallEvent {
    Text {
        len: usize,
    },
    ToolCallStart,
    ToolCallHeader {
        function_name: String,
        markdown_fenced: bool,
    },
    Arguments {
        len: usize,
    },
    ToolCallDelimiter,
    ToolCallEnd {
        markdown_fenced: bool,
    },
}

#[derive(Debug)]
struct PendingFencedToolCall {
    tool_index: usize,
    function_name: String,
    arguments_start: usize,
}

/// Tool parser core for marker-wrapped JSON tool calls.
#[derive(Debug)]
struct JsonToolCallParser {
    config: JsonToolCallConfig,
    buffer: String,
    mode: JsonToolCallMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
    pending_fenced_raw: String,
    pending_fenced_call: Option<PendingFencedToolCall>,
}

impl JsonToolCallParser {
    /// Create a marker-wrapped JSON tool-call parser.
    fn new(config: JsonToolCallConfig) -> Self {
        assert!(
            !config.allow_markdown_fence || config.delimiter.is_none(),
            "Markdown-fenced JSON requires one tool call per marker"
        );
        assert!(
            !config.allow_markdown_fence
                || matches!(config.marker_whitespace, JsonToolCallWhitespace::Exact(_)),
            "Markdown-fenced JSON requires exact marker whitespace"
        );
        Self {
            config,
            buffer: String::new(),
            mode: JsonToolCallMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
            pending_fenced_raw: String::new(),
            pending_fenced_call: None,
        }
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        let config = self.config;

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_json_tool_call_event(input, &mut self.mode, config)
        })? {
            self.apply_event(event, consumed_len, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            JsonToolCallMode::Text => output.push_text(&self.buffer),
            JsonToolCallMode::Header | JsonToolCallMode::Arguments { .. } => {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.config.parser_name
                ));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    /// Apply one parsed JSON tool-call event to parser state and output.
    fn apply_event(
        &mut self,
        event: JsonToolCallEvent,
        consumed_len: usize,
        output: &mut ToolParserOutput,
    ) -> Result<()> {
        match event {
            JsonToolCallEvent::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            JsonToolCallEvent::ToolCallStart => self.mode = JsonToolCallMode::Header,
            JsonToolCallEvent::ToolCallHeader {
                function_name,
                markdown_fenced,
            } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = JsonToolCallMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                    markdown_fenced,
                };
                if markdown_fenced {
                    let JsonToolCallWhitespace::Exact(marker_whitespace) =
                        self.config.marker_whitespace
                    else {
                        unreachable!("fenced parser configuration was checked at construction");
                    };
                    self.pending_fenced_raw.push_str(self.config.start_marker);
                    self.pending_fenced_raw.push_str(marker_whitespace);
                    self.pending_fenced_raw.push_str(&self.buffer[..consumed_len]);
                    self.pending_fenced_call = Some(PendingFencedToolCall {
                        tool_index,
                        function_name,
                        arguments_start: self.pending_fenced_raw.len(),
                    });
                } else {
                    output.push_call(ToolCallDelta {
                        tool_index,
                        name: Some(function_name),
                        arguments: String::new(),
                    });
                }
            }
            JsonToolCallEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "{} arguments without an active tool call",
                        self.config.parser_name
                    ));
                };
                if matches!(
                    &self.mode,
                    JsonToolCallMode::Arguments {
                        markdown_fenced: true,
                        ..
                    }
                ) {
                    self.pending_fenced_raw.push_str(&self.buffer[..consumed_len]);
                } else {
                    output.push_call(ToolCallDelta {
                        tool_index,
                        name: None,
                        arguments: self.buffer[..consumed_len].to_string(),
                    });
                }
            }
            JsonToolCallEvent::ToolCallDelimiter => {
                self.active_tool_index = None;
                self.mode = JsonToolCallMode::Header;
            }
            JsonToolCallEvent::ToolCallEnd { markdown_fenced } => {
                if markdown_fenced {
                    let Some(pending) = self.pending_fenced_call.as_ref() else {
                        return Err(parsing_failed!(
                            "{} fenced tool call ended without a header",
                            self.config.parser_name
                        ));
                    };
                    let arguments = &self.pending_fenced_raw[pending.arguments_start..];
                    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(arguments)
                        .map_err(|error| {
                        parsing_failed!(
                            "invalid fenced {} arguments: {}",
                            self.config.parser_name,
                            error
                        )
                    })?;
                    let arguments = arguments.to_string();
                    let pending = self
                        .pending_fenced_call
                        .take()
                        .expect("fenced tool call was checked above");
                    output.push_call(ToolCallDelta {
                        tool_index: pending.tool_index,
                        name: Some(pending.function_name),
                        arguments,
                    });
                    self.pending_fenced_raw.clear();
                }
                self.active_tool_index = None;
                self.mode = JsonToolCallMode::Text;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        let pending_tool_call_start =
            self.config.allow_markdown_fence && matches!(self.mode, JsonToolCallMode::Header);
        self.mode = JsonToolCallMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        self.pending_fenced_call = None;

        let mut uncommitted = String::new();
        if pending_tool_call_start {
            let JsonToolCallWhitespace::Exact(marker_whitespace) = self.config.marker_whitespace
            else {
                unreachable!("fenced parser configuration was checked at construction");
            };
            uncommitted.push_str(self.config.start_marker);
            uncommitted.push_str(marker_whitespace);
        }
        uncommitted.push_str(&std::mem::take(&mut self.pending_fenced_raw));
        uncommitted.push_str(&std::mem::take(&mut self.buffer));
        uncommitted
    }
}

/// Parse a JSON tool-call event for the current parser mode.
fn parse_next_json_tool_call_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut JsonToolCallMode,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    match mode {
        JsonToolCallMode::Text => parse_text_event(input, config),
        JsonToolCallMode::Header => tool_call_header_event(input, config),
        JsonToolCallMode::Arguments {
            json_scan,
            markdown_fenced,
        } => parse_arguments_event(input, json_scan, *markdown_fenced, config),
    }
}

/// Parse a text-mode JSON tool-call event.
fn parse_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    alt((
        |input: &mut JsonToolInput<'_>| tool_call_start_event(input, config),
        |input: &mut JsonToolInput<'_>| safe_text_event(input, config),
    ))
    .parse_next(input)
}

/// Parse a marker-wrapped JSON tool-call start marker.
fn tool_call_start_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    seq!(
        _: literal(config.start_marker),
        _: |input: &mut JsonToolInput<'_>| marker_whitespace(input, config),
    )
    .value(JsonToolCallEvent::ToolCallStart)
    .parse_next(input)
}

/// Parse a marker-wrapped JSON tool-call header before the raw arguments
/// payload.
pub(crate) fn tool_call_header_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    let (markdown_fenced, function_name) = (|input: &mut JsonToolInput<'_>| {
        let _ = ws0.parse_next(input)?;
        let markdown_fenced = markdown_fence_start(input, config.allow_markdown_fence)?;
        if markdown_fenced {
            let _ = ws0.parse_next(input)?;
        }
        let (function_name,) = seq!(
            _: literal("{"),
            _: ws0,
            _: |input: &mut JsonToolInput<'_>| json_key(input, config.name_key),
            _: ws0,
            _: literal(":"),
            _: ws0,
            json_str,
            _: ws0,
            _: literal(","),
            _: ws0,
            _: |input: &mut JsonToolInput<'_>| json_arguments_key(input, config.arguments_key),
            _: ws0,
            _: literal(":"),
            _: ws0,
        )
        .parse_next(input)?;
        Ok((markdown_fenced, function_name))
    })
    .context(StrContext::Label(config.parser_name))
    .parse_next(input)?;

    Ok(JsonToolCallEvent::ToolCallHeader {
        function_name,
        markdown_fenced,
    })
}

/// Parse a Markdown fence when the next non-whitespace byte starts one.
fn markdown_fence_start(input: &mut JsonToolInput<'_>, enabled: bool) -> ModalResult<bool> {
    if !enabled || !input.starts_with('`') {
        return Ok(false);
    }

    alt((literal("```json\n"), literal("```\n"))).value(true).parse_next(input)
}

/// Parse a configured JSON object key.
fn json_key(input: &mut JsonToolInput<'_>, key: &'static str) -> ModalResult<()> {
    seq!(
        _: literal("\""),
        _: literal(key).context(StrContext::Expected(StrContextValue::StringLiteral(key))),
        _: literal("\""),
    )
    .void()
    .parse_next(input)
}

/// Parse a JSON object key accepting any of `candidates`.
///
/// The full quoted key is consumed and compared against the candidate list,
/// so this works correctly under partial input regardless of key lengths.
///
/// On mismatch, each candidate is attached as its own `Expected` context so the
/// error enumerates every valid key ("expected `a`, expected `b`"). Because
/// `StrContextValue::StringLiteral` carries a single `&'static str`, the
/// contexts are added in a loop over `candidates` rather than through chained
/// `.context(...)` calls, which keeps the diagnostics complete for any number
/// of candidates.
fn json_arguments_key(
    input: &mut JsonToolInput<'_>,
    candidates: &'static [&'static str],
) -> ModalResult<()> {
    let start = input.checkpoint();
    json_str
        .verify(|key: &String| candidates.contains(&key.as_str()))
        .void()
        .parse_next(input)
        .map_err(|err| {
            err.map(|context_error| {
                candidates.iter().fold(context_error, |context_error, candidate| {
                    context_error.add_context(
                        &*input,
                        &start,
                        StrContext::Expected(StrContextValue::StringLiteral(candidate)),
                    )
                })
            })
        })
}

/// Parse one event inside a marker-wrapped JSON tool-call arguments payload.
fn parse_arguments_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
    markdown_fenced: bool,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    if json_scan.complete() {
        tool_call_close_event(input, markdown_fenced, config)
    } else {
        argument_delta_event(input, json_scan)
    }
}

/// Parse a raw JSON arguments delta.
fn argument_delta_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<JsonToolCallEvent> {
    take_json_object(input, json_scan).map(|len| JsonToolCallEvent::Arguments { len })
}

/// Parse a marker-wrapped JSON tool-call close marker.
fn tool_call_close_event(
    input: &mut JsonToolInput<'_>,
    markdown_fenced: bool,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    seq!(_: ws0, _: literal("}")).parse_next(input)?;

    match config.delimiter {
        Some(delimiter) => alt((
            |input: &mut JsonToolInput<'_>| tool_call_end_event(input, markdown_fenced, config),
            |input: &mut JsonToolInput<'_>| tool_call_delimiter_event(input, delimiter),
        ))
        .parse_next(input),
        None => tool_call_end_event(input, markdown_fenced, config),
    }
}

/// Parse a marker-wrapped JSON tool-call end marker.
fn tool_call_end_event(
    input: &mut JsonToolInput<'_>,
    markdown_fenced: bool,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    if markdown_fenced {
        seq!(
            _: |input: &mut JsonToolInput<'_>| marker_whitespace(input, config),
            _: literal("```"),
            _: |input: &mut JsonToolInput<'_>| marker_whitespace(input, config),
            _: literal(config.end_marker),
        )
        .value(JsonToolCallEvent::ToolCallEnd { markdown_fenced })
        .parse_next(input)
    } else {
        seq!(
            _: |input: &mut JsonToolInput<'_>| marker_whitespace(input, config),
            _: literal(config.end_marker),
        )
        .value(JsonToolCallEvent::ToolCallEnd { markdown_fenced })
        .parse_next(input)
    }
}

/// Parse a delimiter between JSON tool calls inside one marker block.
fn tool_call_delimiter_event(
    input: &mut JsonToolInput<'_>,
    delimiter: &'static str,
) -> ModalResult<JsonToolCallEvent> {
    seq!(
        _: ws0,
        _: literal(delimiter),
        _: ws0,
    )
    .value(JsonToolCallEvent::ToolCallDelimiter)
    .parse_next(input)
}

/// Parse configured whitespace around a marker-wrapped JSON tool call.
fn marker_whitespace(input: &mut JsonToolInput<'_>, config: JsonToolCallConfig) -> ModalResult<()> {
    match config.marker_whitespace {
        JsonToolCallWhitespace::Optional => ws0.void().parse_next(input),
        JsonToolCallWhitespace::Exact(whitespace) => literal(whitespace).void().parse_next(input),
    }
}

/// Parse a safe text run before the next marker-wrapped JSON tool call.
fn safe_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    safe_text_len(input, config.start_marker).map(|len| JsonToolCallEvent::Text { len })
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
    use crate::tool::ToolParserOutput;

    const DELIMITED_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
        parser_name: "Delimited JSON",
        start_marker: "<tool_calls>",
        end_marker: "</tool_calls>",
        marker_whitespace: JsonToolCallWhitespace::Optional,
        delimiter: Some("<"),
        name_key: "function",
        arguments_key: &["parameters"],
        allow_markdown_fence: false,
    };

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"{{"function":"{function_name}","parameters":{arguments}}}"#)
    }

    fn build_tool_calls(tool_calls: &[String]) -> String {
        format!("<tool_calls>{}</tool_calls>", tool_calls.join(" <\n"))
    }

    fn collect_chunks(parser: &mut JsonToolCallParser, chunks: &[&str]) -> ToolParserOutput {
        let mut output = ToolParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());
        output.coalesce()
    }

    #[test]
    fn json_tool_call_tolerates_whitespace_before_outer_brace() {
        // Pretty-printed JSON puts whitespace between the arguments object's `}`
        // and the outer object's `}`; it must still parse (json.loads parity).
        let mut whole = JsonToolCallParser::new(DELIMITED_CONFIG);
        let whole_output = collect_chunks(
            &mut whole,
            &[r#"<tool_calls>{"function":"f","parameters":{"x":1} }</tool_calls>"#],
        );
        assert_eq!(whole_output.calls().len(), 1);
        assert_eq!(whole_output.calls()[0].name.as_deref(), Some("f"));
        assert_eq!(whole_output.calls()[0].arguments, r#"{"x":1}"#);

        // Same input, but the whitespace before the outer `}` is split across a
        // chunk boundary (exercises `ws0` returning Incomplete on `Partial`).
        let mut chunked = JsonToolCallParser::new(DELIMITED_CONFIG);
        let chunked_output = collect_chunks(
            &mut chunked,
            &[
                r#"<tool_calls>{"function":"f","parameters":{"x":1}"#,
                " ",
                "}</tool_calls>",
            ],
        );
        assert_eq!(chunked_output.calls().len(), 1);
        assert_eq!(chunked_output.calls()[0].arguments, r#"{"x":1}"#);
    }

    #[test]
    fn json_tool_call_delimiter_extracts_multiple_calls_in_one_block() {
        let input = build_tool_calls(&[
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        ]);
        let mut parser = JsonToolCallParser::new(DELIMITED_CONFIG);

        let output = collect_chunks(&mut parser, &[&input]);

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"location\":\"Shanghai\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "add",
                            ),
                            arguments: "{\"x\":1,\"y\":2}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn json_tool_call_delimiter_can_arrive_in_later_chunk() {
        let mut parser = JsonToolCallParser::new(DELIMITED_CONFIG);
        let chunks = [
            r#"<tool_calls>{"function":"get_weather","parameters":{"location":"Shanghai"}}"#,
            " <\n",
            r#"{"function":"add","parameters":{"x":1,"y":2}}"#,
            "</tool_calls>",
        ];

        let output = collect_chunks(&mut parser, &chunks);

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"location\":\"Shanghai\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "add",
                            ),
                            arguments: "{\"x\":1,\"y\":2}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn json_tool_call_end_marker_wins_over_delimiter_prefix() {
        let mut parser = JsonToolCallParser::new(DELIMITED_CONFIG);
        let chunks = [
            r#"<tool_calls>{"function":"get_weather","parameters":{"location":"Shanghai"}}"#,
            " ",
            "</tool_calls> trailing text",
        ];

        let output = collect_chunks(&mut parser, &chunks);

        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        " trailing text",
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"location\":\"Shanghai\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }
}
