//! Live chat streaming CLI example (AI-0160).
//!
//! Demonstrates end-to-end real-time LLM chat completion streaming using
//! [`ChatCompletionStreamParser`], [`SseParser`], and [`StreamSink`].
//!
//! # Usage
//!
//! ```bash
//! # Default prompt with OpenRouter DeepSeek
//! export OPENROUTER_API_KEY="sk-or-v1-..."
//! cargo run -p bitty-ai-slice --example live_chat
//!
//! # Custom prompt
//! cargo run -p bitty-ai-slice --example live_chat -- "Explain why Rust ensures memory safety in 3 bullet points."
//!
//! # Custom model
//! export OPENROUTER_MODEL="deepseek/deepseek-chat"
//! cargo run -p bitty-ai-slice --example live_chat -- "Hello from Bitty Terminal!"
//! ```

use std::error::Error;
use std::io::{self, Read, Write};
use std::process::{Command, Stdio};

use bitty_ai_runtime::StreamSink;
use bitty_ai_runtime::stream::VecSink;
use bitty_ai_slice::{ChatCompletionStreamParser, ChatStreamDelta};

const DEFAULT_MODEL: &str = "deepseek/deepseek-chat";
const DEFAULT_PROMPT: &str = "Explain what Server-Sent Events (SSE) are and why they are useful for LLM streaming in 2 concise sentences.";
const ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";

fn main() -> Result<(), Box<dyn Error>> {
    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(key) if !key.trim().is_empty() => key.trim().to_owned(),
        _ => {
            eprintln!("Error: OPENROUTER_API_KEY environment variable is not set.");
            eprintln!();
            eprintln!("To run this live streaming example, please set your OpenRouter API key:");
            eprintln!("  export OPENROUTER_API_KEY=\"sk-or-v1-...\"");
            eprintln!();
            eprintln!(
                "You can also optionally specify a model via OPENROUTER_MODEL (default: {DEFAULT_MODEL})."
            );
            std::process::exit(1);
        }
    };

    let model = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let cli_prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if cli_prompt.trim().is_empty() {
        DEFAULT_PROMPT
    } else {
        cli_prompt.as_str()
    };

    println!("============================================================");
    println!("  Bitty AI Live Chat Stream Runner (AI-0160)");
    println!("============================================================");
    println!("Model:    {model}");
    println!("Endpoint: {ENDPOINT}");
    println!("Prompt:   {prompt}");
    println!("------------------------------------------------------------");
    println!("Streaming response:\n");

    let payload = serde_json::json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": true,
    });
    let payload_str = serde_json::to_string(&payload)?;

    let mut child = Command::new("curl")
        .arg("-s")
        .arg("-N") // unbuffered for SSE streaming
        .arg("-X")
        .arg("POST")
        .arg(ENDPOINT)
        .arg("-H")
        .arg(format!("Authorization: Bearer {api_key}"))
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-H")
        .arg("Accept: text/event-stream")
        .arg("-d")
        .arg(&payload_str)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn curl: {e}. Ensure curl is installed and on PATH."))?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or("Failed to capture child stdout")?;

    let mut parser = ChatCompletionStreamParser::new();
    let mut sink = VecSink::new();
    let mut seq = 0;
    let mut buffer = [0u8; 128]; // Small buffer to exercise incremental byte feeding

    loop {
        let n = stdout.read(&mut buffer)?;
        if n == 0 {
            break;
        }

        let deltas = parser.feed(&buffer[..n])?;
        for delta in &deltas {
            match delta {
                ChatStreamDelta::Content(text) => {
                    parser.pipe_to_sink(delta, &mut sink, &mut seq)?;
                    print!("{text}");
                    io::stdout().flush()?;
                }
                ChatStreamDelta::ToolCall { name, .. } => {
                    if let Some(tool_name) = name {
                        print!("\n[Tool Call Requested: {tool_name}]");
                        io::stdout().flush()?;
                    }
                }
                ChatStreamDelta::Finished { reason } => {
                    // Stream finished signal
                    let _ = reason;
                }
                ChatStreamDelta::Usage(_) => {
                    // Final token usage stats
                }
            }
        }
    }

    let status = child.wait()?;
    if !status.success() {
        let mut err_msg = String::new();
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_string(&mut err_msg);
        }
        eprintln!("\nWarning: curl process exited with status: {status}. {err_msg}");
    }

    let turn = parser.finish()?;

    println!("\n");
    println!("------------------------------------------------------------");
    println!("Stream Summary:");
    println!("  Retained Sink Chunks: {}", sink.chunks().len());
    println!("  Total Emitted Bytes:  {}", turn.text.len());
    println!("  Prompt Tokens:        {}", turn.usage.input_tokens);
    println!("  Completion Tokens:    {}", turn.usage.output_tokens);
    println!("============================================================");

    Ok(())
}
