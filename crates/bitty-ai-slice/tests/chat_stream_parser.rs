use bitty_ai_runtime::{FragmentKind, StreamSink, VecSink};
use bitty_ai_slice::{ChatCompletionStreamParser, ChatStreamDelta, ChatStreamError};

#[test]
fn parse_text_stream_end_to_end() {
    let mut parser = ChatCompletionStreamParser::new();

    let chunk1 =
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n";
    let chunk2 = b"data: {\"choices\":[{\"delta\":{\"content\":\" world!\"}}]}\n\n";
    let chunk3 = b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n";
    let chunk4 = b"data: [DONE]\n\n";

    let mut deltas = Vec::new();
    deltas.extend(parser.feed(chunk1).expect("feed 1"));
    deltas.extend(parser.feed(chunk2).expect("feed 2"));
    deltas.extend(parser.feed(chunk3).expect("feed 3"));
    deltas.extend(parser.feed(chunk4).expect("feed 4"));

    assert_eq!(deltas.len(), 4);
    assert_eq!(deltas[0], ChatStreamDelta::Content("Hello".to_string()));
    assert_eq!(deltas[1], ChatStreamDelta::Content(" world!".to_string()));
    assert_eq!(
        deltas[2],
        ChatStreamDelta::Finished {
            reason: "stop".to_string(),
        }
    );
    match &deltas[3] {
        ChatStreamDelta::Usage(u) => {
            assert_eq!(u.input_tokens, 10);
            assert_eq!(u.output_tokens, 2);
        }
        other => panic!("expected Usage delta, got {other:?}"),
    }

    assert!(parser.is_finished());
    assert_eq!(parser.accumulated_text(), "Hello world!");

    let turn = parser.finish().expect("finish");
    assert_eq!(turn.text, "Hello world!");
    assert_eq!(turn.usage.input_tokens, 10);
    assert_eq!(turn.usage.output_tokens, 2);
    assert!(turn.tool_calls.is_empty());
}

#[test]
fn parse_streaming_tool_calls_fragmented() {
    let mut parser = ChatCompletionStreamParser::new();

    // Tool call split across multiple chunks
    let c1 = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"calc_sum\",\"arguments\":\"\"}}]}}]}\n\n";
    let c2 = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\": 1\"}}]}}]}\n\n";
    let c3 = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\", \\\"b\\\": 2}\"}}]}}]}\n\n";
    let c4 = b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n";
    let c5 = b"data: [DONE]\n\n";

    let mut deltas = Vec::new();
    deltas.extend(parser.feed(c1).expect("c1"));
    deltas.extend(parser.feed(c2).expect("c2"));
    deltas.extend(parser.feed(c3).expect("c3"));
    deltas.extend(parser.feed(c4).expect("c4"));
    deltas.extend(parser.feed(c5).expect("c5"));

    assert!(parser.is_finished());
    let turn = parser.finish().expect("turn finish");
    assert_eq!(turn.tool_calls.len(), 1);
    assert_eq!(turn.tool_calls[0].name, "calc_sum");
    assert_eq!(
        std::str::from_utf8(&turn.tool_calls[0].arguments).unwrap(),
        "{\"a\": 1, \"b\": 2}"
    );
}

#[test]
fn pipe_content_deltas_into_stream_sink() {
    let mut parser = ChatCompletionStreamParser::new();
    let mut sink = VecSink::default();
    let mut seq = 0;

    let chunk = b"data: {\"choices\":[{\"delta\":{\"content\":\"Live token 1 \"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"Live token 2\"}}]}\n\ndata: [DONE]\n\n";
    let deltas = parser.feed(chunk).expect("feed");

    for delta in &deltas {
        parser
            .pipe_to_sink(delta, &mut sink, &mut seq)
            .expect("pipe");
    }

    assert_eq!(sink.chunks().len(), 2);
    assert_eq!(sink.chunks()[0].seq, 0);
    assert_eq!(sink.chunks()[0].fragment.kind, FragmentKind::Markdown);
    assert_eq!(sink.chunks()[0].fragment.bytes, b"Live token 1 ");

    assert_eq!(sink.chunks()[1].seq, 1);
    assert_eq!(sink.chunks()[1].fragment.kind, FragmentKind::Markdown);
    assert_eq!(sink.chunks()[1].fragment.bytes, b"Live token 2");

    assert_eq!(
        std::str::from_utf8(&sink.concatenated_bytes()).unwrap(),
        "Live token 1 Live token 2"
    );
}

#[test]
fn malformed_json_fails_closed() {
    let mut parser = ChatCompletionStreamParser::new();
    let bad = b"data: {not valid json}\n\n";
    let err = parser.feed(bad).unwrap_err();
    match err {
        ChatStreamError::JsonParse(msg) => {
            assert!(!msg.is_empty());
        }
        other => panic!("expected JsonParse error, got {other:?}"),
    }
}

#[test]
fn reject_excessive_tool_call_index_fail_closed() {
    let mut parser = ChatCompletionStreamParser::new();

    let malicious_chunk = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":4294967295,\"function\":{\"name\":\"exploit\"}}]}}]}\n\n";

    let err = parser
        .feed(malicious_chunk)
        .expect_err("should reject out of range index");
    match err {
        ChatStreamError::JsonParse(msg) => {
            assert!(msg.contains("tool call index out of range"));
        }
        other => panic!("expected JsonParse error, got {other:?}"),
    }
}
