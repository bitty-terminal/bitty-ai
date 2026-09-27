use bitty_ai_runtime::sse::{
    MAX_SSE_EVENT_BYTES, MAX_SSE_LINE_BYTES, SseError, SseEvent, SseParser,
};

#[test]
fn single_event_dispatch() {
    let mut parser = SseParser::new();
    let events = parser
        .feed(b"data: hello world\n\n")
        .expect("parse success");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0], SseEvent::new("message", "hello world"));
    assert_eq!(events[0].id, None);
    assert_eq!(events[0].retry, None);
    assert!(!events[0].is_done());
}

#[test]
fn multiline_data_joined_with_newline() {
    let mut parser = SseParser::new();
    let input = b"data: first line\ndata: second line\ndata: third line\n\n";
    let events = parser.feed(input).expect("parse success");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, "first line\nsecond line\nthird line");
}

#[test]
fn custom_event_id_and_retry() {
    let mut parser = SseParser::new();
    let input = b"event: update\nid: evt-101\nretry: 2500\ndata: payload-data\n\n";
    let events = parser.feed(input).expect("parse success");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event, "update");
    assert_eq!(events[0].id, Some("evt-101".to_string()));
    assert_eq!(events[0].retry, Some(2500));
    assert_eq!(events[0].data, "payload-data");
    assert_eq!(parser.last_event_id(), Some("evt-101"));
}

#[test]
fn comment_lines_and_empty_dispatch_ignored() {
    let mut parser = SseParser::new();
    let input = b": this is a ping comment\n\n: another comment\ndata: real data\n\n";
    let events = parser.feed(input).expect("parse success");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, "real data");
}

#[test]
fn crlf_and_mixed_line_endings() {
    let mut parser = SseParser::new();
    let input = b"event: tick\r\ndata: crlf-data\r\n\r\nevent: tock\ndata: lf-data\n\n";
    let events = parser.feed(input).expect("parse success");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event, "tick");
    assert_eq!(events[0].data, "crlf-data");
    assert_eq!(events[1].event, "tock");
    assert_eq!(events[1].data, "lf-data");
}

#[test]
fn incremental_chunked_feeding_single_byte_steps() {
    let mut parser = SseParser::new();
    let raw = "event: stream\ndata: chunk 1\ndata: chunk 2\n\nevent: stream\ndata: [DONE]\n\n";
    let mut all_events = Vec::new();

    // Feed one byte at a time to stress-test arbitrary buffer boundaries
    for byte in raw.as_bytes() {
        let events = parser.feed(&[*byte]).expect("byte feed success");
        all_events.extend(events);
    }

    assert_eq!(all_events.len(), 2);
    assert_eq!(all_events[0].event, "stream");
    assert_eq!(all_events[0].data, "chunk 1\nchunk 2");
    assert!(!all_events[0].is_done());

    assert_eq!(all_events[1].event, "stream");
    assert_eq!(all_events[1].data, "[DONE]");
    assert!(all_events[1].is_done());
}

#[test]
fn incremental_split_in_multibyte_utf8() {
    let mut parser = SseParser::new();
    // '你' is [0xE4, 0xBD, 0xA0]
    let part1 = b"data: \xE4";
    let part2 = b"\xBD";
    let part3 = b"\xA0\n\n";

    let ev1 = parser.feed(part1).expect("feed part1");
    assert_eq!(ev1.len(), 0);

    let ev2 = parser.feed(part2).expect("feed part2");
    assert_eq!(ev2.len(), 0);

    let ev3 = parser.feed(part3).expect("feed part3");
    assert_eq!(ev3.len(), 1);
    assert_eq!(ev3[0].data, "你");
}

#[test]
fn leading_space_after_colon_stripped_but_not_more() {
    let mut parser = SseParser::new();
    // One space after colon is stripped, second space is preserved
    let input = b"data:  two spaces\ndata:no space\n\n";
    let events = parser.feed(input).expect("parse success");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, " two spaces\nno space");
}

#[test]
fn line_without_colon_treated_as_empty_value() {
    let mut parser = SseParser::new();
    let input = b"data\n\n";
    let events = parser.feed(input).expect("parse success");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, "");
}

#[test]
fn finish_flushes_unterminated_event() {
    let mut parser = SseParser::new();
    let input = b"data: trailing without double newline";
    let events = parser.feed(input).expect("feed success");
    assert_eq!(events.len(), 0);

    let finished = parser.finish().expect("finish success");
    assert!(finished.is_some());
    let ev = finished.unwrap();
    assert_eq!(ev.data, "trailing without double newline");
}

#[test]
fn oversized_line_fails_closed() {
    let mut parser = SseParser::new();
    let mut oversized = Vec::with_capacity(MAX_SSE_LINE_BYTES + 10);
    oversized.extend_from_slice(b"data: ");
    oversized.resize(MAX_SSE_LINE_BYTES + 10, b'a');

    let err = parser.feed(&oversized).unwrap_err();
    match err {
        SseError::LineTooLong { max, actual } => {
            assert_eq!(max, MAX_SSE_LINE_BYTES);
            assert!(actual > max);
        }
        other => panic!("expected LineTooLong, got {other:?}"),
    }
}

#[test]
fn oversized_event_data_fails_closed() {
    let mut parser = SseParser::new();
    let chunk_size = 16 * 1024;
    let mut line = Vec::with_capacity(chunk_size + 10);
    line.extend_from_slice(b"data: ");
    line.resize(chunk_size + 6, b'x');
    line.push(b'\n');

    // Repeatedly feed data lines until accumulated event data exceeds MAX_SSE_EVENT_BYTES
    let mut failed = false;
    for _ in 0..5 {
        match parser.feed(&line) {
            Ok(_) => {}
            Err(SseError::EventTooLong { max, actual }) => {
                assert_eq!(max, MAX_SSE_EVENT_BYTES);
                assert!(actual > max);
                failed = true;
                break;
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
    assert!(failed, "expected EventTooLong error");
}

#[test]
fn invalid_retry_integer_fails_closed() {
    let mut parser = SseParser::new();
    let input = b"retry: not-a-number\n\n";
    let err = parser.feed(input).unwrap_err();
    match err {
        SseError::InvalidRetry(val) => {
            assert_eq!(val, "not-a-number");
        }
        other => panic!("expected InvalidRetry, got {other:?}"),
    }
}
