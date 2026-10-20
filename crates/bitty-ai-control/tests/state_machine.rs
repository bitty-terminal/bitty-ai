//! Deterministic state-machine, framing, and fail-closed tests.
//!
//! No TLS handshake, socket, thread, or clock is used. The helper state
//! machine is driven over an in-memory `Read + Write` stream and an injected
//! registry seam. Wire fixtures use the pure `bitty-ipc-api`
//! `provider_publication` codec; the tests never invent a second format.

mod common;

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::io::{self, Cursor, Read, Write};
use std::rc::Rc;

use bitty_ai_control::{
    EstablishmentExpectations, FrameError, FrameReader, FrameWriter, HelperControl,
    HelperControlError, ProofState, RegistryError, RegistrySnapshot, RegistryView,
    RevocationReason, Step,
};
use bitty_ipc_api::provider_publication::{
    AckBody, AckId, AdmitBody, BoundBody, Capability, ControlError, ControlRecord, Digest,
    ErrorCode, EstablishmentEcho, InvalidateBody, Kind, LifetimeBody, LifetimeId, ModelEntry,
    Nonce, ProviderEntry, ProviderRoutingProjection, PublicationId, PublishBody, Reason,
    RecordBody, Sender, SessionId, Tuple,
};
use common::{fixtures, valid_manifest};

// ── deterministic in-memory transport ───────────────────────────────────────

struct StreamState {
    input: VecDeque<u8>,
    output: Vec<u8>,
    read_chunk: usize,
    write_chunk: usize,
    fail_after: Option<usize>,
    written: usize,
}

#[derive(Clone)]
struct FakeStream {
    state: Rc<RefCell<StreamState>>,
}

impl FakeStream {
    fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(StreamState {
                input: VecDeque::new(),
                output: Vec::new(),
                read_chunk: 0,
                write_chunk: 0,
                fail_after: None,
                written: 0,
            })),
        }
    }

    fn send_bytes(&self, bytes: &[u8]) {
        self.state.borrow_mut().input.extend(bytes.iter().copied());
    }

    fn send_record(&self, record: &ControlRecord) {
        self.send_bytes(&frame(record));
    }

    fn send_frame_bytes(&self, frame: &[u8]) {
        self.send_bytes(frame);
    }

    fn take_output(&self) -> Vec<u8> {
        std::mem::take(&mut self.state.borrow_mut().output)
    }

    fn output_is_empty(&self) -> bool {
        self.state.borrow().output.is_empty()
    }

    fn set_read_chunk(&self, chunk: usize) {
        self.state.borrow_mut().read_chunk = chunk;
    }

    fn set_write_chunk(&self, chunk: usize) {
        self.state.borrow_mut().write_chunk = chunk;
    }

    fn fail_writes_after(&self, bytes: Option<usize>) {
        let mut state = self.state.borrow_mut();
        state.fail_after = bytes.map(|limit| state.written.saturating_add(limit));
    }

    fn written(&self) -> usize {
        self.state.borrow().written
    }
}

impl Read for FakeStream {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let mut state = self.state.borrow_mut();
        if state.input.is_empty() {
            return Ok(0);
        }
        let limit = if state.read_chunk == 0 {
            buffer.len()
        } else {
            state.read_chunk.min(buffer.len())
        };
        let count = limit.min(state.input.len());
        for slot in &mut buffer[..count] {
            *slot = state.input.pop_front().expect("count bound");
        }
        Ok(count)
    }
}

impl Write for FakeStream {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let mut state = self.state.borrow_mut();
        let allowed = match state.fail_after {
            Some(limit) if state.written >= limit => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "injected write failure",
                ));
            }
            Some(limit) => buffer.len().min(limit - state.written),
            None => buffer.len(),
        };
        let chunk = if state.write_chunk == 0 {
            allowed
        } else {
            state.write_chunk.min(allowed)
        };
        if chunk == 0 {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "injected write failure",
            ));
        }
        state.output.extend_from_slice(&buffer[..chunk]);
        state.written += chunk;
        Ok(chunk)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn frame(record: &ControlRecord) -> Vec<u8> {
    let bytes = record.encode().expect("record encodes");
    let length = u32::try_from(bytes.len()).expect("record length fits u32");
    let mut frame = Vec::with_capacity(4 + bytes.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&bytes);
    frame
}

fn parse_frames(bytes: &[u8]) -> Vec<ControlRecord> {
    let mut cursor = Cursor::new(bytes);
    let mut reader = FrameReader::new();
    let mut records = Vec::new();
    loop {
        match reader.read_record(&mut cursor) {
            Ok(record) => records.push(record),
            Err(FrameError::Closed) => break,
            Err(error) => panic!("output frame rejected: {error:?}"),
        }
    }
    records
}

// ── injected registry seam ──────────────────────────────────────────────────

struct RegistryState {
    feed_revision: u64,
    projection: ProviderRoutingProjection,
    snapshots: usize,
    comparisons: usize,
    last_retained: Option<Vec<u8>>,
    fail_snapshot: bool,
    fail_compare: bool,
}

#[derive(Clone)]
struct TestRegistry {
    state: Rc<RefCell<RegistryState>>,
}

impl TestRegistry {
    fn new(feed_revision: u64, projection: ProviderRoutingProjection) -> Self {
        Self {
            state: Rc::new(RefCell::new(RegistryState {
                feed_revision,
                projection,
                snapshots: 0,
                comparisons: 0,
                last_retained: None,
                fail_snapshot: false,
                fail_compare: false,
            })),
        }
    }

    fn mutate(&self, projection: ProviderRoutingProjection) {
        let mut state = self.state.borrow_mut();
        state.feed_revision = state.feed_revision.checked_add(1).expect("test revision");
        state.projection = projection;
    }

    fn replace_projection_without_revision(&self, projection: ProviderRoutingProjection) {
        self.state.borrow_mut().projection = projection;
    }

    fn set_revision(&self, feed_revision: u64) {
        self.state.borrow_mut().feed_revision = feed_revision;
    }

    fn set_fail_snapshot(&self, fail: bool) {
        self.state.borrow_mut().fail_snapshot = fail;
    }

    fn set_fail_compare(&self, fail: bool) {
        self.state.borrow_mut().fail_compare = fail;
    }

    fn comparisons(&self) -> usize {
        self.state.borrow().comparisons
    }

    fn last_retained(&self) -> Option<Vec<u8>> {
        self.state.borrow().last_retained.clone()
    }
}

impl RegistryView for TestRegistry {
    fn snapshot(&self) -> Result<RegistrySnapshot, RegistryError> {
        let mut state = self.state.borrow_mut();
        state.snapshots += 1;
        if state.fail_snapshot {
            return Err(RegistryError::Unavailable);
        }
        RegistrySnapshot::new(state.feed_revision, state.projection.clone())
    }

    fn retained_matches(&self, feed_revision: u64, retained: &[u8]) -> Result<bool, RegistryError> {
        let mut state = self.state.borrow_mut();
        state.comparisons += 1;
        state.last_retained = Some(retained.to_vec());
        if state.fail_compare {
            return Err(RegistryError::Unavailable);
        }
        if state.feed_revision != feed_revision {
            return Ok(false);
        }
        Ok(state.projection.canonical_bytes() == retained)
    }
}

// ── fixtures ────────────────────────────────────────────────────────────────

fn model(name: &str) -> ModelEntry {
    ModelEntry::new(name, vec![Capability::Text], 128_000, 1, 2).expect("model entry")
}

fn projection(names: &[&str]) -> ProviderRoutingProjection {
    let models: Vec<ModelEntry> = names.iter().map(|name| model(name)).collect();
    ProviderRoutingProjection::new(vec![
        ProviderEntry::new("provider-a", models).expect("provider entry"),
    ])
    .expect("projection")
}

fn empty_projection() -> ProviderRoutingProjection {
    ProviderRoutingProjection::empty()
}

fn echo_for(manifest: &bitty_ai_control::LaunchManifest) -> EstablishmentEcho {
    EstablishmentEcho {
        bootstrap_format_version: 1,
        control_protocol_version: 1,
        core_instance_nonce: Nonce::new(manifest.core_instance_nonce),
        launch_nonce: SessionId::new(manifest.launch_nonce),
        component_name: manifest.component_name.clone(),
        descriptor_version: manifest.descriptor_version.clone(),
        executable_name: manifest.executable_name.clone(),
        executable_sha256: Digest::new(manifest.executable_sha256),
        component_protocol_version: manifest.component_protocol_version,
        control_channel_nonce: Nonce::new(manifest.control_channel_nonce),
        data_channel_nonce: Nonce::new(manifest.data_channel_nonce),
        delegation_purpose: manifest.delegation_purpose.clone(),
        control_profile: manifest.control_profile.clone(),
        key_material_profile: manifest.key_material.profile,
        key_type: manifest.key_material.key_type,
        core_leaf_digest: Digest::of(&manifest.key_material.core_certificate_der),
        helper_leaf_digest: Digest::of(&manifest.key_material.helper_certificate_der),
    }
}

fn core_record(sequence: u64, session: SessionId, body: RecordBody) -> ControlRecord {
    ControlRecord {
        sender: Sender::Core,
        sequence,
        session_id: session,
        body,
    }
}

fn tuple_for(session: SessionId, version: u64, projection: &ProviderRoutingProjection) -> Tuple {
    Tuple::new(
        PublicationId::from_parts(session, version),
        version,
        Digest::of(&projection.canonical_bytes()),
    )
    .expect("tuple")
}

struct Harness {
    stream: FakeStream,
    registry: TestRegistry,
    control: HelperControl<FakeStream, TestRegistry>,
    echo: EstablishmentEcho,
    session: SessionId,
    core_sequence: u64,
    helper_private_key: Vec<u8>,
    core_private_key: Vec<u8>,
}

impl Harness {
    fn new(feed_revision: u64, projection: ProviderRoutingProjection) -> Self {
        let registry = TestRegistry::new(feed_revision, projection);
        Self::with_registry(registry)
    }

    fn with_registry(registry: TestRegistry) -> Self {
        let fixtures = fixtures();
        let manifest = valid_manifest(&fixtures);
        let echo = echo_for(&manifest);
        let session = SessionId::new(manifest.launch_nonce);
        let helper_private_key = fixtures.helper_key.clone();
        let core_private_key = fixtures.core_key.clone();
        let stream = FakeStream::new();
        let control = HelperControl::from_stream(manifest, stream.clone(), registry.clone())
            .expect("control");
        Self {
            stream,
            registry,
            control,
            echo,
            session,
            core_sequence: 0,
            helper_private_key,
            core_private_key,
        }
    }

    fn send_core(&mut self, body: RecordBody) -> u64 {
        self.core_sequence = self.core_sequence.checked_add(1).expect("core sequence");
        let sequence = self.core_sequence;
        self.stream
            .send_record(&core_record(sequence, self.session, body));
        sequence
    }

    fn send_core_with_sequence(&self, sequence: u64, body: RecordBody) {
        self.stream
            .send_record(&core_record(sequence, self.session, body));
    }

    fn send_raw_frame(&self, bytes: &[u8]) {
        self.stream.send_frame_bytes(bytes);
    }

    fn establish(&mut self) {
        self.send_core(RecordBody::Establish(self.echo.clone()));
        self.control.establish().expect("establish");
        let records = self.drain();
        assert_eq!(records.len(), 2, "establish emits exactly two records");
        assert_eq!(records[0].sender, Sender::Helper);
        assert_eq!(records[0].sequence, 1);
        assert_eq!(records[1].sender, Sender::Helper);
        assert_eq!(records[1].sequence, 2);
        let established = match &records[0].body {
            RecordBody::Established(body) => body,
            other => panic!("first record is not Established: {other:?}"),
        };
        assert_eq!(established.echo, self.echo);
        assert!(matches!(records[1].body, RecordBody::Projection(_)));
        assert!(self.stream.output_is_empty());
    }

    fn send_publish(&mut self, version: u64, projection: &ProviderRoutingProjection) -> Tuple {
        let lifetime_id = self.control.lifetime_id().expect("lifetime");
        let feed_revision = self.control.feed_revision().expect("revision");
        let tuple = tuple_for(self.session, version, projection);
        self.send_core(RecordBody::Publish(PublishBody {
            lifetime_id,
            feed_revision,
            tuple,
            payload: projection.clone(),
        }));
        tuple
    }

    fn publish(
        &mut self,
        version: u64,
        projection: &ProviderRoutingProjection,
    ) -> (Tuple, AckBody) {
        let tuple = self.send_publish(version, projection);
        let step = self.control.step().expect("publish step");
        assert_eq!(step, Step::Published { tuple });
        let records = self.drain();
        assert_eq!(records.len(), 1, "publish emits exactly one Ack");
        assert_eq!(records[0].sender, Sender::Helper);
        let ack = match records[0].body {
            RecordBody::Ack(ack) => ack,
            ref other => panic!("publish response is not Ack: {other:?}"),
        };
        assert_eq!(ack.tuple, tuple);
        assert_eq!(ack.lifetime_id, self.control.lifetime_id().unwrap());
        assert_eq!(ack.feed_revision, self.control.feed_revision().unwrap());
        (tuple, ack)
    }

    fn bind(&mut self, tuple: Tuple, ack: AckBody) -> BoundBody {
        self.send_core(RecordBody::Bind(ack));
        let step = self.control.step().expect("bind step");
        let records = self.drain();
        assert_eq!(records.len(), 1, "bind emits exactly one Bound");
        let bound = match records[0].body {
            RecordBody::Bound(bound) => bound,
            ref other => panic!("bind response is not Bound: {other:?}"),
        };
        assert_eq!(bound.tuple, tuple);
        assert_eq!(
            step,
            Step::Bound {
                tuple,
                receipt: bound.receipt,
            }
        );
        bound
    }

    fn send_invalidate(&mut self, lifetime_id: LifetimeId, feed_revision: u64, reason: Reason) {
        self.send_core(RecordBody::Invalidate(InvalidateBody {
            lifetime_id,
            feed_revision,
            reason,
        }));
    }

    fn drain(&self) -> Vec<ControlRecord> {
        parse_frames(&self.stream.take_output())
    }

    fn assert_no_output(&self) {
        assert!(
            self.stream.output_is_empty(),
            "unexpected helper output: {:?}",
            self.stream.take_output()
        );
    }

    fn assert_closed(&self, reason: RevocationReason) {
        assert!(self.control.is_closed());
        assert_eq!(self.control.revocation_reason(), Some(reason));
        assert!(self.control.lifetime_id().is_none());
        assert!(self.control.feed_revision().is_none());
        assert_eq!(self.control.proof_state(), ProofState::Revoked);
    }
}

type EchoCase = (&'static str, fn(&mut EstablishmentEcho));

fn expect_establishment(error: HelperControlError, field: &'static str) {
    match error {
        HelperControlError::EstablishmentMismatch { field: actual } => {
            assert_eq!(actual, field);
        }
        other => panic!("expected establishment mismatch, got {other:?}"),
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn expect_record_error(error: HelperControlError) {
    match error {
        HelperControlError::Frame(FrameError::Record(_)) | HelperControlError::Record(_) => {}
        other => panic!("expected a record rejection, got {other:?}"),
    }
}

// ── framing tests ───────────────────────────────────────────────────────────

#[test]
fn framed_round_trip_preserves_exactly_one_record() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let echo = echo_for(&manifest);
    let session = SessionId::new(manifest.launch_nonce);
    let record = core_record(1, session, RecordBody::Establish(echo));

    let mut writer = FrameWriter::new();
    writer.enqueue(&record).expect("enqueue");
    let mut output = Vec::new();
    writer.flush(&mut output).expect("flush");
    assert_eq!(writer.pending_records(), 0);

    let mut reader = FrameReader::new();
    let mut cursor = Cursor::new(output);
    assert_eq!(reader.read_record(&mut cursor).expect("read"), record);
    assert!(matches!(
        reader.read_record(&mut cursor),
        Err(FrameError::Closed)
    ));
}

#[test]
fn framed_reader_rejects_every_bound_and_truncation_case() {
    let mut reader = FrameReader::new();

    assert!(matches!(
        reader.read_record(&mut Cursor::new(Vec::new())),
        Err(FrameError::Closed)
    ));
    assert!(matches!(
        reader.read_record(&mut Cursor::new(vec![0, 0, 0, 0])),
        Err(FrameError::ZeroLength)
    ));
    assert!(matches!(
        reader.read_record(&mut Cursor::new(vec![0, 4, 0, 1, 0, 0,])),
        Err(FrameError::FrameTooLarge)
    ));
    assert!(matches!(
        reader.read_record(&mut Cursor::new(vec![0, 0, 0, 10])),
        Err(FrameError::FrameShorterThanHeader { actual: 10 })
    ));
    assert!(matches!(
        reader.read_record(&mut Cursor::new(vec![0, 0])),
        Err(FrameError::TruncatedPrefix { received: 2 })
    ));

    let mut body_short = vec![0, 0, 0, 49];
    body_short.extend_from_slice(&[0_u8; 10]);
    assert!(matches!(
        reader.read_record(&mut Cursor::new(body_short)),
        Err(FrameError::TruncatedBody {
            expected: 49,
            received: 10
        })
    ));

    let mut over_small = vec![0, 0, 0x10, 0x01];
    over_small.extend_from_slice(&[0_u8; 4097]);
    // Kind `13` (Error) is a small record; the kind bytes live at body
    // offset 6 from the start of the record.
    over_small[4 + 6] = 0;
    over_small[4 + 7] = 13;
    assert!(matches!(
        reader.read_record(&mut Cursor::new(over_small)),
        Err(FrameError::SmallRecordTooLarge { actual: 4097 })
    ));

    let mut maximum = vec![0, 4, 0, 0];
    maximum.extend_from_slice(&vec![0_u8; 256 * 1024]);
    assert!(matches!(
        reader.read_record(&mut Cursor::new(maximum)),
        Err(FrameError::Record(ControlError::InvalidMagic))
    ));
}

// ── establishment ───────────────────────────────────────────────────────────

#[test]
fn from_stream_rejects_an_invalid_launch_manifest() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.delegation_purpose = "other-purpose".to_owned();
    let error = HelperControl::from_stream(
        manifest,
        FakeStream::new(),
        TestRegistry::new(1, empty_projection()),
    )
    .expect_err("invalid manifest");
    assert!(matches!(
        error,
        HelperControlError::Manifest(bitty_ai_control::ManifestError::UnsupportedDelegationPurpose)
    ));
}

#[test]
fn establishment_emits_established_then_one_projection_and_no_key_material() {
    let projection = projection(&["alpha", "beta"]);
    let mut harness = Harness::new(7, projection.clone());
    harness.send_core(RecordBody::Establish(harness.echo.clone()));
    harness.control.establish().expect("establish");

    let raw = harness.stream.take_output();
    assert!(
        !contains(&raw, &harness.helper_private_key),
        "helper private key appeared on the control stream"
    );
    assert!(
        !contains(&raw, &harness.core_private_key),
        "core private key appeared on the control stream"
    );
    let records = parse_frames(&raw);
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].sequence, 1);
    assert_eq!(records[1].sequence, 2);
    let established = match &records[0].body {
        RecordBody::Established(body) => body.clone(),
        other => panic!("not Established: {other:?}"),
    };
    let announced = match &records[1].body {
        RecordBody::Projection(body) => body.clone(),
        other => panic!("not Projection: {other:?}"),
    };
    assert_eq!(established.echo, harness.echo);
    assert_eq!(announced.lifetime_id, established.lifetime_id);
    assert_eq!(announced.feed_revision, 7);
    assert_eq!(announced.payload, projection);
    assert_eq!(
        Digest::of(&announced.payload.canonical_bytes()),
        Digest::of(&projection.canonical_bytes())
    );
    assert_eq!(harness.control.lifetime_id(), Some(established.lifetime_id));
    assert_eq!(harness.control.feed_revision(), Some(7));
    assert_eq!(harness.control.proof_state(), ProofState::Unbound);
    assert_eq!(harness.control.publication_version(), 0);
    assert_eq!(harness.control.session_id(), harness.session);
    assert!(!harness.control.is_closed());
    harness.assert_no_output();
}

#[test]
fn establishment_rejects_each_mutable_echo_field() {
    struct Case {
        field: &'static str,
        mutate: fn(&mut EstablishmentEcho),
    }

    let cases = [
        Case {
            field: "core_instance_nonce",
            mutate: |echo| echo.core_instance_nonce = Nonce::new([0x99; 32]),
        },
        Case {
            field: "component_name",
            mutate: |echo| echo.component_name = "SECRET-COMPONENT-NAME".to_owned(),
        },
        Case {
            field: "descriptor_version",
            mutate: |echo| echo.descriptor_version = "9.9.9-secret".to_owned(),
        },
        Case {
            field: "executable_name",
            mutate: |echo| echo.executable_name = "other-executable".to_owned(),
        },
        Case {
            field: "executable_sha256",
            mutate: |echo| echo.executable_sha256 = Digest::of(b"other-executable"),
        },
        Case {
            field: "component_protocol_version",
            mutate: |echo| echo.component_protocol_version = 99,
        },
        Case {
            field: "control_channel_nonce",
            mutate: |echo| echo.control_channel_nonce = Nonce::new([0x99; 32]),
        },
        Case {
            field: "data_channel_nonce",
            mutate: |echo| echo.data_channel_nonce = Nonce::new([0x99; 32]),
        },
        Case {
            field: "core_leaf_digest",
            mutate: |echo| echo.core_leaf_digest = Digest::of(b"other-core-leaf"),
        },
        Case {
            field: "helper_leaf_digest",
            mutate: |echo| echo.helper_leaf_digest = Digest::of(b"other-helper-leaf"),
        },
    ];

    for case in cases {
        let mut harness = Harness::new(1, empty_projection());
        let mut echo = harness.echo.clone();
        (case.mutate)(&mut echo);
        harness.stream.send_record(&core_record(
            1,
            harness.session,
            RecordBody::Establish(echo),
        ));
        let error = harness.control.establish().expect_err("mismatch");
        let text = format!("{error}");
        assert!(
            !text.contains("SECRET-COMPONENT-NAME") && !text.contains("9.9.9-secret"),
            "diagnostic echoed a peer value: {text}"
        );
        expect_establishment(error, case.field);
        harness.assert_closed(RevocationReason::EstablishmentMismatch);
        let records = harness.drain();
        assert_eq!(records.len(), 1, "only the best-effort Error is written");
        assert!(matches!(
            records[0].body,
            RecordBody::Error(ErrorCode::InvalidControl)
        ));
    }
}

#[test]
fn establishment_expectations_cover_fixed_constants_and_session() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let expectations = EstablishmentExpectations::from_manifest(&manifest);
    let mut echo = echo_for(&manifest);
    assert_eq!(expectations.verify(&echo), Ok(()));
    assert_eq!(
        expectations.launch_nonce(),
        SessionId::new(manifest.launch_nonce)
    );

    let cases: [EchoCase; 8] = [
        ("bootstrap_format_version", |echo| {
            echo.bootstrap_format_version = 2;
        }),
        ("control_protocol_version", |echo| {
            echo.control_protocol_version = 2;
        }),
        ("launch_nonce", |echo| {
            echo.launch_nonce = SessionId::new([0x99; 32]);
        }),
        ("delegation_purpose", |echo| {
            echo.delegation_purpose = "other-purpose".to_owned();
        }),
        ("control_profile", |echo| {
            echo.control_profile = "other-profile".to_owned();
        }),
        ("key_material_profile", |echo| {
            echo.key_material_profile = 2;
        }),
        ("key_type", |echo| {
            echo.key_type = 2;
        }),
        ("component_name", |echo| {
            echo.component_name.clear();
        }),
    ];
    for (field, mutate) in cases {
        let mut candidate = echo.clone();
        mutate(&mut candidate);
        assert_eq!(expectations.verify(&candidate), Err(field));
    }

    // Restore a valid echo before the loop above's last mutation is irrelevant.
    echo.component_name = manifest.component_name.clone();
    assert_eq!(expectations.verify(&echo), Ok(()));
}

#[test]
fn establishment_rejects_wrong_sender_direction_and_raw_violations() {
    let mut harness = Harness::new(1, empty_projection());
    // Encoding rejects the illegal direction; splice the sender byte into a
    // valid frame instead.
    let mut raw = frame(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    raw[12] = Sender::Helper.code();
    harness.send_raw_frame(&raw);
    let error = harness.control.establish().expect_err("wrong sender");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, empty_projection());
    let mut raw = frame(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    raw[8..10].copy_from_slice(&2_u16.to_be_bytes());
    harness.send_raw_frame(&raw);
    let error = harness.control.establish().expect_err("wrong version");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, empty_projection());
    let mut raw = frame(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    // Roles follow the two version fields: the body starts after the four
    // prefix bytes and the 49-byte record header; the core role is body
    // offset 4.
    raw[4 + 49 + 4] = 9;
    harness.send_raw_frame(&raw);
    let error = harness.control.establish().expect_err("wrong role");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn establishment_rejects_sequence_and_session_mismatch_after_decode() {
    let mut harness = Harness::new(1, empty_projection());
    harness.stream.send_record(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    harness.control.establish().expect("establish");
    let _ = harness.drain();

    // A Publish with a stale sequence is rejected before body handling.
    let tuple = tuple_for(harness.session, 1, &empty_projection());
    let lifetime_id = harness.control.lifetime_id().unwrap();
    let feed_revision = harness.control.feed_revision().unwrap();
    harness.send_core_with_sequence(
        3,
        RecordBody::Publish(PublishBody {
            lifetime_id,
            feed_revision,
            tuple,
            payload: empty_projection(),
        }),
    );
    let error = harness.control.step().expect_err("sequence mismatch");
    assert!(matches!(error, HelperControlError::SequenceMismatch));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn establishment_rejects_zero_sequence_and_wrong_session() {
    let mut harness = Harness::new(1, empty_projection());
    let mut raw = frame(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    raw[13..21].copy_from_slice(&0_u64.to_be_bytes());
    harness.send_raw_frame(&raw);
    let error = harness.control.establish().expect_err("zero sequence");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, empty_projection());
    harness.stream.send_record(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    harness.control.establish().expect("establish");
    let _ = harness.drain();

    let mut raw = frame(&core_record(
        1,
        SessionId::new([0x99; 32]),
        RecordBody::Invalidate(InvalidateBody {
            lifetime_id: harness.control.lifetime_id().unwrap(),
            feed_revision: harness.control.feed_revision().unwrap(),
            reason: Reason::SourceMutation,
        }),
    ));
    // Keep the body's launch/session-independent fields; flip the header
    // session to a value the state machine did not establish.
    raw[21..53].copy_from_slice(&[0x99; 32]);
    harness.send_raw_frame(&raw);
    let error = harness.control.step().expect_err("session mismatch");
    assert!(matches!(error, HelperControlError::SessionMismatch));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn establishment_requires_a_valid_registry_revision() {
    let registry = TestRegistry::new(0, empty_projection());
    let mut harness = Harness::with_registry(registry);
    harness.stream.send_record(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    let error = harness.control.establish().expect_err("zero revision");
    assert!(matches!(
        error,
        HelperControlError::Registry(RegistryError::InvalidRevision)
    ));
    harness.assert_closed(RevocationReason::RegistryUnavailable);
    let records = harness.drain();
    assert_eq!(records.len(), 1);
    assert!(matches!(
        records[0].body,
        RecordBody::Error(ErrorCode::Unavailable)
    ));
}

#[test]
fn establishment_survives_partial_reads_and_writes() {
    let mut harness = Harness::new(1, projection(&["alpha"]));
    harness.stream.set_read_chunk(1);
    harness.stream.set_write_chunk(1);
    harness.establish();
    harness.control.close();
}

// ── publish / ack / bind / bound ────────────────────────────────────────────

#[test]
fn publish_ack_bind_bound_golden_path() {
    let projection = projection(&["alpha", "beta"]);
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();

    let (tuple, ack) = harness.publish(1, &projection);
    assert_eq!(harness.control.proof_state(), ProofState::Acknowledged);
    assert_eq!(harness.control.publication_version(), 1);

    let bound = harness.bind(tuple, ack);
    assert_eq!(
        harness.control.proof_state(),
        ProofState::Bound {
            receipt: bound.receipt,
        }
    );
    assert_eq!(bound.lifetime_id, harness.control.lifetime_id().unwrap());
    assert_eq!(
        bound.feed_revision,
        harness.control.feed_revision().unwrap()
    );
    assert_eq!(harness.registry.comparisons(), 1);
    assert_eq!(
        harness.registry.last_retained(),
        Some(projection.canonical_bytes())
    );
    harness.assert_no_output();
}

#[test]
fn a_new_publication_replaces_the_previous_receipt() {
    let projection = projection(&["alpha"]);
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let (tuple_one, ack_one) = harness.publish(1, &projection);
    let bound_one = harness.bind(tuple_one, ack_one);

    let (tuple_two, ack_two) = harness.publish(2, &projection);
    assert_eq!(harness.control.proof_state(), ProofState::Acknowledged);
    let bound_two = harness.bind(tuple_two, ack_two);
    assert_ne!(bound_one.receipt, bound_two.receipt);
    assert_eq!(
        harness.control.proof_state(),
        ProofState::Bound {
            receipt: bound_two.receipt,
        }
    );
    assert_eq!(harness.control.publication_version(), 2);
}

#[test]
fn publish_rejects_pending_stale_future_and_wrong_identity() {
    let projection = projection(&["alpha"]);
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let _ = harness.publish(1, &projection);
    // A second Publish before Bind is a simultaneous pipeline.
    let tuple = tuple_for(harness.session, 2, &projection);
    let lifetime_id = harness.control.lifetime_id().unwrap();
    let feed_revision = harness.control.feed_revision().unwrap();
    harness.send_core(RecordBody::Publish(PublishBody {
        lifetime_id,
        feed_revision,
        tuple,
        payload: projection.clone(),
    }));
    let error = harness.control.step().expect_err("pending publication");
    assert!(matches!(
        error,
        HelperControlError::PublicationAlreadyPending
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let lifetime_id = harness.control.lifetime_id().unwrap();
    let feed_revision = harness.control.feed_revision().unwrap();
    harness.send_core(RecordBody::Publish(PublishBody {
        lifetime_id,
        feed_revision: feed_revision + 1,
        tuple: tuple_for(harness.session, 1, &projection),
        payload: projection.clone(),
    }));
    let error = harness.control.step().expect_err("future revision");
    assert!(matches!(error, HelperControlError::StaleAuthority));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    harness.send_core(RecordBody::Publish(PublishBody {
        lifetime_id: LifetimeId::new([0x99; 32]),
        feed_revision: harness.control.feed_revision().unwrap(),
        tuple: tuple_for(harness.session, 1, &projection),
        payload: projection.clone(),
    }));
    let error = harness.control.step().expect_err("wrong lifetime");
    assert!(matches!(error, HelperControlError::StaleAuthority));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let wrong_id = PublicationId::from_parts(SessionId::new([0x99; 32]), 1);
    let wrong_tuple =
        Tuple::new(wrong_id, 1, Digest::of(&projection.canonical_bytes())).expect("tuple");
    harness.send_core(RecordBody::Publish(PublishBody {
        lifetime_id: harness.control.lifetime_id().unwrap(),
        feed_revision: harness.control.feed_revision().unwrap(),
        tuple: wrong_tuple,
        payload: projection,
    }));
    let error = harness.control.step().expect_err("wrong launch identity");
    assert!(matches!(
        error,
        HelperControlError::PublicationIdentityMismatch
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn publish_version_must_strictly_increase_and_never_wrap() {
    let projection = projection(&["alpha"]);
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let (tuple, ack) = harness.publish(1, &projection);
    let _ = harness.bind(tuple, ack);

    let lifetime_id = harness.control.lifetime_id().unwrap();
    let feed_revision = harness.control.feed_revision().unwrap();
    harness.send_core(RecordBody::Publish(PublishBody {
        lifetime_id,
        feed_revision,
        tuple: tuple_for(harness.session, 1, &projection),
        payload: projection.clone(),
    }));
    let error = harness.control.step().expect_err("repeated version");
    assert!(matches!(
        error,
        HelperControlError::PublicationVersionNotIncreasing
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let (tuple, ack) = harness.publish(u64::MAX, &projection);
    let _ = harness.bind(tuple, ack);
    let lifetime_id = harness.control.lifetime_id().unwrap();
    let feed_revision = harness.control.feed_revision().unwrap();
    harness.send_core(RecordBody::Publish(PublishBody {
        lifetime_id,
        feed_revision,
        tuple: tuple_for(harness.session, u64::MAX, &projection),
        payload: projection,
    }));
    let error = harness.control.step().expect_err("version wrap");
    assert!(matches!(
        error,
        HelperControlError::PublicationVersionNotIncreasing
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn bind_rejects_mismatch_duplicate_and_absent_acknowledgement() {
    let projection = projection(&["alpha"]);
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    harness.send_core(RecordBody::Bind(AckBody {
        lifetime_id: harness.control.lifetime_id().unwrap(),
        feed_revision: harness.control.feed_revision().unwrap(),
        tuple: tuple_for(harness.session, 1, &projection),
        ack_id: AckId::new([0x99; 32]),
    }));
    let error = harness.control.step().expect_err("bind before publish");
    assert!(matches!(error, HelperControlError::NoPendingBind));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let (_tuple, ack) = harness.publish(1, &projection);
    let mut wrong = ack;
    wrong.ack_id = AckId::new([0x99; 32]);
    harness.send_core(RecordBody::Bind(wrong));
    let error = harness.control.step().expect_err("wrong ack");
    assert!(matches!(error, HelperControlError::BindMismatch));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let (tuple, ack) = harness.publish(1, &projection);
    let _ = harness.bind(tuple, ack);
    harness.send_core(RecordBody::Bind(ack));
    let error = harness.control.step().expect_err("duplicate bind");
    assert!(matches!(error, HelperControlError::NoPendingBind));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn mutation_between_ack_and_bind_refuses_without_a_bound() {
    let initial = projection(&["alpha"]);
    let mut harness = Harness::new(1, initial.clone());
    harness.establish();
    let (_tuple, ack) = harness.publish(1, &initial);

    harness.registry.mutate(projection(&["gamma"]));
    harness.send_core(RecordBody::Bind(ack));
    let error = harness.control.step().expect_err("registry moved");
    assert!(matches!(error, HelperControlError::RetainedBytesMismatch));
    harness.assert_closed(RevocationReason::RetainedBytesMismatch);
    let records = harness.drain();
    assert!(
        records
            .iter()
            .all(|record| !matches!(record.body, RecordBody::Bound(_)))
    );
    assert_eq!(harness.registry.comparisons(), 1);

    let mut harness = Harness::new(1, initial.clone());
    harness.establish();
    let (_, ack) = harness.publish(1, &initial);
    harness
        .registry
        .replace_projection_without_revision(projection(&["delta"]));
    harness.send_core(RecordBody::Bind(ack));
    let error = harness.control.step().expect_err("bytes moved");
    assert!(matches!(error, HelperControlError::RetainedBytesMismatch));
    harness.assert_closed(RevocationReason::RetainedBytesMismatch);

    let mut harness = Harness::new(1, initial.clone());
    harness.establish();
    let (_, ack) = harness.publish(1, &initial);
    harness.registry.set_fail_compare(true);
    harness.send_core(RecordBody::Bind(ack));
    let error = harness.control.step().expect_err("registry failure");
    assert!(matches!(
        error,
        HelperControlError::Registry(RegistryError::Unavailable)
    ));
    harness.assert_closed(RevocationReason::RegistryUnavailable);
}

#[test]
fn publish_rejects_payload_digest_and_canonical_violations() {
    let initial = projection(&["alpha"]);
    let mut harness = Harness::new(1, initial.clone());
    harness.establish();
    let lifetime_id = harness.control.lifetime_id().unwrap();
    let feed_revision = harness.control.feed_revision().unwrap();
    let tuple = tuple_for(harness.session, 1, &initial);
    let canonical = initial.canonical_bytes();
    let record = core_record(
        2,
        harness.session,
        RecordBody::Publish(PublishBody {
            lifetime_id,
            feed_revision,
            tuple,
            payload: initial,
        }),
    );
    let mut raw = frame(&record);
    let payload_start = raw
        .windows(canonical.len())
        .rposition(|window| window == canonical)
        .expect("payload present");
    raw[payload_start] = b' ';
    harness.send_raw_frame(&raw);
    let error = harness.control.step().expect_err("digest mismatch");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, projection(&["alpha"]));
    harness.establish();
    let lifetime_id = harness.control.lifetime_id().unwrap();
    let feed_revision = harness.control.feed_revision().unwrap();
    let initial = projection(&["alpha"]);
    let tuple = tuple_for(harness.session, 1, &initial);
    let canonical = initial.canonical_bytes();
    let record = core_record(
        2,
        harness.session,
        RecordBody::Publish(PublishBody {
            lifetime_id,
            feed_revision,
            tuple,
            payload: initial,
        }),
    );
    let mut raw = frame(&record);
    let payload_start = raw
        .windows(canonical.len())
        .rposition(|window| window == canonical)
        .expect("payload present");
    raw[payload_start] = b' ';
    let mut mutated = canonical.clone();
    mutated[0] = b' ';
    let digest = Digest::of(&mutated);
    // Tuple digest is the 32 bytes after the 40-byte publication identity
    // and before the payload length.
    let tuple_digest_start = payload_start
        .checked_sub(4)
        .and_then(|end| end.checked_sub(32))
        .expect("tuple digest");
    raw[tuple_digest_start..tuple_digest_start + 32].copy_from_slice(digest.as_bytes());
    harness.send_raw_frame(&raw);
    let error = harness.control.step().expect_err("non-canonical payload");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

// ── invalidation and rotation ───────────────────────────────────────────────

#[test]
fn core_invalidation_rotates_same_revision_and_keeps_one_predecessor() {
    let projection = projection(&["alpha"]);
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let first = harness.control.lifetime_id().expect("first lifetime");
    let (tuple, ack) = harness.publish(1, &projection);
    let _ = harness.bind(tuple, ack);

    harness.send_invalidate(first, 1, Reason::SourceMutation);
    let step = harness.control.step().expect("core invalidation");
    let second = harness.control.lifetime_id().expect("second lifetime");
    assert_ne!(first, second);
    assert_eq!(
        step,
        Step::LifetimeRotated {
            previous: first,
            current: second,
            feed_revision: 1,
            reason: Reason::SourceMutation,
        }
    );
    let records = harness.drain();
    assert_eq!(records.len(), 2, "no helper Invalidate for Core origin");
    assert_eq!(records[0].sequence, 5);
    assert!(
        matches!(records[0].body, RecordBody::Lifetime(LifetimeBody {
        lifetime_id,
        feed_revision: 1,
    }) if lifetime_id == second)
    );
    assert!(matches!(records[1].body, RecordBody::Projection(ref body)
        if body.lifetime_id == second && body.feed_revision == 1));
    assert_eq!(harness.control.proof_state(), ProofState::Unbound);

    // A fresh Core publication works on the new lifetime.
    let (tuple, ack) = harness.publish(2, &projection);
    let _ = harness.bind(tuple, ack);

    // The one retired predecessor may be correlated once.
    harness.send_invalidate(first, 1, Reason::SourceMutation);
    assert_eq!(
        harness.control.step().expect("crossed predecessor"),
        Step::PredecessorCorrelated
    );
    harness.assert_no_output();

    // A second duplicate is refused.
    harness.send_invalidate(first, 1, Reason::SourceMutation);
    let error = harness.control.step().expect_err("duplicate predecessor");
    assert!(matches!(error, HelperControlError::StaleAuthority));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn core_reset_and_supersession_rotate_at_the_same_feed_revision() {
    for reason in [Reason::Reset, Reason::Supersession] {
        let projection = projection(&["alpha"]);
        let mut harness = Harness::new(1, projection.clone());
        harness.establish();
        let first = harness.control.lifetime_id().expect("first lifetime");
        let _ = harness.drain();

        harness.send_invalidate(first, 1, reason);
        let step = harness.control.step().expect("core rotation");
        let second = harness.control.lifetime_id().expect("second lifetime");
        assert_ne!(first, second);
        assert_eq!(
            step,
            Step::LifetimeRotated {
                previous: first,
                current: second,
                feed_revision: 1,
                reason,
            }
        );
        assert_eq!(harness.control.feed_revision(), Some(1));
        let records = harness.drain();
        assert_eq!(records.len(), 2);
        assert!(
            matches!(records[0].body, RecordBody::Lifetime(LifetimeBody {
            lifetime_id,
            feed_revision: 1,
        }) if lifetime_id == second)
        );
        assert!(matches!(records[1].body, RecordBody::Projection(ref body)
            if body.lifetime_id == second && body.feed_revision == 1));
    }
}

#[test]
fn core_grant_loss_and_shutdown_close_without_a_new_lifetime() {
    for reason in [Reason::GrantLoss, Reason::Shutdown] {
        let mut harness = Harness::new(1, empty_projection());
        harness.establish();
        let lifetime = harness.control.lifetime_id().expect("lifetime");
        harness.send_invalidate(lifetime, 1, reason);
        assert_eq!(
            harness.control.step().expect("withdrawal"),
            Step::AuthorityClosed { reason }
        );
        harness.assert_closed(RevocationReason::PeerWithdrewAuthority);
        harness.assert_no_output();
    }
}

#[test]
fn helper_reset_emits_invalidate_lifetime_projection_and_correlates_crossed_core() {
    let projection = projection(&["alpha"]);
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let first = harness.control.lifetime_id().expect("lifetime");
    let (tuple, ack) = harness.publish(1, &projection);
    let _ = harness.bind(tuple, ack);

    let step = harness.control.reset().expect("reset");
    let second = harness.control.lifetime_id().expect("second lifetime");
    assert_ne!(first, second);
    assert_eq!(
        step,
        Step::LifetimeRotated {
            previous: first,
            current: second,
            feed_revision: 1,
            reason: Reason::Reset,
        }
    );
    let records = harness.drain();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].sequence, 5);
    assert_eq!(records[1].sequence, 6);
    assert_eq!(records[2].sequence, 7);
    assert!(
        matches!(records[0].body, RecordBody::Invalidate(InvalidateBody {
        lifetime_id,
        feed_revision: 1,
        reason: Reason::Reset,
    }) if lifetime_id == first)
    );
    assert!(
        matches!(records[1].body, RecordBody::Lifetime(LifetimeBody {
        lifetime_id,
        feed_revision: 1,
    }) if lifetime_id == second)
    );
    assert!(matches!(records[2].body, RecordBody::Projection(ref body)
        if body.lifetime_id == second && body.feed_revision == 1));
    assert_eq!(harness.control.proof_state(), ProofState::Unbound);

    harness.send_invalidate(first, 1, Reason::SourceMutation);
    assert_eq!(
        harness.control.step().expect("crossed predecessor"),
        Step::PredecessorCorrelated
    );
    harness.assert_no_output();
}

#[test]
fn reset_after_unannounced_registry_change_closes_after_invalidate() {
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    harness.registry.set_revision(2);
    let error = harness.control.reset().expect_err("stale registry");
    assert!(matches!(error, HelperControlError::StaleAuthority));
    harness.assert_closed(RevocationReason::ProtocolViolation);
    let records = harness.drain();
    assert!(matches!(records[0].body, RecordBody::Invalidate(_)));
    assert!(
        records
            .iter()
            .all(|record| !matches!(record.body, RecordBody::Lifetime(_)))
    );
}

#[test]
fn reset_after_registry_failure_closes_after_invalidate() {
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    harness.registry.set_fail_snapshot(true);
    let error = harness.control.reset().expect_err("registry failure");
    assert!(matches!(
        error,
        HelperControlError::Registry(RegistryError::Unavailable)
    ));
    harness.assert_closed(RevocationReason::RegistryUnavailable);
    let records = harness.drain();
    assert!(matches!(records[0].body, RecordBody::Invalidate(_)));
    assert!(
        records
            .iter()
            .all(|record| !matches!(record.body, RecordBody::Lifetime(_)))
    );
}

#[test]
fn registry_mutation_orders_invalidate_before_mutation_and_advances_revision() {
    let initial = projection(&["alpha"]);
    let next = projection(&["beta"]);
    let mut harness = Harness::new(1, initial.clone());
    harness.establish();
    let first = harness.control.lifetime_id().expect("lifetime");
    let stream = harness.stream.clone();
    let next_for_closure = next.clone();
    let mut observed = None;
    let step = harness
        .control
        .registry_mutated(|registry| {
            observed = Some(parse_frames(&stream.take_output()));
            registry.mutate(next_for_closure);
            Ok(())
        })
        .expect("registry mutation");
    let second = harness.control.lifetime_id().expect("second lifetime");
    assert_ne!(first, second);
    assert_eq!(harness.control.feed_revision(), Some(2));
    assert_eq!(
        step,
        Step::LifetimeRotated {
            previous: first,
            current: second,
            feed_revision: 2,
            reason: Reason::RegistryMutation,
        }
    );
    let observed = observed.expect("captured predecessor invalidation");
    assert_eq!(observed.len(), 1);
    assert!(
        matches!(observed[0].body, RecordBody::Invalidate(InvalidateBody {
        lifetime_id,
        feed_revision: 1,
        reason: Reason::RegistryMutation,
    }) if lifetime_id == first)
    );

    let records = harness.drain();
    assert_eq!(records.len(), 2);
    assert!(
        matches!(records[0].body, RecordBody::Lifetime(LifetimeBody {
        lifetime_id,
        feed_revision: 2,
    }) if lifetime_id == second)
    );
    assert!(matches!(records[1].body, RecordBody::Projection(ref body)
        if body.lifetime_id == second
            && body.feed_revision == 2
            && body.payload == next));
    assert_eq!(harness.control.proof_state(), ProofState::Unbound);

    let (tuple, ack) = harness.publish(2, &next);
    let _ = harness.bind(tuple, ack);
}

#[test]
fn registry_mutation_failures_close_before_announcing_a_lifetime() {
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let error = harness
        .control
        .registry_mutated(|_| Err(RegistryError::Unavailable))
        .expect_err("mutation failure");
    assert!(matches!(
        error,
        HelperControlError::Registry(RegistryError::Unavailable)
    ));
    harness.assert_closed(RevocationReason::RegistryUnavailable);
    let records = harness.drain();
    assert!(matches!(records[0].body, RecordBody::Invalidate(_)));
    assert!(
        records
            .iter()
            .all(|record| !matches!(record.body, RecordBody::Lifetime(_)))
    );

    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let error = harness
        .control
        .registry_mutated(|_| Ok(()))
        .expect_err("missing revision step");
    assert!(matches!(
        error,
        HelperControlError::Registry(RegistryError::InvalidRevision)
    ));
    harness.assert_closed(RevocationReason::RegistryUnavailable);

    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let error = harness
        .control
        .registry_mutated(|registry| {
            registry.set_revision(3);
            Ok(())
        })
        .expect_err("revision jump");
    assert!(matches!(
        error,
        HelperControlError::Registry(RegistryError::InvalidRevision)
    ));
    harness.assert_closed(RevocationReason::RegistryUnavailable);
}

#[test]
fn registry_mutation_revision_overflow_is_refused_before_mutation() {
    let mut harness = Harness::new(u64::MAX, empty_projection());
    harness.establish();
    let called = Cell::new(false);
    let error = harness
        .control
        .registry_mutated(|_| {
            called.set(true);
            Ok(())
        })
        .expect_err("revision overflow");
    assert!(matches!(error, HelperControlError::VersionExhausted));
    assert!(!called.get());
    harness.assert_closed(RevocationReason::VersionExhausted);
    let records = harness.drain();
    assert_eq!(records.len(), 1);
    assert!(matches!(
        records[0].body,
        RecordBody::Error(ErrorCode::ResourceLimit)
    ));
}

#[test]
fn outrun_rotation_window_closes_control() {
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let first = harness.control.lifetime_id().expect("first");
    let _ = harness.control.reset().expect("reset one");
    let _ = harness.drain();
    let second = harness.control.lifetime_id().expect("second");

    harness.send_invalidate(second, 1, Reason::SourceMutation);
    let _ = harness.control.step().expect("rotate from second");
    let _ = harness.drain();
    let third = harness.control.lifetime_id().expect("third");
    assert_ne!(second, third);

    harness.send_invalidate(first, 1, Reason::SourceMutation);
    let error = harness.control.step().expect_err("older predecessor");
    assert!(matches!(error, HelperControlError::StaleAuthority));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

// ── unsupported / malformed / fail-closed ───────────────────────────────────

#[test]
fn admit_is_unsupported_and_strict_stays_disabled() {
    let projection = empty_projection();
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let (tuple, ack) = harness.publish(1, &projection);
    let bound = harness.bind(tuple, ack);

    harness.send_core(RecordBody::Admit(AdmitBody {
        lifetime_id: harness.control.lifetime_id().unwrap(),
        feed_revision: harness.control.feed_revision().unwrap(),
        tuple,
        receipt: bound.receipt,
        request_id: 1,
        data_frame_digest: Digest::of(b"strict-frame"),
    }));
    let error = harness.control.step().expect_err("admit unsupported");
    assert!(matches!(
        error,
        HelperControlError::UnexpectedRecord { kind: Kind::Admit }
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);
    let records = harness.drain();
    assert_eq!(records.len(), 1);
    assert!(matches!(
        records[0].body,
        RecordBody::Error(ErrorCode::InvalidControl)
    ));
}

#[test]
fn wrong_direction_and_unknown_wire_values_revoke() {
    // Helper-origin record presented by Core.
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    let lifetime = harness.control.lifetime_id().unwrap();
    let ack = AckBody {
        lifetime_id: lifetime,
        feed_revision: 1,
        tuple: tuple_for(harness.session, 1, &empty_projection()),
        ack_id: AckId::new([0x42; 32]),
    };
    let mut raw = frame(&ControlRecord {
        sender: Sender::Helper,
        sequence: 2,
        session_id: harness.session,
        body: RecordBody::Ack(ack),
    });
    raw[12] = Sender::Core.code();
    harness.send_raw_frame(&raw);
    let error = harness.control.step().expect_err("wrong direction");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);

    // Unknown kind, reserved kind, unknown version, unknown sender.
    for (offset, value, _label) in [
        (10_usize, 14_u16, "kind"),
        (10, 9, "reserved"),
        (8, 2, "version"),
    ] {
        let mut harness = Harness::new(1, empty_projection());
        let mut raw = frame(&core_record(
            1,
            harness.session,
            RecordBody::Establish(harness.echo.clone()),
        ));
        raw[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
        harness.send_raw_frame(&raw);
        let error = harness.control.establish().expect_err("unknown value");
        expect_record_error(error);
        harness.assert_closed(RevocationReason::ProtocolViolation);
    }

    let mut harness = Harness::new(1, empty_projection());
    let mut raw = frame(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    raw[12] = 3;
    harness.send_raw_frame(&raw);
    let error = harness.control.establish().expect_err("unknown sender");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);

    // Unknown reason on an authenticated Invalidate.
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    let mut raw = frame(&core_record(
        2,
        harness.session,
        RecordBody::Invalidate(InvalidateBody {
            lifetime_id: harness.control.lifetime_id().unwrap(),
            feed_revision: 1,
            reason: Reason::SourceMutation,
        }),
    ));
    raw[4 + 49 + 32 + 8] = 7;
    harness.send_raw_frame(&raw);
    let error = harness.control.step().expect_err("unknown reason");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);

    // Unknown error code on an authenticated Error record.
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    let mut raw = frame(&core_record(
        2,
        harness.session,
        RecordBody::Error(ErrorCode::InvalidControl),
    ));
    raw[4 + 49] = 5;
    harness.send_raw_frame(&raw);
    let error = harness.control.step().expect_err("unknown error code");
    expect_record_error(error);
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn helper_sender_inbound_invalidate_and_error_are_refused() {
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    let lifetime = harness.control.lifetime_id().expect("lifetime");
    harness.stream.send_record(&ControlRecord {
        sender: Sender::Helper,
        sequence: 2,
        session_id: harness.session,
        body: RecordBody::Invalidate(InvalidateBody {
            lifetime_id: lifetime,
            feed_revision: 1,
            reason: Reason::SourceMutation,
        }),
    });
    let error = harness
        .control
        .step()
        .expect_err("helper sender invalidate");
    assert!(matches!(
        error,
        HelperControlError::UnexpectedSender {
            sender: Sender::Helper
        }
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    harness.stream.send_record(&ControlRecord {
        sender: Sender::Helper,
        sequence: 2,
        session_id: harness.session,
        body: RecordBody::Error(ErrorCode::InvalidControl),
    });
    let error = harness.control.step().expect_err("helper sender error");
    assert!(matches!(
        error,
        HelperControlError::UnexpectedSender {
            sender: Sender::Helper
        }
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn core_registry_mutation_invalidation_is_refused() {
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    let lifetime = harness.control.lifetime_id().expect("lifetime");
    harness.send_invalidate(lifetime, 1, Reason::RegistryMutation);
    let error = harness.control.step().expect_err("core registry mutation");
    assert!(matches!(error, HelperControlError::CoreRegistryMutation));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn failed_error_write_does_not_delay_authority_clearing() {
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    let lifetime = harness.control.lifetime_id().expect("lifetime");
    harness.stream.fail_writes_after(Some(0));
    harness.send_invalidate(lifetime, 1, Reason::RegistryMutation);
    let error = harness.control.step().expect_err("core registry mutation");
    assert!(matches!(error, HelperControlError::CoreRegistryMutation));
    harness.assert_closed(RevocationReason::ProtocolViolation);
    assert!(harness.stream.output_is_empty());
}

#[test]
fn peer_error_is_terminal_without_an_error_reply() {
    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    harness.send_core(RecordBody::Error(ErrorCode::Unavailable));
    assert_eq!(
        harness.control.step().expect("peer error"),
        Step::PeerError {
            code: ErrorCode::Unavailable
        }
    );
    harness.assert_closed(RevocationReason::PeerError);
    harness.assert_no_output();
}

#[test]
fn control_eof_and_truncated_frames_close_without_output() {
    let mut harness = Harness::new(1, empty_projection());
    let error = harness.control.establish().expect_err("eof");
    assert!(matches!(
        error,
        HelperControlError::Frame(FrameError::Closed)
    ));
    harness.assert_closed(RevocationReason::ControlTransport);
    harness.assert_no_output();

    let mut harness = Harness::new(1, empty_projection());
    harness.stream.send_bytes(&[0, 0]);
    let error = harness.control.establish().expect_err("truncated prefix");
    assert!(matches!(
        error,
        HelperControlError::Frame(FrameError::TruncatedPrefix { received: 2 })
    ));
    harness.assert_closed(RevocationReason::ControlTransport);

    let mut harness = Harness::new(1, empty_projection());
    harness.stream.send_bytes(&[0, 0, 0, 49, 1, 2, 3, 4, 5]);
    let error = harness.control.establish().expect_err("truncated body");
    assert!(matches!(
        error,
        HelperControlError::Frame(FrameError::TruncatedBody {
            expected: 49,
            received: 5
        })
    ));
    harness.assert_closed(RevocationReason::ControlTransport);

    let mut harness = Harness::new(1, empty_projection());
    harness.stream.send_bytes(&[0, 4, 0, 1]);
    let error = harness.control.establish().expect_err("oversize frame");
    assert!(matches!(
        error,
        HelperControlError::Frame(FrameError::FrameTooLarge)
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, empty_projection());
    harness.stream.send_bytes(&[0, 0, 0, 0]);
    let error = harness.control.establish().expect_err("zero frame");
    assert!(matches!(
        error,
        HelperControlError::Frame(FrameError::ZeroLength)
    ));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}

#[test]
fn write_failures_close_without_retrying_an_old_record() {
    let mut harness = Harness::new(1, empty_projection());
    harness.stream.send_record(&core_record(
        1,
        harness.session,
        RecordBody::Establish(harness.echo.clone()),
    ));
    harness.stream.fail_writes_after(Some(0));
    let error = harness.control.establish().expect_err("write failure");
    assert!(matches!(
        error,
        HelperControlError::Frame(FrameError::Io(_))
    ));
    harness.assert_closed(RevocationReason::ControlTransport);
    harness.assert_no_output();
    assert_eq!(harness.stream.written(), 0);

    let mut harness = Harness::new(1, empty_projection());
    harness.establish();
    let _ = harness.drain();
    harness.stream.fail_writes_after(Some(0));
    let error = harness
        .control
        .reset()
        .expect_err("invalidation write failure");
    assert!(matches!(
        error,
        HelperControlError::Frame(FrameError::Io(_))
    ));
    harness.assert_closed(RevocationReason::ControlTransport);
    harness.assert_no_output();
}

#[test]
fn step_before_establish_is_refused_without_consuming_input() {
    let mut harness = Harness::new(1, empty_projection());
    let error = harness.control.step().expect_err("not established");
    assert!(matches!(error, HelperControlError::NotEstablished));
    assert!(!harness.control.is_closed());
    harness.establish();
}

#[test]
fn establishment_cannot_repeat_and_close_is_terminal() {
    let projection = empty_projection();
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let error = harness
        .control
        .establish()
        .expect_err("already established");
    assert!(matches!(error, HelperControlError::AlreadyEstablished));
    harness.assert_closed(RevocationReason::ProtocolViolation);

    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let (tuple, ack) = harness.publish(1, &projection);
    let _ = harness.bind(tuple, ack);
    harness.control.close();
    assert_eq!(harness.control.proof_state(), ProofState::Revoked);
    assert_eq!(
        harness.control.revocation_reason(),
        Some(RevocationReason::LocalClose)
    );
    assert!(harness.control.lifetime_id().is_none());
    assert!(harness.control.feed_revision().is_none());
    assert_eq!(harness.control.publication_version(), 1);
    harness.control.close();
    assert!(matches!(
        harness.control.step().unwrap_err(),
        HelperControlError::Closed
    ));
    assert!(matches!(
        harness.control.reset().unwrap_err(),
        HelperControlError::Closed
    ));
    assert!(matches!(
        harness.control.registry_mutated(|_| Ok(())).unwrap_err(),
        HelperControlError::Closed
    ));
    assert!(matches!(
        harness.control.establish().unwrap_err(),
        HelperControlError::Closed
    ));
    harness.assert_no_output();
}

#[test]
fn diagnostics_never_echo_payload_or_identifiers() {
    let projection = projection(&["sensitive-model-name"]);
    let mut harness = Harness::new(1, projection.clone());
    harness.establish();
    let (tuple, ack) = harness.publish(1, &projection);
    let lifetime = harness.control.lifetime_id().unwrap();
    let mut wrong = ack;
    wrong.ack_id = AckId::new([0xab; 32]);
    harness.send_core(RecordBody::Bind(wrong));
    let error = harness.control.step().expect_err("bind mismatch");
    let text = format!("{error}");
    assert!(!text.contains("sensitive-model-name"));
    assert!(!text.contains(&tuple.publication_id.to_hex()));
    assert!(!text.contains(&lifetime.to_hex()));
    assert!(!text.contains(&wrong.ack_id.to_hex()));
    assert!(!text.contains(&harness.session.to_hex()));
    assert!(!text.contains('/'));
    assert!(text.contains("bind"));
    let debug = format!("{:?}", harness.control);
    assert!(!debug.contains(&lifetime.to_hex()));
    assert!(!debug.contains(&harness.session.to_hex()));
    assert!(!debug.contains(&tuple.publication_id.to_hex()));
    harness.assert_closed(RevocationReason::ProtocolViolation);
}
