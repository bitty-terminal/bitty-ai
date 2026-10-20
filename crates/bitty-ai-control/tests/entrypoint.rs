//! Library-level launch-entrypoint tests.
//!
//! The tests never touch a real descriptor: bootstrap is an in-memory reader,
//! control is a two-way byte channel, and the mutual TLS session runs over the
//! checked-in certificate fixtures. The helper runs on a test-only thread
//! because a blocking TLS duplex needs both sides at once; the library itself
//! creates no thread.

mod common;

use std::collections::BTreeSet;
use std::io::{self, Cursor, Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::Duration;

use bitty_ai_control::{
    CHILD_BOOTSTRAP_DESCRIPTOR, CHILD_CONTROL_DESCRIPTOR, CONTROL_ALPN, ControlOnlyRegistry,
    FrameReader, FrameWriter, HelperExit, run_helper, run_helper_on_platform,
};
use bitty_ipc_api::provider_publication::{
    AckBody, BoundBody, ControlRecord, Digest, InvalidateBody, ProviderRoutingProjection,
    PublicationId, PublishBody, Reason, RecordBody, Sender as RecordSender, SessionId, Tuple,
};
use common::{
    client_config, client_connection, establishment_echo, fixtures, valid_manifest,
    valid_manifest_bytes,
};
use rustls::ClientConnection;

/// Bound every blocking test read so a protocol regression fails instead of
/// hanging the suite. Test-only policy; the library owns no clock.
const TEST_IO_TIMEOUT: Duration = Duration::from_secs(20);

// ── in-memory duplex channel (test only) ────────────────────────────────────

struct ChannelReader {
    receiver: Receiver<Vec<u8>>,
    pending: Vec<u8>,
    offset: usize,
}

impl ChannelReader {
    fn new(receiver: Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            pending: Vec::new(),
            offset: 0,
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        while self.offset == self.pending.len() {
            match self.receiver.recv_timeout(TEST_IO_TIMEOUT) {
                Ok(chunk) => {
                    self.pending = chunk;
                    self.offset = 0;
                }
                Err(RecvTimeoutError::Disconnected) => return Ok(0),
                Err(RecvTimeoutError::Timeout) => {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "test duplex timed out",
                    ));
                }
            }
        }
        let count = buffer.len().min(self.pending.len() - self.offset);
        buffer[..count].copy_from_slice(&self.pending[self.offset..self.offset + count]);
        self.offset += count;
        Ok(count)
    }
}

struct ChannelWriter {
    sender: Sender<Vec<u8>>,
}

impl Write for ChannelWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.sender
            .send(buffer.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "test duplex closed"))?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct ChannelEnd {
    reader: ChannelReader,
    writer: ChannelWriter,
}

impl Read for ChannelEnd {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buffer)
    }
}

impl Write for ChannelEnd {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.writer.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

/// Build two connected in-memory duplex ends: `(client, helper)`.
fn channel_pair() -> (ChannelEnd, ChannelEnd) {
    let (client_writer, helper_reader) = mpsc::channel();
    let (helper_writer, client_reader) = mpsc::channel();
    (
        ChannelEnd {
            reader: ChannelReader::new(client_reader),
            writer: ChannelWriter {
                sender: client_writer,
            },
        },
        ChannelEnd {
            reader: ChannelReader::new(helper_reader),
            writer: ChannelWriter {
                sender: helper_writer,
            },
        },
    )
}

// ── test client over the in-memory TLS session ──────────────────────────────

struct TestClient {
    connection: ClientConnection,
    io: ChannelEnd,
    sequence: u64,
}

impl TestClient {
    fn with_io(io: ChannelEnd) -> Self {
        let fixtures = fixtures();
        Self {
            connection: client_connection(client_config(&fixtures, CONTROL_ALPN, false)),
            io,
            sequence: 0,
        }
    }

    fn complete_handshake(&mut self) {
        for _ in 0..64 {
            while self.connection.wants_write() {
                self.connection
                    .write_tls(&mut self.io)
                    .expect("client TLS write");
            }
            if !self.connection.is_handshaking() {
                break;
            }
            let read = self
                .connection
                .read_tls(&mut self.io)
                .expect("client TLS read");
            assert!(read > 0, "helper closed during the TLS handshake");
            self.connection
                .process_new_packets()
                .expect("client TLS processing");
        }
        assert!(
            !self.connection.is_handshaking(),
            "client TLS handshake did not complete"
        );
        assert_eq!(self.connection.alpn_protocol(), Some(CONTROL_ALPN));
        while self.connection.wants_write() {
            self.connection
                .write_tls(&mut self.io)
                .expect("client TLS write");
        }
    }

    fn send(&mut self, session: SessionId, body: RecordBody) {
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("test core sequence is bounded");
        let record = ControlRecord {
            sender: RecordSender::Core,
            sequence: self.sequence,
            session_id: session,
            body,
        };
        let mut writer = FrameWriter::new();
        writer.enqueue(&record).expect("client record encodes");
        let mut stream = rustls::Stream::new(&mut self.connection, &mut self.io);
        writer.flush(&mut stream).expect("client record flushes");
    }

    fn read(&mut self) -> ControlRecord {
        let mut reader = FrameReader::new();
        let mut stream = rustls::Stream::new(&mut self.connection, &mut self.io);
        reader
            .read_record(&mut stream)
            .expect("helper record reads")
    }
}

fn empty_projection() -> ProviderRoutingProjection {
    ProviderRoutingProjection::empty()
}

fn tuple_for(session: SessionId, version: u64) -> Tuple {
    let projection = empty_projection();
    Tuple::new(
        PublicationId::from_parts(session, version),
        version,
        Digest::of(&projection.canonical_bytes()),
    )
    .expect("empty-projection tuple")
}

fn established_body(record: ControlRecord) -> bitty_ipc_api::provider_publication::EstablishedBody {
    match record.body {
        RecordBody::Established(body) => body,
        other => panic!("expected Established, got {other:?}"),
    }
}

fn projection_body(record: ControlRecord) -> bitty_ipc_api::provider_publication::ProjectionBody {
    match record.body {
        RecordBody::Projection(body) => body,
        other => panic!("expected Projection, got {other:?}"),
    }
}

fn ack_body(record: ControlRecord) -> AckBody {
    match record.body {
        RecordBody::Ack(body) => body,
        other => panic!("expected Ack, got {other:?}"),
    }
}

fn bound_body(record: ControlRecord) -> BoundBody {
    match record.body {
        RecordBody::Bound(body) => body,
        other => panic!("expected Bound, got {other:?}"),
    }
}

// ── establishment and the publication sequence over TLS ─────────────────────

#[test]
fn establishment_over_in_memory_tls_announces_the_empty_projection() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let echo = establishment_echo(&manifest);
    let session = SessionId::new(manifest.launch_nonce);
    let bootstrap = valid_manifest_bytes(&fixtures).to_vec();
    let (client_io, helper_io) = channel_pair();

    let helper = thread::spawn(move || {
        run_helper(
            Cursor::new(bootstrap),
            helper_io,
            ControlOnlyRegistry::new(),
        )
    });

    let mut client = TestClient::with_io(client_io);
    client.complete_handshake();
    client.send(session, RecordBody::Establish(echo.clone()));

    let established = established_body(client.read());
    assert_eq!(established.echo, echo);
    let announced = projection_body(client.read());
    assert_eq!(announced.lifetime_id, established.lifetime_id);
    assert_eq!(announced.feed_revision, 1);
    assert!(announced.payload.providers().is_empty());

    client.send(
        session,
        RecordBody::Invalidate(InvalidateBody {
            lifetime_id: established.lifetime_id,
            feed_revision: 1,
            reason: Reason::GrantLoss,
        }),
    );
    assert_eq!(
        helper.join().expect("helper thread joined"),
        HelperExit::AuthorityClosed
    );
}

#[test]
fn publish_ack_bind_bound_with_the_empty_control_registry() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let echo = establishment_echo(&manifest);
    let session = SessionId::new(manifest.launch_nonce);
    let bootstrap = valid_manifest_bytes(&fixtures).to_vec();
    let (client_io, helper_io) = channel_pair();

    let helper = thread::spawn(move || {
        run_helper(
            Cursor::new(bootstrap),
            helper_io,
            ControlOnlyRegistry::new(),
        )
    });

    let mut client = TestClient::with_io(client_io);
    client.complete_handshake();
    client.send(session, RecordBody::Establish(echo));

    let established = established_body(client.read());
    let _announced = projection_body(client.read());
    let lifetime_id = established.lifetime_id;
    let feed_revision = 1_u64;

    let tuple = tuple_for(session, 1);
    client.send(
        session,
        RecordBody::Publish(PublishBody {
            lifetime_id,
            feed_revision,
            tuple,
            payload: empty_projection(),
        }),
    );
    let ack = ack_body(client.read());
    assert_eq!(ack.lifetime_id, lifetime_id);
    assert_eq!(ack.feed_revision, feed_revision);
    assert_eq!(ack.tuple, tuple);

    client.send(session, RecordBody::Bind(ack));
    let bound = bound_body(client.read());
    assert_eq!(bound.lifetime_id, lifetime_id);
    assert_eq!(bound.feed_revision, feed_revision);
    assert_eq!(bound.tuple, tuple);

    client.send(
        session,
        RecordBody::Invalidate(InvalidateBody {
            lifetime_id,
            feed_revision,
            reason: Reason::Shutdown,
        }),
    );
    assert_eq!(
        helper.join().expect("helper thread joined"),
        HelperExit::AuthorityClosed
    );
}

#[test]
fn peer_error_after_establishment_maps_to_peer_closed() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let echo = establishment_echo(&manifest);
    let session = SessionId::new(manifest.launch_nonce);
    let bootstrap = valid_manifest_bytes(&fixtures).to_vec();
    let (client_io, helper_io) = channel_pair();

    let helper = thread::spawn(move || {
        run_helper(
            Cursor::new(bootstrap),
            helper_io,
            ControlOnlyRegistry::new(),
        )
    });

    let mut client = TestClient::with_io(client_io);
    client.complete_handshake();
    client.send(session, RecordBody::Establish(echo));
    let _established = established_body(client.read());
    let _announced = projection_body(client.read());

    client.send(
        session,
        RecordBody::Error(bitty_ipc_api::provider_publication::ErrorCode::Unavailable),
    );
    assert_eq!(
        helper.join().expect("helper thread joined"),
        HelperExit::PeerClosed
    );
}

#[test]
fn invalid_step_after_establishment_maps_to_control() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let echo = establishment_echo(&manifest);
    let session = SessionId::new(manifest.launch_nonce);
    let bootstrap = valid_manifest_bytes(&fixtures).to_vec();
    let (client_io, helper_io) = channel_pair();

    let helper = thread::spawn(move || {
        run_helper(
            Cursor::new(bootstrap),
            helper_io,
            ControlOnlyRegistry::new(),
        )
    });

    let mut client = TestClient::with_io(client_io);
    client.complete_handshake();
    client.send(session, RecordBody::Establish(echo.clone()));
    let _established = established_body(client.read());
    let _announced = projection_body(client.read());

    // A second Establish is not legal after establishment.
    client.send(session, RecordBody::Establish(echo));
    assert_eq!(
        helper.join().expect("helper thread joined"),
        HelperExit::Control
    );
}

#[test]
fn non_establish_first_record_maps_to_establish() {
    let fixtures = fixtures();
    let bootstrap = valid_manifest_bytes(&fixtures).to_vec();
    let session = SessionId::new(valid_manifest(&fixtures).launch_nonce);
    let (client_io, helper_io) = channel_pair();

    let helper = thread::spawn(move || {
        run_helper(
            Cursor::new(bootstrap),
            helper_io,
            ControlOnlyRegistry::new(),
        )
    });

    let mut client = TestClient::with_io(client_io);
    client.complete_handshake();
    client.send(
        session,
        RecordBody::Error(bitty_ipc_api::provider_publication::ErrorCode::Unavailable),
    );

    assert_eq!(
        helper.join().expect("helper thread joined"),
        HelperExit::Establish
    );
}

#[test]
fn control_eof_after_tls_before_establish_maps_to_establish() {
    let fixtures = fixtures();
    let bootstrap = valid_manifest_bytes(&fixtures).to_vec();
    let (client_io, helper_io) = channel_pair();

    let helper = thread::spawn(move || {
        run_helper(
            Cursor::new(bootstrap),
            helper_io,
            ControlOnlyRegistry::new(),
        )
    });

    let mut client = TestClient::with_io(client_io);
    client.complete_handshake();
    drop(client);

    assert_eq!(
        helper.join().expect("helper thread joined"),
        HelperExit::Establish
    );
}

// ── bootstrap failures ──────────────────────────────────────────────────────

#[test]
fn malformed_and_oversized_bootstrap_records_fail_closed() {
    for bytes in [vec![0x00], vec![0x00, 0x01, 0x02, 0x03]] {
        assert_eq!(
            run_helper(
                Cursor::new(bytes),
                Cursor::new(Vec::<u8>::new()),
                ControlOnlyRegistry::new()
            ),
            HelperExit::BootstrapManifest
        );
    }

    let oversized = vec![0_u8; 5 * 1024];
    assert_eq!(
        run_helper(
            Cursor::new(oversized),
            Cursor::new(Vec::<u8>::new()),
            ControlOnlyRegistry::new()
        ),
        HelperExit::BootstrapManifest
    );
}

#[test]
fn trailing_bootstrap_bytes_are_rejected() {
    let fixtures = fixtures();
    let mut bytes = valid_manifest_bytes(&fixtures).to_vec();
    bytes.push(0x00);
    assert_eq!(
        run_helper(
            Cursor::new(bytes),
            Cursor::new(Vec::<u8>::new()),
            ControlOnlyRegistry::new()
        ),
        HelperExit::BootstrapManifest
    );
}

#[test]
fn a_reader_that_resumes_after_eof_is_rejected() {
    let fixtures = fixtures();
    let reader = EofThenBytes::new(valid_manifest_bytes(&fixtures).to_vec());
    assert_eq!(
        run_helper(
            reader,
            Cursor::new(Vec::<u8>::new()),
            ControlOnlyRegistry::new()
        ),
        HelperExit::BootstrapTrailingData
    );
}

#[test]
fn bootstrap_reader_is_dropped_before_control_work() {
    let fixtures = fixtures();
    let dropped = Arc::new(AtomicBool::new(false));
    let bootstrap = DropSignalingReader {
        reader: Cursor::new(valid_manifest_bytes(&fixtures).to_vec()),
        _signal: DropSignal(dropped.clone()),
    };
    let control = BootstrapDropProbe {
        dropped: dropped.clone(),
    };

    assert_eq!(
        run_helper(bootstrap, control, ControlOnlyRegistry::new()),
        HelperExit::Handshake
    );
    assert!(
        dropped.load(Ordering::SeqCst),
        "bootstrap reader outlived manifest ingestion"
    );
}

#[test]
fn a_bootstrap_read_error_fails_closed() {
    assert_eq!(
        run_helper(
            FailingReader,
            Cursor::new(Vec::<u8>::new()),
            ControlOnlyRegistry::new()
        ),
        HelperExit::BootstrapIo
    );
}

#[test]
fn mismatched_helper_key_is_rejected_during_manifest_ingestion() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.key_material.helper_private_key_pkcs8_der =
        zeroize::Zeroizing::new(fixtures.core_key.clone());
    let bootstrap = manifest.encode().to_vec();

    // The manifest parser validates the helper key pair, so a mismatch never
    // reaches the control-server builder.
    assert_eq!(
        run_helper(
            Cursor::new(bootstrap),
            Cursor::new(Vec::<u8>::new()),
            ControlOnlyRegistry::new()
        ),
        HelperExit::BootstrapManifest
    );
}

// ── control failures and platform policy ────────────────────────────────────

#[test]
fn a_control_stream_that_closes_before_tls_maps_to_handshake() {
    let fixtures = fixtures();
    let bootstrap = valid_manifest_bytes(&fixtures).to_vec();
    assert_eq!(
        run_helper(
            Cursor::new(bootstrap),
            Cursor::new(Vec::<u8>::new()),
            ControlOnlyRegistry::new()
        ),
        HelperExit::Handshake
    );
}

#[test]
fn non_unix_platform_fails_closed_before_reading() {
    assert_eq!(
        run_helper_on_platform(false, PanicReader, PanicStream, ControlOnlyRegistry::new()),
        HelperExit::UnsupportedPlatform
    );
    assert_eq!(HelperExit::UnsupportedPlatform.code(), 2);
    assert_eq!(
        HelperExit::UnsupportedPlatform.diagnostic(),
        "helper launch is unsupported on this platform"
    );
}

#[test]
fn exit_codes_and_diagnostics_are_fixed_and_distinct() {
    const ALL: [HelperExit; 10] = [
        HelperExit::AuthorityClosed,
        HelperExit::PeerClosed,
        HelperExit::UnsupportedPlatform,
        HelperExit::BootstrapIo,
        HelperExit::BootstrapManifest,
        HelperExit::BootstrapTrailingData,
        HelperExit::ServerConfiguration,
        HelperExit::Handshake,
        HelperExit::Establish,
        HelperExit::Control,
    ];

    let mut codes = BTreeSet::new();
    for exit in ALL {
        assert!(
            exit.diagnostic().is_ascii(),
            "diagnostic is not ASCII: {exit:?}"
        );
        assert!(!exit.diagnostic().is_empty());
        assert!(
            !exit.diagnostic().contains('/') && !exit.diagnostic().contains('\\'),
            "diagnostic contains a path: {exit:?}"
        );
        assert_eq!(exit.to_string(), exit.diagnostic());
        assert_eq!(exit.is_success(), exit.code() == 0);
        if !exit.is_success() {
            assert!(
                codes.insert(exit.code()),
                "duplicate exit code for {exit:?}"
            );
        }
    }
    assert!(
        !codes.contains(&0),
        "a failure reused the success exit code"
    );

    assert_eq!(HelperExit::AuthorityClosed.code(), 0);
    assert_eq!(HelperExit::PeerClosed.code(), 0);
    assert_eq!(HelperExit::BootstrapIo.code(), 3);
    assert_eq!(HelperExit::BootstrapManifest.code(), 4);
    assert_eq!(HelperExit::BootstrapTrailingData.code(), 5);
    assert_eq!(HelperExit::ServerConfiguration.code(), 6);
    assert_eq!(HelperExit::Handshake.code(), 7);
    assert_eq!(HelperExit::Establish.code(), 8);
    assert_eq!(HelperExit::Control.code(), 9);

    assert_eq!(CHILD_BOOTSTRAP_DESCRIPTOR, 3);
    assert_eq!(CHILD_CONTROL_DESCRIPTOR, 4);
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Bootstrap reader that records when its owned reader is dropped.
struct DropSignalingReader {
    reader: Cursor<Vec<u8>>,
    _signal: DropSignal,
}

impl Read for DropSignalingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buffer)
    }
}

struct DropSignal(Arc<AtomicBool>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Control stream that asserts the bootstrap reader was already released.
struct BootstrapDropProbe {
    dropped: Arc<AtomicBool>,
}

impl Read for BootstrapDropProbe {
    fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
        assert!(
            self.dropped.load(Ordering::SeqCst),
            "bootstrap reader was still open during control I/O"
        );
        Ok(0)
    }
}

impl Write for BootstrapDropProbe {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Reader that returns a complete manifest, then one EOF, then one extra byte.
struct EofThenBytes {
    payload: Vec<u8>,
    offset: usize,
    eof_returned: bool,
    extra_delivered: bool,
}

impl EofThenBytes {
    fn new(payload: Vec<u8>) -> Self {
        Self {
            payload,
            offset: 0,
            eof_returned: false,
            extra_delivered: false,
        }
    }
}

impl Read for EofThenBytes {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.offset < self.payload.len() {
            let count = buffer.len().min(self.payload.len() - self.offset);
            buffer[..count].copy_from_slice(&self.payload[self.offset..self.offset + count]);
            self.offset += count;
            return Ok(count);
        }
        if !self.eof_returned {
            self.eof_returned = true;
            return Ok(0);
        }
        if !self.extra_delivered {
            self.extra_delivered = true;
            buffer[0] = 0x7f;
            return Ok(1);
        }
        Ok(0)
    }
}

struct FailingReader;

impl Read for FailingReader {
    fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "injected bootstrap failure",
        ))
    }
}

struct PanicReader;

impl Read for PanicReader {
    fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
        panic!("bootstrap was read on an unsupported platform");
    }
}

struct PanicStream;

impl Read for PanicStream {
    fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
        panic!("control was read on an unsupported platform");
    }
}

impl Write for PanicStream {
    fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
        panic!("control was written on an unsupported platform");
    }

    fn flush(&mut self) -> io::Result<()> {
        panic!("control was flushed on an unsupported platform");
    }
}
