//! End-to-end in-memory TLS session -> HelperControl tests.
//!
//! The server and client rustls connections are pumped in one thread over
//! shared byte queues. No socket, thread, or clock is created. These tests
//! exercise the production `ControlServer::accept` and
//! `HelperControl::from_session` path, including the manifest/session
//! cross-check in `HelperControl::new`.

mod common;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{self, Cursor, Read, Write};
use std::rc::Rc;

use bitty_ai_control::{
    CONTROL_ALPN, CertificatePin, ControlServer, HelperControl, HelperControlError, RegistryError,
    RegistrySnapshot, RegistryView,
};
use bitty_ipc_api::provider_publication::{
    ControlRecord, Digest, EstablishmentEcho, Nonce, ProviderRoutingProjection, RecordBody, Sender,
    SessionId,
};
use common::{client_config, client_connection, fixtures, valid_manifest};
use rustls::{ClientConnection, Stream};

fn frame(record: &ControlRecord) -> Vec<u8> {
    let bytes = record.encode().expect("record encodes");
    let length = u32::try_from(bytes.len()).expect("record length");
    let mut frame = Vec::with_capacity(4 + bytes.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&bytes);
    frame
}

fn parse_frames(bytes: &[u8]) -> Vec<ControlRecord> {
    let mut cursor = Cursor::new(bytes);
    let mut records = Vec::new();
    loop {
        let mut prefix = [0_u8; 4];
        match cursor.read_exact(&mut prefix) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => panic!("frame prefix: {error}"),
        }
        let length = u32::from_be_bytes(prefix) as usize;
        let mut body = vec![0_u8; length];
        cursor.read_exact(&mut body).expect("frame body");
        records.push(ControlRecord::decode(&body).expect("record decodes"));
    }
    records
}

/// Shared client-side view of the in-memory TLS pipe.
#[derive(Clone)]
struct Link {
    client: Rc<RefCell<ClientConnection>>,
    client_to_server: Rc<RefCell<VecDeque<u8>>>,
    server_to_client: Rc<RefCell<VecDeque<u8>>>,
}

impl Link {
    fn new(client: ClientConnection) -> Self {
        Self {
            client: Rc::new(RefCell::new(client)),
            client_to_server: Rc::new(RefCell::new(VecDeque::new())),
            server_to_client: Rc::new(RefCell::new(VecDeque::new())),
        }
    }

    /// Feed server ciphertext to the client, process it, and drain any client
    /// response ciphertext back toward the server.
    fn pump(&self) {
        let mut client = self.client.borrow_mut();
        loop {
            let chunk = {
                let mut inbound = self.server_to_client.borrow_mut();
                if inbound.is_empty() {
                    break;
                }
                inbound.drain(..).collect::<Vec<u8>>()
            };
            let mut cursor = Cursor::new(chunk);
            while client.read_tls(&mut cursor).expect("client read_tls") > 0 {}
        }
        client.process_new_packets().expect("client process");

        while client.wants_write() {
            let written = {
                let mut outbound = self.client_to_server.borrow_mut();
                client
                    .write_tls(&mut QueueSink {
                        output: &mut outbound,
                    })
                    .expect("client write_tls")
            };
            if written == 0 {
                break;
            }
        }
    }

    fn send(&self, record: &ControlRecord) {
        let bytes = frame(record);
        let mut client = self.client.borrow_mut();
        let mut io = ClientIo {
            output: self.client_to_server.clone(),
        };
        let mut stream = Stream::new(&mut *client, &mut io);
        stream.write_all(&bytes).expect("client write");
        stream.flush().expect("client flush");
    }

    fn receive_records(&self) -> Vec<ControlRecord> {
        self.pump();
        let mut client = self.client.borrow_mut();
        let mut io = ClientIo {
            output: self.client_to_server.clone(),
        };
        let mut stream = Stream::new(&mut *client, &mut io);
        let mut plaintext = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => plaintext.extend_from_slice(&buffer[..read]),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => panic!("client read: {error}"),
            }
        }
        parse_frames(&plaintext)
    }
}

impl Read for Link {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.pump();
        let mut inbound = self.client_to_server.borrow_mut();
        let count = buffer.len().min(inbound.len());
        for slot in &mut buffer[..count] {
            *slot = inbound.pop_front().expect("count bound");
        }
        Ok(count)
    }
}

impl Write for Link {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.server_to_client
            .borrow_mut()
            .extend(buffer.iter().copied());
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct QueueSink<'a> {
    output: &'a mut VecDeque<u8>,
}

impl Write for QueueSink<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.output.extend(buffer.iter().copied());
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct ClientIo {
    output: Rc<RefCell<VecDeque<u8>>>,
}

impl Read for ClientIo {
    fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
        // The client transport is fed only by `Link::pump`; do not report an
        // EOF that rustls would treat as a truncated TLS stream.
        Err(io::Error::from(io::ErrorKind::WouldBlock))
    }
}

impl Write for ClientIo {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.output.borrow_mut().extend(buffer.iter().copied());
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct EmptyRegistry {
    revision: u64,
    projection: ProviderRoutingProjection,
}

impl EmptyRegistry {
    fn new(revision: u64) -> Self {
        Self {
            revision,
            projection: ProviderRoutingProjection::empty(),
        }
    }
}

impl RegistryView for EmptyRegistry {
    fn snapshot(&self) -> Result<RegistrySnapshot, RegistryError> {
        RegistrySnapshot::new(self.revision, self.projection.clone())
    }

    fn retained_matches(&self, feed_revision: u64, retained: &[u8]) -> Result<bool, RegistryError> {
        if feed_revision != self.revision {
            return Ok(false);
        }
        Ok(self.projection.canonical_bytes() == retained)
    }
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

#[test]
fn handshake_preserves_launch_expectations() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let session_id = SessionId::new(manifest.launch_nonce);
    let server = ControlServer::new(manifest).expect("server");
    let handshake = server.handshake().expect("handshake");
    assert_eq!(
        handshake.establishment_expectations().launch_nonce(),
        session_id
    );
}

#[test]
fn from_session_establishes_over_a_real_tls_session() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let echo = echo_for(&manifest);
    let session_id = SessionId::new(manifest.launch_nonce);
    let server = ControlServer::new(manifest).expect("server");
    let link = Link::new(client_connection(client_config(
        &fixtures,
        CONTROL_ALPN,
        false,
    )));
    let session = server.accept(link.clone()).expect("accept");
    assert_eq!(
        session.peer().certificate_pin(),
        CertificatePin::of_der(&fixtures.core_certificate)
    );
    assert_eq!(
        session.establishment_expectations().launch_nonce(),
        session_id
    );

    let mut control = HelperControl::from_session(session, EmptyRegistry::new(1));
    link.send(&core_record(
        1,
        session_id,
        RecordBody::Establish(echo.clone()),
    ));
    control.establish().expect("establish");

    let records = link.receive_records();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].sender, Sender::Helper);
    assert_eq!(records[0].sequence, 1);
    let established = match &records[0].body {
        RecordBody::Established(body) => body,
        other => panic!("not Established: {other:?}"),
    };
    assert_eq!(established.echo, echo);
    let projection = match &records[1].body {
        RecordBody::Projection(body) => body,
        other => panic!("not Projection: {other:?}"),
    };
    assert_eq!(projection.lifetime_id, established.lifetime_id);
    assert_eq!(projection.feed_revision, 1);
    assert_eq!(control.lifetime_id(), Some(established.lifetime_id));
}

#[test]
fn new_refuses_a_manifest_that_does_not_match_the_tls_session() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let server = ControlServer::new(manifest).expect("server");
    let link = Link::new(client_connection(client_config(
        &fixtures,
        CONTROL_ALPN,
        false,
    )));
    let session = server.accept(link).expect("accept");

    let mut mismatched = valid_manifest(&fixtures);
    mismatched.component_name = "other-component".to_owned();
    let error = HelperControl::new(mismatched, session, EmptyRegistry::new(1))
        .expect_err("manifest/session mismatch");
    assert!(matches!(error, HelperControlError::LaunchMismatch));
}
