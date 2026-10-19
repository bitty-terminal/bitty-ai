# Control-transport test fixtures

These files are **test-only and non-secret**. They are intentionally committed
so the crate's manifest and TLS tests are deterministic, offline, and
cross-platform. They must never be used outside tests, must never be trusted by
a real launch, and are not production credentials or a PKI.

## Files

| File                       | Content                                                  |
| -------------------------- | -------------------------------------------------------- |
| `core-cert.der`            | Self-signed `core-supervisor` client leaf (DER)          |
| `core-key-pkcs8.der`       | Matching ECDSA P-256 private key (PKCS#8 DER)            |
| `helper-cert.der`          | Self-signed `ai-helper` server leaf (DER)                |
| `helper-key-pkcs8.der`     | Matching ECDSA P-256 private key (PKCS#8 DER)            |
| `rogue-core-cert.der`      | Unpinned client leaf used by the pin-mismatch test (DER) |
| `rogue-core-key-pkcs8.der` | Matching rogue private key (PKCS#8 DER)                  |
| `p384-cert.der`            | P-384 leaf used only to prove curve rejection (DER)      |
| `p384-key-pkcs8.der`       | Matching P-384 private key (PKCS#8 DER)                  |

## Generation

Generated on 2026-10-19 with OpenSSL 3.6.5. Run the commands from a temporary
directory; only the DER files are copied into this directory. Replace the
subject CNs only if the fixture roles change.

```sh
# Core-supervisor client leaf.
openssl req -new -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
  -keyout core-key.pem -out core-cert.pem -days 3650 \
  -subj "/CN=bitty-test-core-supervisor" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=clientAuth" \
  -sha256

# AI-helper server leaf.
openssl req -new -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
  -keyout helper-key.pem -out helper-cert.pem -days 3650 \
  -subj "/CN=bitty-test-ai-helper" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=serverAuth" \
  -sha256

# Unpinned rogue client leaf for the pin-mismatch test.
openssl req -new -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
  -keyout rogue-core-key.pem -out rogue-core-cert.pem -days 3650 \
  -subj "/CN=bitty-test-rogue-core" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=clientAuth" \
  -sha256

# P-384 leaf proving the key_type=1 curve check rejects a non-P-256 key.
openssl req -new -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-384 -nodes \
  -keyout p384-key.pem -out p384-cert.pem -days 3650 \
  -subj "/CN=bitty-test-p384-helper" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=serverAuth" \
  -sha256

# Convert to the checked-in encodings.
for name in core helper rogue-core p384; do
  openssl pkcs8 -topk8 -nocrypt -in "$name-key.pem" -outform DER \
    -out "$name-key-pkcs8.der"
  openssl x509 -in "$name-cert.pem" -outform DER -out "$name-cert.der"
done
```

## Properties

- ECDSA P-256, SHA-256 signatures, self-signed leaves.
- `basicConstraints` CA false, `keyUsage` digitalSignature, role-appropriate
  `extendedKeyUsage`.
- No subject alternative names, matching the launch-leaf policy.
- One leaf per role; no intermediates or chains.
- The DER certificate sizes are well under `LAUNCH_CERT_MAX_BYTES` and the
  PKCS#8 key sizes are well under `LAUNCH_KEY_MAX_BYTES`.
