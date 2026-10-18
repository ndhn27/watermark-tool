#!/usr/bin/env python3
"""
c2pa_provenance.py - C2PA (Content Credentials) manifest signing, layered on
top of watermark.py's existing provenance signals (the visible grid, the
invisible lsb/spread marks, and EXIF/PNG --author/--copyright metadata).

Why this on top of what's already there:
  - The grid, jitter, and invisible marks all answer "is a recoverable trace
    of the mark still in these pixels after some edit?" - a pixel-robustness
    question with no cryptographic guarantee (see README's Threat model
    section: regeneration/diffusion-purification attacks can defeat any of
    them, this scheme included).
  - EXIF/PNG --author/--copyright metadata is real, but it's plain text
    anyone can strip or rewrite with one command (exiftool, a re-save, a
    platform that scrubs metadata on upload) with zero trace of tampering.
  - C2PA sidesteps both problems by asking a different question entirely:
    "does this exact file's content match a manifest that was cryptographically
    signed at edit time?" - a signed hash-binding, not a recoverable pattern.
    Editing the pixels *at all* breaks the signature rather than just maybe
    surviving it, and the manifest chains back to the parent file via a
    `parentOf` ingredient with its own hash, so there's an auditable edit
    history, not just a flat claim. This is a genuinely different, additive
    layer - not a replacement for the pixel-domain marks, which still matter
    once someone re-encodes/screenshots the image and the manifest is gone.

The self-signed-certificate caveat (read this before shipping to anyone):
  By default this module auto-generates and reuses a LOCAL, SELF-SIGNED dev
  certificate (see get_or_create_dev_cert() below) so --c2pa works out of the
  box with no setup. The resulting signature is completely real and
  tamper-evident - c2pa-rs's own hash/signature checks pass on it - but
  because nothing vouches for the certificate itself, any C2PA-aware reader
  (c2pa_verify.py included) reports the signing credential as *untrusted*
  rather than showing a verified checkmark. That's expected, not a bug: get
  an actual "verified" badge only by signing with a certificate chaining up
  to a CA in the C2PA trust list (see README.md's C2PA section for where to
  get one) and passing it via --c2pa-cert/--c2pa-key.

Certificate requirements, the hard way:
  c2pa-rs rejects a certificate it doesn't like with a bare "the certificate
  is invalid" - no further detail - which makes reverse-engineering exactly
  what it wants unavoidable if you don't already have a known-good example.
  Cross-checked against c2pa-python's own ES256 test fixtures
  (tests/fixtures/es256_certs.pem in contentauth/c2pa-python), the leaf cert
  needs ALL of: BasicConstraints, KeyUsage, and ExtendedKeyUsage(emailProtection)
  marked `critical=True` (not just present - non-critical EKU alone still
  gets rejected), KeyUsage with digitalSignature+contentCommitment (aka
  nonRepudiation), and a SubjectKeyIdentifier that the issuing (CA) cert's
  own AuthorityKeyIdentifier points back to. Miss any one of these and
  signing fails with that same undifferentiated "certificate is invalid" -
  _generate_dev_cert() below reproduces the working combination exactly.

No timestamp authority is contacted by default. C2paSignerInfo supports a
`ta_url` for RFC 3161 trusted timestamping (proves the signature existed
before the cert's own expiry, which matters for long-term validity), but
that means every signing call makes an outbound network request to a third
party - a surprising default and a new external point of failure/latency for
what's otherwise a fully offline tool. This module leaves it off; pass
ta_url= directly to sign_file() (there's no CLI flag for it) if you're
wiring up your own signer/CA setup and want it.

Usage (as a library - watermark.py and video.py call this, not the user
directly):
    import c2pa_provenance
    c2pa_provenance.sign_file(
        "out.png", title="out.png", author="Jane Doe", parent_path="in.png",
    )
    info = c2pa_provenance.read_manifest("out.png")
"""

import datetime
import json
import mimetypes
import os
from pathlib import Path

try:
    import c2pa
except ImportError:
    c2pa = None

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
except ImportError:
    x509 = None

# Both packages are required - c2pa-python does the actual signing/reading,
# cryptography generates the dev cert. Neither is in the base requirements
# (see requirements.txt's comment) since --c2pa is opt-in and the two
# packages together pull in a real Rust binary extension.
AVAILABLE = c2pa is not None and x509 is not None

# The dev certificate is ES256 (P-256) only - see _generate_dev_cert()'s
# docstring for why a from-scratch generator only targets one algorithm.
# Bring your own --c2pa-cert/--c2pa-key for anything else.
DEFAULT_ALG = "es256"
DEFAULT_CERT_DIR = Path.home() / ".watermark-tool"
CLAIM_GENERATOR_NAME = "watermark-tool"
CLAIM_GENERATOR_VERSION = "8.0"

# A real, registered C2PA action ID (c2pa.org's actions taxonomy) - not a
# made-up custom string - so any spec-compliant C2PA reader recognizes it,
# not just this tool's own c2pa_verify.py.
WATERMARK_ACTION = "c2pa.watermarked"
# Prefix of the schema.org `identifier` that carries the watermark ID (see sign_file()).
WM_ID_IDENTIFIER_PREFIX = "wmid:"


class C2PAUnavailable(RuntimeError):
    """Raised when --c2pa (or read_manifest/sign_file directly) is used but
    c2pa-python and/or cryptography aren't installed. Deliberately NOT a
    subclass of any c2pa.C2paError, since c2pa may be None here - callers
    that want to let genuine signing failures through as notes (see
    watermark.py/video.py) should catch this one separately and treat it as
    fatal/config-error instead."""


def _require_available():
    if not AVAILABLE:
        raise C2PAUnavailable(
            "C2PA support needs two extra packages this tool doesn't install "
            "by default: pip install c2pa-python cryptography (both are listed, "
            "commented out, in requirements.txt)."
        )


# ---------------------------------------------------------------------------
# Dev certificate (auto-generated, self-signed, ES256-only - see module
# docstring for the exact extension set this reproduces and why)
# ---------------------------------------------------------------------------

def dev_cert_paths(cert_dir=None):
    cert_dir = Path(cert_dir) if cert_dir else DEFAULT_CERT_DIR
    return cert_dir / "dev_cert.pem", cert_dir / "dev_key.pem"


def _generate_dev_cert(cert_path, key_path):
    """Write a fresh self-signed ES256 leaf+CA chain to cert_path (both certs,
    leaf first) and the leaf's private key to key_path. See the module
    docstring's "Certificate requirements" section for why each extension
    below is there - this isn't a minimal cert, it's the minimal set that
    c2pa-rs's validator has been observed to actually accept.
    """
    _require_available()
    now = datetime.datetime.now(datetime.timezone.utc)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "watermark-tool local dev"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "FOR TESTING ONLY - NOT TRUSTED"),
        x509.NameAttribute(NameOID.COMMON_NAME, "watermark-tool dev root CA"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False, data_encipherment=False,
                key_agreement=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "watermark-tool local dev"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "FOR TESTING ONLY - NOT TRUSTED"),
        x509.NameAttribute(NameOID.COMMON_NAME, "watermark-tool dev signer"),
    ])
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))  # CA/Browser Forum's own TLS-cert cap; arbitrary but conventional for a leaf
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=True,
                key_encipherment=False, data_encipherment=False, key_agreement=False,
                key_cert_sign=False, crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    cert_path = Path(cert_path)
    key_path = Path(key_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    # Leaf first, then its issuer - the order c2pa-rs (and the es256_certs.pem
    # fixture it's tested against) expects a chain in.
    cert_path.write_bytes(
        leaf_cert.public_bytes(serialization.Encoding.PEM)
        + ca_cert.public_bytes(serialization.Encoding.PEM)
    )
    key_path.write_bytes(
        leaf_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    try:
        os.chmod(key_path, 0o600)  # best-effort; irrelevant on filesystems that ignore unix perms
    except OSError:
        pass


def get_or_create_dev_cert(cert_dir=None):
    """Return (cert_path, key_path, created) as strings, generating the pair
    once at DEFAULT_CERT_DIR (or cert_dir) and reusing it on every later call
    - reusing it matters as much as generating it: a stable identity across
    runs is the whole point of a signing cert, and a fresh one every
    invocation would make every past signature's certificate a dead end.

    Call this once yourself, up front, before any --batch multiprocessing
    pool starts (watermark.py's main() does exactly this) rather than
    letting sign_file() below auto-create it lazily in a worker: the
    check-then-write here isn't lock-protected, and several worker processes
    racing to generate the same not-yet-existing files on first use could
    corrupt one or both. Once the files exist, concurrent *reads* of them
    (which is all sign_file() does with a pre-resolved path) are fine.
    """
    cert_path, key_path = dev_cert_paths(cert_dir)
    created = False
    if not (cert_path.exists() and key_path.exists()):
        _generate_dev_cert(cert_path, key_path)
        created = True
    return str(cert_path), str(key_path), created


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def _mime_type(path):
    return mimetypes.guess_type(str(path))[0]


def sign_file(
    path,
    title=None,
    author=None,
    parent_path=None,
    cert_path=None,
    key_path=None,
    alg=DEFAULT_ALG,
    ta_url=None,
    action=WATERMARK_ACTION,
    wm_id=None,
):
    """Sign the file already sitting at `path` (watermark.py/video.py call
    this AFTER saving the watermarked output, not before) with a C2PA
    manifest, in place - `path` is overwritten with the signed version via a
    same-directory temp file + os.replace(), so a signing failure partway
    through never leaves `path` truncated or corrupted.

    If `parent_path` is given and exists, the manifest is built as an EDIT
    of it: an ingredient with relationship "parentOf" pointing at
    `parent_path` is added before the custom action, which makes the
    C2PA/c2pa-rs side auto-generate a `c2pa.opened` action (a hash-link back
    to the parent) ahead of the explicit `action` (default: c2pa.watermarked,
    a real registered C2PA action ID) this function adds - so the manifest
    reads as "opened this specific parent, then watermarked it", an
    auditable two-step history, not just an unqualified claim. Without
    `parent_path` (or if it no longer exists on disk), the manifest is built
    as a plain CREATE instead.

    cert_path/key_path: a PEM file (or two) for your own signing identity.
    Give both or neither - neither means the auto-generated/reused dev
    certificate from get_or_create_dev_cert() (see its docstring on why that
    should be resolved once up front rather than left to this function in a
    --batch worker). `alg` is silently forced to DEFAULT_ALG when the dev
    cert is in use, since that cert is only ever ES256 - callers (main()
    below) should warn the user about this themselves rather than relying on
    a silent override here.

    Raises C2PAUnavailable if the required packages aren't installed, or
    whatever c2pa.C2paError subclass the underlying library raises on a
    genuine signing problem (unsupported format, bad cert, timestamp-authority
    network failure, ...) - callers are expected to catch the latter and
    degrade to a printed note, the same way a missing `piexif` degrades
    EXIF writing elsewhere in this tool, since a C2PA signing failure
    shouldn't discard an otherwise-successful watermark.
    """
    _require_available()

    if (cert_path is None) != (key_path is None):
        raise ValueError("cert_path and key_path must be given together (or both omitted for the dev cert)")
    if cert_path is None:
        cert_path, key_path, _ = get_or_create_dev_cert()
        alg = DEFAULT_ALG

    cert_pem = Path(cert_path).read_bytes()
    key_pem = Path(key_path).read_bytes()

    signer_info = c2pa.C2paSignerInfo(
        alg=alg.encode("ascii") if isinstance(alg, str) else alg,
        sign_cert=cert_pem,
        private_key=key_pem,
        ta_url=ta_url.encode("utf-8") if isinstance(ta_url, str) else ta_url,
    )

    manifest = {
        "claim_generator_info": [{"name": CLAIM_GENERATOR_NAME, "version": CLAIM_GENERATOR_VERSION}],
        "title": title or os.path.basename(path),
    }
    if author or wm_id:
        # stds.schema-org.CreativeWork is a standard C2PA assertion label
        # for exactly this - free-text author attribution inside the signed
        # manifest, readable by any C2PA-aware tool even if it has no idea
        # about --author's separate EXIF/PNG-metadata role elsewhere in this
        # tool. Unlike that EXIF/PNG metadata, this works for video too.
        # The watermark ID (--id) rides in the same assertion as its schema.org
        # `identifier`, prefixed "wmid:" so it can't be mistaken for another
        # kind of identifier a tool might have added.
        creative_work = {"@context": "https://schema.org", "@type": "CreativeWork"}
        if author:
            creative_work["author"] = [{"@type": "Person", "name": author}]
        if wm_id:
            creative_work["identifier"] = WM_ID_IDENTIFIER_PREFIX + wm_id
        manifest["assertions"] = [{"label": "stds.schema-org.CreativeWork", "data": creative_work}]

    tmp_path = f"{path}.c2pa_tmp"
    try:
        with c2pa.Signer.from_info(signer_info) as signer, c2pa.Builder(manifest) as builder:
            has_parent = parent_path and os.path.exists(parent_path)
            if has_parent:
                builder.set_intent(c2pa.C2paBuilderIntent.EDIT)
                parent_mime = _mime_type(parent_path) or _mime_type(path)
                ingredient_json = {"relationship": "parentOf", "title": os.path.basename(parent_path)}
                with open(parent_path, "rb") as ingredient_file:
                    builder.add_ingredient(ingredient_json, parent_mime, ingredient_file)
            else:
                builder.set_intent(c2pa.C2paBuilderIntent.CREATE)
            builder.add_action({"action": action})
            builder.sign_file(path, tmp_path, signer)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Reading back (used by c2pa_verify.py)
# ---------------------------------------------------------------------------

def read_manifest(path):
    """Read back the active C2PA manifest from `path`, if any.

    Returns {"has_manifest": False} if the file has no embedded manifest at
    all (c2pa.C2paError.ManifestNotFound is c2pa-python's normal, expected
    way of saying so - not treated as an error here). Otherwise returns a
    dict with the manifest's title/claim_generator/actions/ingredients/
    author plus reader.get_validation_state() and .get_validation_results()
    verbatim, so c2pa_verify.py (or any other caller) can decide how much of
    the full validation detail to print.
    """
    _require_available()
    try:
        with c2pa.Reader(path) as reader:
            full = json.loads(reader.json())
            active = full.get("active_manifest")
            manifest = full.get("manifests", {}).get(active, {}) if active else {}

            actions = []
            author = None
            wm_id = None
            for assertion in manifest.get("assertions", []):
                label = assertion.get("label", "")
                if label.startswith("c2pa.actions"):
                    for act in assertion.get("data", {}).get("actions", []):
                        if act.get("action"):
                            actions.append(act["action"])
                elif label == "stds.schema-org.CreativeWork":
                    data = assertion.get("data", {})
                    authors = data.get("author") or []
                    if authors:
                        author = authors[0].get("name")
                    ident = data.get("identifier")
                    for cand in (ident if isinstance(ident, list) else [ident]):
                        if isinstance(cand, str) and cand.startswith(WM_ID_IDENTIFIER_PREFIX):
                            wm_id = cand[len(WM_ID_IDENTIFIER_PREFIX):]

            generator_info = manifest.get("claim_generator_info") or []
            claim_generator = None
            if generator_info:
                gen = generator_info[0]
                claim_generator = f"{gen.get('name')} {gen.get('version')}".strip()

            return {
                "has_manifest": True,
                "title": manifest.get("title"),
                "claim_generator": claim_generator,
                "actions": actions,
                "ingredients": [ing.get("title") for ing in manifest.get("ingredients", []) if ing.get("title")],
                "author": author,
                "wm_id": wm_id,
                "validation_state": reader.get_validation_state(),
                "validation_results": reader.get_validation_results(),
            }
    except c2pa.C2paError.ManifestNotFound:
        return {"has_manifest": False}


def sign_file_or_note(output_path, *, title=None, author=None, parent_path=None, cert_path=None,
                      key_path=None, alg=DEFAULT_ALG, notify=print, wm_id=None):
    """sign_file(), but a signing failure becomes a note instead of an
    exception - the graceful-degradation shape this tool uses everywhere for
    optional extras (a missing `piexif` is a note too). A file that took real
    CPU-minutes to watermark shouldn't be thrown away over a C2PA problem
    alone. C2PAUnavailable (the packages just aren't installed) is the one
    exception: callers check for that up front, so meeting it this late means
    something is set up strangely, and re-raising is more honest than a note.
    Returns True if the manifest was written.

    Shared by the still-image path (wmapp.pipeline) and video.py, which each
    used to carry their own copy of this try/except.
    """
    try:
        sign_file(
            output_path, title=title, author=author, parent_path=parent_path,
            cert_path=cert_path, key_path=key_path, alg=alg, wm_id=wm_id,
        )
        return True
    except C2PAUnavailable:
        raise
    except Exception as exc:
        notify(
            f"Note: --c2pa signing failed for {output_path} ({exc}); "
            f"the watermark itself was still saved without a C2PA manifest."
        )
        return False
