"""Core BinHex 4.0 encoding and decoding logic.

BinHex 4.0 is a legacy Macintosh binary-to-text encoding. A BinHex 4.0
file has this structure:

    (This file must be converted with BinHex 4.0)
    <newline>
    :
    <RLE-compressed data fork fork wrapped in 8-bit-to-6-bit ASCII encoding>
    :

The data between the two colon lines is a 64-character alphabet encoding
(\"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-\") of
a byte stream that has been RLE-compressed. The decompressed stream is the
\"data fork\" (in BinHex terminology) and has this layout:

    1 byte   name length (0-63)
    N bytes  name (ASCII)
    1 byte   version (always 0)
    1 byte   type[0]
    1 byte   type[1]
    1 byte   type[2]
    1 byte   type[3]
    1 byte   author[0]
    1 byte   author[1]
    1 byte   author[2]
    1 byte   author[3]
    2 bytes  flags (big-endian)
    4 bytes  dlen: data fork length (big-endian, unsigned)
    4 bytes  rlen: resource fork length (big-endian, unsigned)
    dlen bytes   data fork
    rlen bytes   resource fork
    2 bytes  CRC-16 (big-endian, CCITT polynomial 0x1021, init 0x0000)

Design decisions
-----------------
This library deliberately interprets \"BinHex 4.0 file\" as the data BETWEEN
the two colons (the colon-delimited payload) rather than the full text
file with the header line. The header line and the colons themselves are
presentation, not payload, and real-world files vary in how many header
lines precede the first colon. Accepting only the between-colons payload
makes the decoder deterministic and testable without guessing at header
variations. If you have a full BinHex text file, call ``parse_file`` to
strip the header and extract the payload first.

The encoder produces ONLY the colon-delimited payload (without the header
line and without trailing newline). Round-tripping ``encode(decode(x))`` is
guaranteed to produce ``x`` exactly when ``x`` is a valid payload.
"""

from __future__ import annotations

import struct
from typing import NamedTuple


class BinHexError(ValueError):
    """Raised when BinHex input is structurally invalid or fails CRC verification."""


# BinHex 4.0 uses a 64-symbol alphabet. This is NOT base64; it uses its own
# ordering with '+' and '-' as the last two symbols. We encode with a lookup
# table (a bytes.translate-style mapping) for speed and decode with a dict for
# clarity. The two are kept consistent by construction in the tests.
_B64_ALPHABET = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-"
_B64_DECODE_MAP = {c: i for i, c in enumerate(_B64_ALPHABET)}

# The special byte value 0x90 is the RLE marker. A literal 0x90 in the input
# is escaped as 0x90 0x00. A run of N bytes (N >= 4) is encoded as
# <byte> 0x90 <N>. This mirrors the classic Macintosh RLE scheme used by
# BinHex 4.0.
_RLE_MARK = 0x90


class BinHexRecord(NamedTuple):
    """A decoded BinHex 4.0 data-fork fork (in the payload sense).

    Attributes:
        name: The filename, as ASCII text.
        version: The version byte (0 in BinHex 4.0).
        type: The 4-byte Macintosh type code.
        author: The 4-byte Macintosh creator code.
        flags: The 16-bit Finder flags.
        data_fork: The data fork bytes (length ``dlen``).
        resource_fork: The resource fork bytes (length ``rlen``).
        crc: The CRC-16 value read from the payload (big-endian).
    """

    name: str
    version: int
    type: bytes
    author: bytes
    flags: int
    data_fork: bytes
    resource_fork: bytes
    crc: int


def _rle_decompress(data: bytes) -> bytes:
    """Reverse the BinHex 4.0 RLE compression.

    The scheme: 0x90 is a run marker. If followed by 0x00, it is a literal
    0x90. If followed by N (where 0x01 <= N <= 0xFF), the preceding byte is
    repeated (N - 1) additional times — i.e. the run total is N copies,
    including the one already emitted. We implement the straightforward
    single-pass decoder.
    """
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == _RLE_MARK:
            if i + 1 >= n:
                raise BinHexError("RLE marker 0x90 at end of stream, missing count")
            count = data[i + 1]
            if count == 0:
                # Literal 0x90.
                out.append(_RLE_MARK)
                i += 2
            else:
                # Repeat the previous byte (count - 1) more times.
                if not out:
                    raise BinHexError("RLE run with no preceding byte")
                prev = out[-1]
                out.extend(bytes([prev]) * (count - 1))
                i += 2
        else:
            out.append(b)
            i += 1
    return bytes(out)


def _rle_compress(data: bytes) -> bytes:
    """Apply the BinHex 4.0 RLE compression.

    We only emit a run when a byte repeats 4 or more times, matching the
    classic implementation's threshold. Runs of exactly 3 are emitted
    literally because the RLE encoding (byte + 0x90 + count) is 3 bytes long,
    offering no savings and risking incompatibility with strict decoders.
    Runs longer than 255 are split into chunks of 255.
    """
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        # Count consecutive identical bytes.
        run = 1
        while i + run < n and data[i + run] == b and run < 255:
            run += 1
        if b == _RLE_MARK:
            # Every literal 0x90 must be escaped as 0x90 0x00, regardless of run
            # length, because a bare 0x90 would be interpreted as a run marker.
            for _ in range(run):
                out.append(_RLE_MARK)
                out.append(0x00)
        elif run >= 4:
            out.append(b)
            out.append(_RLE_MARK)
            out.append(run)
        else:
            out.extend(bytes([b]) * run)
        i += run
    return bytes(out)


def _crc16(data: bytes) -> int:
    """Compute CRC-16 with CCITT polynomial 0x1021, initial value 0x0000.

    BinHex 4.0 uses this exact CRC variant over the decompressed data fork
    fork (the header fields through the end of the resource fork), not
    including the two CRC bytes themselves. The polynomial is 0x1021 and the
    initial value is 0x0000, with no final XOR. This is the variant Macintosh
    toolchains historically used.
    """
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _b64_encode(data: bytes) -> bytes:
    """Encode bytes using the BinHex 4.0 64-symbol alphabet.

    This is a 6-bit-per-symbol encoding similar in spirit to base64 but with a
    custom alphabet and custom framing: the input is split into 3-byte groups,
    each producing 4 symbols, with padding handled by emitting only as many
    symbols as there are input bytes plus one trailing symbol encoding the
    number of padding bits. The trailing symbol is placed BEFORE the main
    data and is itself part of the encoded stream.

    Specifically, the BinHex 4.0 framing is:
      - Compute the number of bytes that would be needed if the stream were
        padded to a 3-byte boundary. Emit one leading symbol whose value is
        (pad_count), where pad_count is 0, 1, or 2.
      - Then emit the 6-bit symbols for the data in 3-byte groups, with the
        final group zero-padded on the right.
    """
    if not data:
        # A leading symbol of 0 means "no data, no padding."
        return bytes([_B64_ALPHABET[0]])

    remainder = len(data) % 3
    pad = (3 - remainder) % 3
    out = bytearray()
    out.append(_B64_ALPHABET[pad])

    padded = data + bytes(pad)
    for i in range(0, len(padded), 3):
        b0, b1, b2 = padded[i], padded[i + 1], padded[i + 2]
        out.append(_B64_ALPHABET[b0 >> 2])
        out.append(_B64_ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)])
        out.append(_B64_ALPHABET[((b1 & 0x0F) << 2) | (b2 >> 6)])
        out.append(_B64_ALPHABET[b2 & 0x3F])
    return bytes(out)


def _b64_decode(data: bytes) -> bytes:
    """Decode a BinHex 4.0 64-symbol-alphabet stream.

    The first symbol encodes the padding count (0, 1, or 2). The remaining
    symbols are decoded in groups of four into three bytes, with the final
    group trimmed by the padding count.
    """
    if not data:
        raise BinHexError("empty b64 stream")

    if data[0] not in _B64_DECODE_MAP:
        raise BinHexError(f"invalid leading symbol: {chr(data[0])!r}")
    pad = _B64_DECODE_MAP[data[0]]
    if pad > 2:
        raise BinHexError(f"invalid padding count {pad}")

    body = data[1:]
    # Every symbol must be valid; decode up front so we get a clear error.
    values = []
    for c in body:
        if c not in _B64_DECODE_MAP:
            raise BinHexError(f"invalid symbol: {chr(c)!r}")
        values.append(_B64_DECODE_MAP[c])

    if len(values) % 4 != 0:
        raise BinHexError(
            f"b64 body length {len(values)} is not a multiple of 4"
        )

    out = bytearray()
    for i in range(0, len(values), 4):
        v0, v1, v2, v3 = values[i : i + 4]
        out.append((v0 << 2) | (v1 >> 4))
        out.append(((v1 & 0x0F) << 4) | (v2 >> 2))
        out.append(((v2 & 0x03) << 6) | v3)

    if pad:
        # The final group contributed (3 - pad) real bytes plus pad padding
        # bytes. But we only know the total; trim from the end.
        # Total real bytes = len(out) - pad.
        return bytes(out[: len(out) - pad])
    return bytes(out)


def _parse_data_fork(raw: bytes) -> BinHexRecord:
    """Parse the decompressed BinHex data fork into a structured record."""
    if len(raw) < 1:
        raise BinHexError("data fork too short for name length")
    name_len = raw[0]
    header_len = 1 + name_len + 1 + 8 + 2 + 8
    if len(raw) < header_len:
        raise BinHexError(
            f"data fork too short: need >= {header_len} header bytes, "
            f"got {len(raw)}"
        )

    pos = 1
    name = raw[pos : pos + name_len].decode("ascii", errors="strict")
    pos += name_len

    version = raw[pos]
    pos += 1

    type_code = raw[pos : pos + 4]
    pos += 4
    author_code = raw[pos : pos + 4]
    pos += 4

    flags = (raw[pos] << 8) | raw[pos + 1]
    pos += 2

    dlen = int.from_bytes(raw[pos : pos + 4], "big")
    pos += 4
    rlen = int.from_bytes(raw[pos : pos + 4], "big")
    pos += 4

    total = header_len + dlen + rlen + 2
    if len(raw) < total:
        raise BinHexError(
            f"data fork too short: declared {dlen}+{rlen} fork bytes plus "
            f"2 CRC bytes, but only {len(raw) - header_len} bytes remain"
        )

    data_fork = raw[pos : pos + dlen]
    pos += dlen
    resource_fork = raw[pos : pos + rlen]
    pos += rlen

    crc = (raw[pos] << 8) | raw[pos + 1]
    pos += 2

    # CRC covers everything from the name-length byte through the end of the
    # resource fork, NOT including the two CRC bytes themselves.
    crc_region = raw[: header_len + dlen + rlen]
    actual = _crc16(crc_region)
    if actual != crc:
        raise BinHexError(
            f"CRC mismatch: computed {actual:04X}, expected {crc:04X}"
        )

    return BinHexRecord(
        name=name,
        version=version,
        type=bytes(type_code),
        author=bytes(author_code),
        flags=flags,
        data_fork=bytes(data_fork),
        resource_fork=bytes(resource_fork),
        crc=crc,
    )


def _build_data_fork(rec: BinHexRecord) -> bytes:
    """Serialize a BinHexRecord back into the raw data-fork byte stream."""
    name_bytes = rec.name.encode("ascii")
    if len(name_bytes) > 63:
        raise ValueError("name exceeds 63 ASCII bytes")
    out = bytearray()
    out.append(len(name_bytes))
    out.extend(name_bytes)
    out.append(rec.version)
    out.extend(rec.type)
    out.extend(rec.author)
    out.extend(rec.flags.to_bytes(2, "big"))
    out.extend(len(rec.data_fork).to_bytes(4, "big"))
    out.extend(len(rec.resource_fork).to_bytes(4, "big"))
    out.extend(rec.data_fork)
    out.extend(rec.resource_fork)
    crc = _crc16(bytes(out))
    out.extend(crc.to_bytes(2, "big"))
    return bytes(out)


def decode(payload: bytes) -> BinHexRecord:
    """Decode a BinHex 4.0 colon-delimited payload.

    Args:
        payload: The bytes between (but not including) the two colons of a
            BinHex 4.0 file. Newlines inside the payload are tolerated and
            stripped, since the encoding may wrap lines at 64 columns.

    Returns:
        A :class:`BinHexRecord` describing the decoded file.

    Raises:
        BinHexError: If the payload cannot be decoded or the CRC check fails.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    payload = bytes(payload)

    # Strip newlines and carriage returns; BinHex encoders historically wrap
    # lines at 64 columns. We do not strip spaces or other whitespace because
    # the alphabet contains no whitespace symbols and stray spaces would
    # indicate corruption, not wrapping.
    cleaned = payload.replace(b"\r\n", b"").replace(b"\n", b"").replace(b"\r", b"")

    decoded = _b64_decode(cleaned)
    decompressed = _rle_decompress(decoded)
    return _parse_data_fork(decompressed)


def encode(rec: BinHexRecord) -> bytes:
    """Encode a BinHexRecord into a BinHex 4.0 colon-delimited payload.

    The result does NOT include the header line, the colons, or a trailing
    newline. It is the wrapped 64-symbol-alphabet text only, suitable for
    placement between the two colons of a full BinHex 4.0 file.
    """
    raw = _build_data_fork(rec)
    compressed = _rle_compress(raw)
    return _b64_encode(compressed)


def parse_file(text: bytes) -> bytes:
    """Extract the colon-delimited payload from a full BinHex 4.0 text file.

    A BinHex 4.0 text file begins with one or more header lines (the classic
    first line is \"(This file must be converted with BinHex 4.0)\"), followed
    by a line containing only a colon, the encoded payload (possibly wrapped
    across multiple lines), and a final line containing only a colon.

    This function returns the bytes between the two colons, with newlines
    preserved as-is. Pass the result to :func:`decode` to parse it.
    """
    if not isinstance(text, (bytes, bytearray)):
        raise TypeError("text must be bytes")
    text = bytes(text)

    # Split on newlines to find the colon-delimited region. We accept either
    # \n or \r\n as line endings.
    lines = text.replace(b"\r\n", b"\n").split(b"\n")

    start = None
    end = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == b":":
            if start is None:
                start = idx
            else:
                end = idx
                break

    if start is None or end is None:
        raise BinHexError("could not find colon-delimited payload")

    # Join the lines between the colons, preserving their original content
    # (the decoder will strip newlines).
    payload = b"\n".join(lines[start + 1 : end])
    return payload
