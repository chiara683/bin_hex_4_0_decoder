# bin_hex_4_0_decoder

Decodes and encodes Apple BinHex 4.0 files, a legacy Macintosh binary-to-text encoding format. The library handles the 64-symbol alphabet, RLE compression, the data-fork header layout, and CRC-16 verification.

```python
from bin_hex_4_0_decoder import BinHexRecord, decode, encode

new_record = BinHexRecord(
    name="out.txt",
    version=0,
    type=b"TEXT",
    author=b"ttxt",
    flags=0,
    data_fork=b"Hello, World!",
    resource_fork=b"",
    crc=0,
)
new_payload = encode(new_record)
decoded = decode(new_payload)
print(decoded.name, decoded.type, decoded.author)
print(len(decoded.data_fork), "data bytes")
```

## Why this exists

BinHex 4.0 files show up in old Macintosh software archives, and the format has enough moving parts (a custom alphabet, RLE, a structured header, and a CRC) that parsing one by hand is error-prone. This library exists to do that parsing correctly and reversibly.

The main design decision: the `decode` function takes the bytes *between* the two colons of a BinHex file, not the full text including the header line. Real-world files vary in how many header lines precede the first colon, so treating the header as someone else's problem keeps the decoder deterministic. If you have a full BinHex text file, call `parse_file` to strip the header and extract the payload first. The `encode` function returns only the colon-delimited payload text, without the header line or trailing newline.

## Edge cases

- **Newline wrapping**: BinHex encoders historically wrap the payload at 64 columns. `decode` strips `\n` and `\r` from the payload before decoding. It does not strip spaces or tabs — those are not part of the alphabet and would indicate corruption.
- **RLE threshold**: The encoder only compresses runs of 4 or more identical bytes. Runs of exactly 3 are emitted literally, because the RLE encoding is 3 bytes long and offers no savings. This matches the classic Macintosh behavior.
- **Name encoding**: Filenames are encoded as ASCII. Non-ASCII names will raise `UnicodeEncodeError` on encode. Names longer than 63 bytes raise `ValueError`.
- **CRC variant**: Uses CRC-16 with polynomial 0x1021, initial value 0x0000, no final XOR. This is the variant BinHex 4.0 uses; it is not the same as the more common 0xFFFF-init CRC-16/CCITT.
