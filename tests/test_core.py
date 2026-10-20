import struct
import unittest

from bin_hex_4_0_decoder import BinHexError, BinHexRecord, decode, encode
from bin_hex_4_0_decoder.core import (
    _b64_decode,
    _b64_encode,
    _build_data_fork,
    _crc16,
    _parse_data_fork,
    _rle_compress,
    _rle_decompress,
    parse_file,
)


class TestCRC16(unittest.TestCase):
    def test_known_vector(self):
        # The CRC-16/CCITT (0x1021, init 0x0000) of "123456789" is 0x31C3.
        self.assertEqual(_crc16(b"123456789"), 0x31C3)

    def test_empty_input_is_zero(self):
        self.assertEqual(_crc16(b""), 0x0000)


class TestRLE(unittest.TestCase):
    def test_round_trip_no_runs(self):
        data = b"Hello, World!"
        self.assertEqual(_rle_decompress(_rle_compress(data)), data)

    def test_literal_0x90_escaped(self):
        data = bytes([0x90])
        compressed = _rle_compress(data)
        self.assertEqual(compressed, bytes([0x90, 0x00]))
        self.assertEqual(_rle_decompress(compressed), data)

    def test_run_of_four_is_compressed(self):
        data = b"AAAA"
        compressed = _rle_compress(data)
        self.assertEqual(compressed, bytes([ord("A"), 0x90, 0x04]))
        self.assertEqual(_rle_decompress(compressed), data)

    def test_run_of_three_not_compressed(self):
        data = b"AAA"
        compressed = _rle_compress(data)
        self.assertEqual(compressed, b"AAA")

    def test_long_run_split_at_255(self):
        data = b"B" * 300
        compressed = _rle_compress(data)
        # 255 + 45 -> two runs: [B 0x90 255] [B 0x90 45]
        self.assertEqual(
            compressed,
            bytes([ord("B"), 0x90, 255, ord("B"), 0x90, 45]),
        )
        self.assertEqual(_rle_decompress(compressed), data)

    def test_decompress_run_marker_at_end_raises(self):
        with self.assertRaises(BinHexError):
            _rle_decompress(bytes([0x90]))

    def test_decompress_run_with_no_preceding_byte_raises(self):
        with self.assertRaises(BinHexError):
            _rle_decompress(bytes([0x90, 0x05]))


class TestB64(unittest.TestCase):
    def test_round_trip_empty(self):
        self.assertEqual(_b64_decode(_b64_encode(b"")), b"")

    def test_round_trip_one_byte(self):
        data = b"\x41"
        self.assertEqual(_b64_decode(_b64_encode(data)), data)

    def test_round_trip_two_bytes(self):
        data = b"\x41\x42"
        self.assertEqual(_b64_decode(_b64_encode(data)), data)

    def test_round_trip_three_bytes(self):
        data = b"\x41\x42\x43"
        self.assertEqual(_b64_decode(_b64_encode(data)), data)

    def test_round_trip_long(self):
        data = bytes(range(256))
        self.assertEqual(_b64_decode(_b64_encode(data)), data)

    def test_invalid_symbol_raises(self):
        # '!' is not in the BinHex alphabet.
        with self.assertRaises(BinHexError):
            _b64_decode(b"0!ABCD")

    def test_invalid_padding_count_raises(self):
        # A leading symbol with value 3 is an invalid padding count.
        # 'D' -> 3 in the alphabet (0-9,A-Z,a-z,+,- -> 0..63), so 'D' is value 3.
        with self.assertRaises(BinHexError):
            _b64_decode(b"D" + b"0" * 4)

    def test_body_not_multiple_of_four_raises(self):
        # '0' -> 0, valid padding. Then 3 symbols is not a multiple of 4.
        with self.assertRaises(BinHexError):
            _b64_decode(b"0ABC")


class TestRecordParsing(unittest.TestCase):
    def _make_record(
        self,
        name="test.txt",
        version=0,
        type_code=b"TEXT",
        author_code=b"ttxt",
        flags=0,
        data_fork=b"Hello",
        resource_fork=b"",
    ):
        return BinHexRecord(
            name=name,
            version=version,
            type=type_code,
            author=author_code,
            flags=flags,
            data_fork=data_fork,
            resource_fork=resource_fork,
            crc=0,
        )

    def test_round_trip_record(self):
        rec = self._make_record()
        raw = _build_data_fork(rec)
        parsed = _parse_data_fork(raw)
        self.assertEqual(parsed.name, "test.txt")
        self.assertEqual(parsed.version, 0)
        self.assertEqual(parsed.type, b"TEXT")
        self.assertEqual(parsed.author, b"ttxt")
        self.assertEqual(parsed.flags, 0)
        self.assertEqual(parsed.data_fork, b"Hello")
        self.assertEqual(parsed.resource_fork, b"")
        # The CRC is recomputed on build; the parsed CRC should match.
        self.assertEqual(parsed.crc, _crc16(raw[:-2]))

    def test_crc_mismatch_raises(self):
        rec = self._make_record()
        raw = bytearray(_build_data_fork(rec))
        # Flip a bit in the data fork.
        raw[20] ^= 0x01
        with self.assertRaises(BinHexError):
            _parse_data_fork(bytes(raw))

    def test_truncated_header_raises(self):
        rec = self._make_record()
        raw = _build_data_fork(rec)
        with self.assertRaises(BinHexError):
            _parse_data_fork(raw[:5])

    def test_truncated_fork_raises(self):
        rec = self._make_record(data_fork=b"X" * 100)
        raw = _build_data_fork(rec)
        # Remove the last byte (part of resource fork or CRC).
        with self.assertRaises(BinHexError):
            _parse_data_fork(raw[:-1])

    def test_long_name(self):
        rec = self._make_record(name="a" * 63)
        raw = _build_data_fork(rec)
        parsed = _parse_data_fork(raw)
        self.assertEqual(parsed.name, "a" * 63)

    def test_non_ascii_name_raises_on_build(self):
        rec = self._make_record(name="résumé")
        with self.assertRaises(UnicodeEncodeError):
            _build_data_fork(rec)

    def test_name_too_long_raises_on_build(self):
        rec = self._make_record(name="a" * 64)
        with self.assertRaises(ValueError):
            _build_data_fork(rec)


class TestEncodeDecodeIntegration(unittest.TestCase):
    def test_round_trip_payload(self):
        rec = BinHexRecord(
            name="hello.txt",
            version=0,
            type=b"TEXT",
            author=b"ttxt",
            flags=0,
            data_fork=b"Hello, World!",
            resource_fork=b"",
            crc=0,
        )
        payload = encode(rec)
        decoded = decode(payload)
        self.assertEqual(decoded.name, "hello.txt")
        self.assertEqual(decoded.type, b"TEXT")
        self.assertEqual(decoded.author, b"ttxt")
        self.assertEqual(decoded.data_fork, b"Hello, World!")
        self.assertEqual(decoded.resource_fork, b"")

    def test_round_trip_with_resource_fork(self):
        rec = BinHexRecord(
            name="icon",
            version=0,
            type=b"ICN#",
            author=b"MACS",
            flags=0x0200,
            data_fork=b"",
            resource_fork=b"\x00\x01\x02\x03",
            crc=0,
        )
        payload = encode(rec)
        decoded = decode(payload)
        self.assertEqual(decoded.resource_fork, b"\x00\x01\x02\x03")
        self.assertEqual(decoded.flags, 0x0200)

    def test_decode_strips_newlines_in_payload(self):
        rec = BinHexRecord(
            name="x",
            version=0,
            type=b"TEXT",
            author=b"ttxt",
            flags=0,
            data_fork=b"data",
            resource_fork=b"",
            crc=0,
        )
        payload = encode(rec)
        # Insert newlines every few characters.
        wrapped = b"\n".join(payload[i : i + 4] for i in range(0, len(payload), 4))
        decoded = decode(wrapped)
        self.assertEqual(decoded.data_fork, b"data")

    def test_decode_rejects_non_bytes(self):
        with self.assertRaises(TypeError):
            decode("not bytes")  # type: ignore[arg-type]


class TestParseFile(unittest.TestCase):
    def test_extracts_payload(self):
        rec = BinHexRecord(
            name="x",
            version=0,
            type=b"TEXT",
            author=b"ttxt",
            flags=0,
            data_fork=b"data",
            resource_fork=b"",
            crc=0,
        )
        payload = encode(rec)
        full = (
            b"(This file must be converted with BinHex 4.0)\r\n"
            b"\r\n"
            b":\n"
            + payload
            + b"\n:\n"
        )
        extracted = parse_file(full)
        self.assertEqual(extracted, payload)
        decoded = decode(extracted)
        self.assertEqual(decoded.data_fork, b"data")

    def test_missing_colons_raises(self):
        with self.assertRaises(BinHexError):
            parse_file(b"just some text without colons")

    def test_rejects_non_bytes(self):
        with self.assertRaises(TypeError):
            parse_file("not bytes")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
