"""Direct tests for media_downloader's SSRF guards -- no network access
needed, since validation happens (and raises) before any request is sent.
Literal IPs and localhost resolve without a real DNS lookup."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import media_downloader  # noqa: E402


class BlockedTargetTests(unittest.TestCase):
    BLOCKED_URLS = [
        "http://127.0.0.1/x.jpg",
        "http://localhost/x.jpg",
        "http://0.0.0.0/x.jpg",
        "http://169.254.169.254/latest/meta-data/",   # cloud metadata endpoint
        "http://10.0.0.5/x.jpg",
        "http://172.16.0.1/x.jpg",
        "http://192.168.1.1/x.jpg",
        "http://[::1]/x.jpg",
    ]

    def test_private_and_internal_targets_are_rejected(self):
        for url in self.BLOCKED_URLS:
            with self.subTest(url=url):
                with self.assertRaises(media_downloader.UnsafeURL):
                    media_downloader.fetch_url(url, max_bytes=1024, timeout=1.0)

    def test_disallowed_schemes_are_rejected(self):
        for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://x"):
            with self.subTest(url=url):
                with self.assertRaises(media_downloader.UnsafeURL):
                    media_downloader.fetch_url(url, max_bytes=1024, timeout=1.0)

    def test_url_with_no_host_is_rejected(self):
        with self.assertRaises(media_downloader.UnsafeURL):
            media_downloader.fetch_url("http:///no-host", max_bytes=1024, timeout=1.0)


class Base64DecodeTests(unittest.TestCase):
    def test_invalid_base64_raises(self):
        with self.assertRaises(media_downloader.InvalidBase64):
            media_downloader.decode_base64("not valid base64!!", max_bytes=1024)

    def test_oversized_payload_raises(self):
        import base64
        big = base64.b64encode(b"x" * 100).decode()
        with self.assertRaises(media_downloader.PayloadTooLarge):
            media_downloader.decode_base64(big, max_bytes=10)

    def test_empty_payload_raises(self):
        with self.assertRaises(media_downloader.InvalidBase64):
            media_downloader.decode_base64("", max_bytes=1024)

    def test_valid_payload_decodes(self):
        import base64
        data = media_downloader.decode_base64(base64.b64encode(b"hello").decode(), max_bytes=1024)
        self.assertEqual(data, b"hello")


if __name__ == "__main__":
    unittest.main(verbosity=2)
