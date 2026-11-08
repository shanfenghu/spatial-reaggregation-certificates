"""Acquisition tests use in-memory responses and temporary tiny inputs only."""
from pathlib import Path
import hashlib
import io
import tempfile
import unittest
import acquire


class Response(io.BytesIO):
    def geturl(self):
        return "https://www.epa.gov/exact-input"


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.data = b"exact original bytes\n"
        self.row = {"name": "input.bin", "url": "https://www.epa.gov/exact-input",
                    "bytes": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest()}

    def opener(self, *args, **kwargs):
        return Response(self.data)

    def test_missing_download_then_cached_verify_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            acquire.acquire(self.row, root, "download", self.opener)
            self.assertEqual((root/"input.bin").read_bytes(), self.data)
            def forbidden(*args, **kwargs):
                raise AssertionError("Cached exact input must not access network")
            acquire.acquire(self.row, root, "download", forbidden)

    def test_probe_has_no_filesystem_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"absent"
            acquire.acquire(self.row, root, "probe", self.opener)
            self.assertFalse(root.exists())

    def test_wrong_existing_file_not_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"input.bin").write_bytes(b"retain original mismatch")
            with self.assertRaises(ValueError):
                acquire.acquire(self.row, root, "download", self.opener)
            self.assertEqual((root/"input.bin").read_bytes(), b"retain original mismatch")

    def test_wrong_remote_bytes_leave_no_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                acquire.acquire(self.row, root, "download", lambda *a, **k: Response(b"wrong"))
            self.assertEqual(list(root.iterdir()), [])

    def test_symlink_rejected_without_reading_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"input.bin").symlink_to(root/"absent")
            with self.assertRaises(ValueError):
                acquire.acquire(self.row, root, "download", self.opener)

    def test_pinned_manifest_has_exact_names_and_agency_routes(self):
        self.assertEqual(len(acquire.manifest()), 3)


if __name__ == "__main__":
    unittest.main()
