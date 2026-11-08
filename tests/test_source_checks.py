"""Exercise real path escapes and source drift at manifest read boundaries."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import reproduce


class SourceBoundaryTests(unittest.TestCase):
    def test_parent_and_absolute_paths_cannot_name_external_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();root=base/'source';root.mkdir()
            outside=base/'outside';outside.write_bytes(b'preserve external bytes')
            for name in ['../outside',str(outside),'a/../outside','a\\outside','./outside']:
                with self.subTest(name=name),self.assertRaises(ValueError):
                    reproduce.checked_file(root,name)
            self.assertEqual(outside.read_bytes(),b'preserve external bytes')

    def test_directory_and_manifest_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();root=base/'source';root.mkdir()
            outside=base/'outside';outside.write_text('external')
            (root/'linked').symlink_to(base,target_is_directory=True)
            with self.assertRaises(ValueError):reproduce.checked_file(root,'linked/outside')
            (root/'checksums.json').symlink_to(outside)
            with patch.object(reproduce,'ROOT',root),self.assertRaises(ValueError):
                reproduce.verify_source()

    def test_valid_manifest_verifies_and_changed_bytes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();path=root/'valid';path.write_bytes(b'fixed source')
            manifest={'valid':hashlib.sha256(path.read_bytes()).hexdigest()}
            (root/'checksums.json').write_text(json.dumps(manifest))
            with patch.object(reproduce,'ROOT',root):
                reproduce.verify_source()
                path.write_bytes(b'changed source')
                with self.assertRaises(ValueError):reproduce.verify_source()
                manifest={'../outside':'0'*64}
                (root/'checksums.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):reproduce.verify_source()


if __name__=='__main__':unittest.main()
