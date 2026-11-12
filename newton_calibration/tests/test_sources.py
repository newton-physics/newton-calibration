# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sys
import types
import unittest
from unittest import mock

from newton_calibration.sources import FileSourceHuggingFace, FileSourceLocal, file_sha256
from newton_calibration.tests.unittest_utils import TempPathTestCase

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class TestFileSources(TempPathTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp_path / "repo"
        (self.root / "data").mkdir(parents=True)
        self.file = self.root / "data" / "a.bin"
        self.file.write_bytes(b"recorded")
        self.digest = hashlib.sha256(b"recorded").hexdigest()

    def test_local_source_returns_files_that_match_their_declared_digest(self):
        """Fetch declared files in order and reject changed contents."""
        source = FileSourceLocal(self.root)
        self.assertEqual(file_sha256(self.file), self.digest)
        self.assertEqual(source.fetch_all({"data/a.bin": self.digest}), {"data/a.bin": self.file.resolve()})
        self.file.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "differs from the declared"):
            source.fetch("data/a.bin", self.digest)
        self.assertEqual(source.provenance, {"provider": "local"})

    def test_digest_and_path_must_be_declared_and_inside_the_root(self):
        """Require a full digest and keep relative paths under the root."""
        source = FileSourceLocal(self.root)
        for digest in ("", self.digest[:20], self.digest.upper(), None):
            with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, "sha256"):
                source.fetch("data/a.bin", digest)
        with self.assertRaisesRegex(ValueError, "leaves the source root"):
            source.locate("../outside.bin")

    def test_hugging_face_source_downloads_the_pinned_commit(self):
        """Download through huggingface_hub at the pinned commit and record provenance."""
        calls = []

        def download(repo_id, filename, *, repo_type, revision):
            calls.append((repo_id, filename, repo_type, revision))
            return str(self.root / filename)

        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = download
        source = FileSourceHuggingFace("organization/name", COMMIT)
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertEqual(source.fetch("data/a.bin", self.digest), self.file)
        self.assertEqual(calls, [("organization/name", "data/a.bin", "dataset", COMMIT)])
        self.assertEqual(
            source.provenance,
            {"provider": "huggingface", "repo_id": "organization/name", "repo_type": "dataset", "revision": COMMIT},
        )

    def test_hugging_face_source_requires_an_immutable_revision(self):
        """Reject branch names, short hashes, unknown repository types and empty IDs."""
        for revision in ("main", COMMIT[:12], COMMIT.upper()):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "commit hash"):
                FileSourceHuggingFace("organization/name", revision)
        with self.assertRaisesRegex(ValueError, "repo_type"):
            FileSourceHuggingFace("organization/name", COMMIT, repo_type="bucket")
        with self.assertRaisesRegex(ValueError, "repo_id"):
            FileSourceHuggingFace("", COMMIT)

    def test_missing_hugging_face_package_names_the_dependency(self):
        """Explain which package a Hugging Face fetch needs."""
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            with self.assertRaisesRegex(ImportError, "huggingface-hub"):
                FileSourceHuggingFace("organization/name", COMMIT).locate("data/a.bin")


if __name__ == "__main__":
    unittest.main()
