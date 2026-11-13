# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pinned, sha256-checked source files from a local directory or a Hugging Face repository."""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FileSource(ABC):
    """Locates source files by relative path and checks each against a declared SHA-256.

    Subclasses locate files; :meth:`fetch` verifies them. Every file must be declared
    with its digest, so a run reads exactly the recorded bytes.
    """

    @abstractmethod
    def locate(self, relative: str) -> Path:
        """Return a local path for ``relative`` without checking its contents."""

    @property
    @abstractmethod
    def provenance(self) -> dict:
        """Where the files come from, suitable for :class:`~newton_calibration.Evidence` ``sources``."""

    def fetch(self, relative: str, sha256: str) -> Path:
        """Return the local path of ``relative`` after checking its SHA-256.

        Raises:
            ValueError: If ``sha256`` is not a lowercase 64-digit hexadecimal digest or
                the file's digest differs from it.
        """
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise ValueError(f"{relative}: declare a lowercase hexadecimal sha256, got {sha256!r}")
        path = self.locate(relative)
        if (actual := file_sha256(path)) != sha256:
            raise ValueError(f"{relative}: sha256 {actual} differs from the declared {sha256}")
        return path

    def fetch_all(self, files: Mapping[str, str]) -> dict[str, Path]:
        """Fetch every ``{relative path: sha256}`` entry, preserving order."""
        return {relative: self.fetch(relative, sha256) for relative, sha256 in files.items()}


class FileSourceLocal(FileSource):
    """Files under a local directory, for example an offline copy of a repository."""

    def __init__(self, root: str | Path):
        self.root = Path(os.path.abspath(Path(root).expanduser()))

    def locate(self, relative: str) -> Path:
        # Normalize without following symlinks: cached files are often links to
        # extensionless blobs, and readers pick the format from the declared name.
        path = Path(os.path.normpath(self.root / relative))
        if not path.is_relative_to(self.root):
            raise ValueError(f"{relative}: path leaves the source root")
        return path

    @property
    def provenance(self) -> dict:
        return {"provider": "local"}


class FileSourceHuggingFace(FileSource):
    """Files from a Hugging Face Hub repository at a pinned commit.

    Downloads go through ``huggingface_hub`` into its cache, which is imported on the
    first fetch. Install ``huggingface-hub``, for example through the ``huggingface`` extra.

    Args:
        repo_id: Repository ID, for example ``"organization/name"``.
        revision: Full 40-digit commit hash; branch and tag names are rejected because
            they can move.
        repo_type: ``"dataset"``, ``"model"`` or ``"space"``.
    """

    def __init__(self, repo_id: str, revision: str, *, repo_type: str = "dataset"):
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("repo_id must be a nonempty string")
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"revision must be a full 40-digit commit hash, got {revision!r}")
        if repo_type not in {"dataset", "model", "space"}:
            raise ValueError(f"unsupported repo_type {repo_type!r}")
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type

    def locate(self, relative: str) -> Path:
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415 (optional dependency)
        except ImportError as error:
            raise ImportError("FileSourceHuggingFace requires the huggingface-hub package") from error
        return Path(hf_hub_download(self.repo_id, relative, repo_type=self.repo_type, revision=self.revision))

    @property
    def provenance(self) -> dict:
        return {
            "provider": "huggingface",
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
        }
