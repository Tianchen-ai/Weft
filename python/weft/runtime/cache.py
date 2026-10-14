from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from .driver import CompilationError


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def publish_json(destination: Path, value: dict) -> Path:
    """Publish a complete JSON record before making it visible to readers."""
    destination = destination.absolute()
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".record-", dir=destination.parent,
                                     delete=False) as temporary:
        path = Path(temporary.name)
        try:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(path, destination)
        finally:
            path.unlink(missing_ok=True)
    descriptor = os.open(destination.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


class ArtifactCache:
    """Exact build identities with atomic publication and one lock per program."""

    def __init__(self, directory: str | Path | None = None) -> None:
        if directory is None:
            directory = os.environ.get("WEFT_CACHE_DIR")
        if directory is None:
            directory = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "weft" / "artifacts"
        self.directory = Path(directory).expanduser().absolute()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "locks").mkdir(exist_ok=True)
        (self.directory / "entries").mkdir(exist_ok=True)
        (self.directory / "records").mkdir(exist_ok=True)
        with self._database() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS artifacts "
                               "(identity TEXT PRIMARY KEY, slot TEXT UNIQUE NOT NULL)")

    @contextmanager
    def _database(self):
        connection = sqlite3.connect(self.directory / "index.sqlite", timeout=120)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _slot(self, identity: str) -> str:
        with self._database() as connection:
            connection.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?)",
                               (identity, uuid.uuid4().hex))
            return connection.execute("SELECT slot FROM artifacts WHERE identity = ?", (identity,)).fetchone()[0]

    def get_or_build(self, identity: dict, builder: Callable[[Path], None]) -> tuple[Path, bool]:
        serialized = canonical_json(identity)
        slot = self._slot(serialized)
        destination = self.directory / "entries" / slot
        with (self.directory / "locks" / slot).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if destination.exists():
                identity_path = destination / "identity.json"
                if not identity_path.is_file() or identity_path.read_text() != serialized:
                    raise CompilationError(f"cache entry has an inconsistent build identity: {destination}")
                if not (destination / "manifest.json").is_file():
                    raise CompilationError(f"cache entry has no completed manifest: {destination}")
                return destination, True
            with tempfile.TemporaryDirectory(prefix=f".{slot}-", dir=destination.parent) as temporary:
                staging = Path(temporary) / "artifact"
                staging.mkdir()
                builder(staging)
                if not (staging / "manifest.json").is_file():
                    raise CompilationError("artifact builder did not produce a completed manifest")
                (staging / "identity.json").write_text(serialized, encoding="utf-8")
                for path in staging.iterdir():
                    if path.is_file():
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                os.rename(staging, destination)
                descriptor = os.open(destination.parent, os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return destination, False

    def entries(self) -> tuple[Path, ...]:
        """List committed entries only; this does not load executable code."""
        return tuple(path for path in sorted((self.directory / "entries").iterdir())
                     if not path.name.startswith(".") and (path / "manifest.json").is_file())

    def record(self, identity: dict, producer: Callable[[], dict], *, refresh: bool = False) -> tuple[dict, bool, Path]:
        """Publish an atomic selection record using the same exact-key index/locks.

        Refresh replaces a record only after its producer succeeds. A failed
        refresh propagates the error and leaves the previous record intact.
        """
        serialized = canonical_json(identity)
        slot = self._slot(serialized)
        destination = self.directory / "records" / (slot + ".json")
        with (self.directory / "locks" / slot).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if destination.exists() and not refresh:
                stored = json.loads(destination.read_text(encoding="utf-8"))
                if stored["identity"] != serialized:
                    raise CompilationError(f"selection record has an inconsistent build identity: {destination}")
                return stored["record"], True, destination
            record = producer()
            publish_json(destination, {"identity": serialized, "record": record})
            return record, False, destination
