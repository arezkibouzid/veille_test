"""Create a consistent SQLite backup and atomically publish it."""
from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

try:
    from .db import database_path
except ImportError:
    from db import database_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / 'dashboard/data/snapshots/sequoia_v2_snapshot.db'


def snapshot_path():
    return Path(os.getenv('SQLITE_SNAPSHOT_PATH') or DEFAULT_SNAPSHOT).expanduser()


def create_snapshot(source: Path, destination: Path):
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination or (source.exists() and destination.exists() and os.path.samefile(source, destination)):
        raise ValueError('Le snapshot ne peut pas remplacer la base vivante.')
    if not source.is_file():
        raise FileNotFoundError(f'Base SQLite source introuvable : {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.snapshot-', suffix='.db', dir=destination.parent, delete=False) as temp:
        temporary = Path(temp.name)
    src = sqlite3.connect(f'{source.as_uri()}?mode=ro', uri=True, timeout=30)
    dst = sqlite3.connect(temporary, timeout=30)
    try:
        src.backup(dst)
        result = dst.execute('PRAGMA integrity_check').fetchone()
        if not result or result[0] != 'ok':
            raise RuntimeError(f'Échec integrity_check : {result}')
        if dst.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise RuntimeError('Le snapshot contient des relations invalides.')
        dst.commit()
    finally:
        dst.close()
        src.close()
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(create_snapshot(database_path(), args.output or snapshot_path()))


if __name__ == '__main__':
    main()
