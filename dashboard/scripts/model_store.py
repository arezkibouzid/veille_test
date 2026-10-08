"""Immutable model bundles; the active version is recorded in SQLite."""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path

from config.config import ARTIFACT_DIR
from dashboard.scripts.db import connect_db


def bundle_paths(directory: Path) -> dict:
    return {
        'directory': directory,
        'classifier': directory / 'sgd_classifier.joblib',
        'encoder': directory / 'mpnet_encoder',
        'references': directory / 'subaxis_references.joblib',
        'metadata': directory / 'metadata.json',
    }


def active_bundle() -> tuple[str, dict]:
    with connect_db(readonly=True) as con:
        rows = con.execute(
            "SELECT version, artifact_path FROM model_versions WHERE status='active'").fetchall()
    if len(rows) > 1:
        raise RuntimeError('Plusieurs modèles actifs en SQLite ; activation à corriger.')
    if rows:
        if not rows[0]['artifact_path']:
            raise ValueError('Le modèle actif ne possède pas de chemin de bundle en SQLite.')
        directory = Path(rows[0]['artifact_path'])
        if not directory.is_absolute():
            directory = ARTIFACT_DIR / directory
        paths = bundle_paths(directory)
        for key in ('classifier', 'encoder', 'references'):
            if not paths[key].exists():
                raise FileNotFoundError(f"Artefact du modèle actif absent : {paths[key]}")
        return rows[0]['version'], paths
    paths = bundle_paths(ARTIFACT_DIR)
    for key in ('classifier', 'encoder', 'references'):
        if not paths[key].exists():
            raise FileNotFoundError(f"Artefact du modèle absent : {paths[key]}")
    digest = hashlib.sha256()
    for key in ('classifier', 'references', 'metadata'):
        if paths[key].is_file():
            digest.update(paths[key].read_bytes())
    # Include the encoder weights/configuration in the version fingerprint.
    for file in sorted(paths['encoder'].rglob('*')):
        if file.is_file():
            digest.update(str(file.relative_to(paths['encoder'])).encode())
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
    version = 'legacy-' + digest.hexdigest()[:16]
    directory = ARTIFACT_DIR / 'versions' / version
    if not directory.exists():
        temporary = directory.parent / ('.' + version + '-' + uuid.uuid4().hex)
        temporary.mkdir(parents=True, exist_ok=False)
        for key in ('classifier', 'references', 'metadata'):
            if paths[key].is_file():
                shutil.copy2(paths[key], temporary / paths[key].name)
        shutil.copytree(paths['encoder'], temporary / 'mpnet_encoder')
        os.replace(temporary, directory)
    with connect_db() as con:
        con.execute('BEGIN IMMEDIATE')
        active = con.execute("SELECT version,artifact_path FROM model_versions WHERE status='active'").fetchone()
        if active:
            current = Path(active['artifact_path'])
            return active['version'], bundle_paths(current if current.is_absolute() else ARTIFACT_DIR / current)
        con.execute(
            "INSERT INTO model_versions(version,status,artifact_path,metrics_json) "
            "VALUES (?, 'active', ?, '{}')", (version, str(directory.relative_to(ARTIFACT_DIR))))
    return version, bundle_paths(directory)


def load_bundle(*, include_classifier=True):
    import joblib
    from sentence_transformers import SentenceTransformer
    version, paths = active_bundle()
    encoder = SentenceTransformer(str(paths['encoder']), local_files_only=True)
    classifier = joblib.load(paths['classifier']) if include_classifier else None
    references = joblib.load(paths['references'])
    return version, encoder, classifier, references
