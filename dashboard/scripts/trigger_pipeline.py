"""Déclenche un traitement sur la VM SequoIA via l'API de validation et attend sa fin.

Exécuté par GitHub Actions (.github/workflows/hal-sync.yml). Le traitement
« predict » interroge l'API HAL depuis la VM, enrichit et classe les nouvelles
publications : toutes les données restent dans la base SQLite de la VM.

Variables d'environnement : SEQUOIA_API_URL, SEQUOIA_API_USER, SEQUOIA_API_PASSWORD.
Bibliothèque standard uniquement, pour s'exécuter sans installation.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request


def call(method: str, path: str) -> dict:
    token = base64.b64encode(
        f"{os.environ['SEQUOIA_API_USER']}:{os.environ['SEQUOIA_API_PASSWORD']}".encode()).decode()
    request = urllib.request.Request(
        os.environ["SEQUOIA_API_URL"].rstrip("/") + path, method=method,
        headers={"Authorization": f"Basic {token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("detail")
        except ValueError:
            detail = None
        raise SystemExit(f"{method} {path} : HTTP {exc.code} — {detail or exc.reason}") from None


def wait(job_id: str, timeout: int, interval: int = 20) -> dict:
    deadline = time.monotonic() + timeout
    step = None
    while True:
        try:
            job = call("GET", f"/api/pipelines/jobs/{job_id}")
        except urllib.error.URLError as exc:  # coupure passagère : le traitement continue sur la VM
            print(f"API injoignable ({exc.reason}), nouvel essai…", flush=True)
            job = {"status": "running", "step": step}
        if job["step"] != step:
            step = job["step"]
            print(step, flush=True)
        if job["status"] != "running":
            return job
        if time.monotonic() > deadline:
            raise SystemExit(f"Traitement {job_id} toujours en cours après {timeout} s.")
        time.sleep(interval)


def markdown_summary(kind: str, job: dict) -> str:
    result = job.get("result") or {}
    lines = [f"## Traitement « {kind} » : {job['status']}", ""]
    hal = result.get("hal")
    if hal:
        lines += [
            f"- Notices HAL reçues : **{hal.get('received')}**",
            f"- Nouvelles publications : **{hal.get('inserted')}**",
            f"- Notices mises à jour : **{hal.get('updated')}**",
            f"- Articles en base : **{hal.get('total')}**",
        ]
    if "predicted" in result:
        lines.append(f"- Publications classées, à valider : **{result['predicted']}**")
    if result.get("model_version"):
        lines.append(f"- Modèle : `{result['model_version']}`")
    if job.get("error"):
        lines.append(f"- Erreur : {job['error']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Déclenche un traitement SequoIA sur la VM")
    parser.add_argument("kind", nargs="?", default="predict", choices=["predict", "retrain", "report"])
    parser.add_argument("--timeout", type=int, default=3600, help="Attente maximale en secondes")
    args = parser.parse_args()

    job_id = call("POST", f"/api/pipelines/{args.kind}")["job_id"]
    print(f"Traitement {args.kind} démarré : {job_id}", flush=True)
    job = wait(job_id, args.timeout)

    summary = markdown_summary(args.kind, job)
    print(summary)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(summary)
    if job["status"] != "succeeded":
        sys.exit(1)


if __name__ == "__main__":
    main()
