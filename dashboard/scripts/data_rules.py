from __future__ import annotations

import re

PRIVATE_PARTNER_RULES = {
    "Airbus": ["airbus"],
    "Atermes": ["atermes"],
    "Blacknut": ["blacknut"],
    "CLS": ["collecte localisation satellites", "collecte localisation satellite"],
    "Enedis": ["enedis"],
    "Eodyn": ["eodyn"],
    "Eviden": ["eviden"],
    "Imatag": ["imatag"],
    "Naval Group": ["naval group"],
    "Orange": ["orange"],
    "Ouest France": ["ouest france", "ouest-france"],
    "Purecontrol": ["purecontrol"],
    "Safran": ["safran"],
    "Thales": ["thales"],
    "Whispeak": ["whispeak"],
}

LAB_RULES = {
    "Lab-STICC": [
        "Laboratoire des sciences et techniques de l'information",
        "Lab-STICC",
    ],
    "IRISA": [
        "Institut de Recherche en Informatique et Systèmes Aléatoires",
        "IRISA",
    ],
    "Inria Rennes": [
        "Centre Inria de l'Université de Rennes",
        "Inria Rennes",
    ],
    "IETR": [
        "Institut d'Électronique et des Technologies du numéRique",
        "IETR",
    ],
    "LOPS": [
        "Laboratoire d'Océanographie Physique et Spatiale",
        "LOPS",
    ],
    "LaTIM": [
        "Laboratoire de Traitement de l'Information Médicale",
        "LaTIM",
    ],
    "LP3C": ["Laboratoire de Psychologie", "LP3C"],
    "LMBA": ["Laboratoire de Mathématiques de Bretagne Atlantique", "LMBA"],
    "IRMAR": ["Institut de Recherche Mathématique de Rennes", "IRMAR"],
    "LETG": ["Littoral, Environnement, Télédétection, Géomatique", "LETG"],
}


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return " ".join(text.split())


def split_values(value, kind: str) -> list[str]:
    text = clean_text(value)
    if not text:
        return []

    if kind == "authors":
        parts = re.split(r"\s*\|\s*|\s*,\s*", text)
    elif kind in {"keywords", "domains"}:
        parts = re.split(r"\s*;\s*|\s*\|\s*", text)
    else:
        parts = re.split(r"\s*\|\s*", text)

    result: list[str] = []
    seen: set[str] = set()

    for part in parts:
        item = clean_text(part)
        if item and item not in seen:
            seen.add(item)
            result.append(item)

    return result


def infer_private_partners(institutions) -> list[str]:
    text = clean_text(institutions).casefold()
    if not text:
        return []

    found = []
    for partner, patterns in PRIVATE_PARTNER_RULES.items():
        if any(pattern.casefold() in text for pattern in patterns):
            found.append(partner)
    return found


def normalize_lab(name: str) -> str | None:
    text = clean_text(name)
    if not text:
        return None

    lowered = text.casefold()
    for short_name, patterns in LAB_RULES.items():
        if any(pattern.casefold() in lowered for pattern in patterns):
            return short_name
    return None
