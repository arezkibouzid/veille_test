from __future__ import annotations

import logging
from typing import Any

import requests

HAL_API_URL = "https://api.archives-ouvertes.fr/search/SEQUOIA/"
MAX_ROWS_PER_PAGE = 1000
SEQUOIA_AXES = {
    "Core AI": "core AI foundations learning secure trustworthy frugal embedded hybrid explainable AI hardware signal image language robotics theory human AI interaction regulation ethics",
    "AI, Cybersecurity and Defense": "AI cybersecurity defense systems infrastructures information intelligence digital influence security adversarial attacks privacy anomaly detection networks cyber defense disinformation robotics",
    "AI, Environment and Ocean": "AI environment ocean earth observation modeling climate remote sensing digital twins data assimilation uncertainty marine pollution maritime traffic ocean dynamics",
}

SEQUOIA_SAXES = {

    "Core AI": {

        "Theoretical and Conceptual Foundations of AI": [
            "generalization",
            "statistical bounds",
            "continual learning",
            "optimal transport",
            "long-tail distributions",
            "rare events",
            "time series",
            "graphs",
            "geometric deep learning",
        ],

        "Symbolic / Statistical Hybridization": [
            "hybrid AI",
            "neuro-symbolic AI",
            "knowledge graphs",
            "counterfactual explanations",
            "epistemic planning",
            "reasoning",
        ],

        "Formal Evaluation of ML Technologies": [
            "formal methods",
            "model checking",
            "certification",
            "certified robustness",
            "neural network verification",
            "static analysis",
        ],

        "Frugal AI, Edge AI and Hardware Architectures": [
            "frugal AI",
            "quantization",
            "sparsity",
            "AI accelerators",
            "model compression",
            "energy efficiency",
            "edge AI",
            "compilation",
        ],

        "Human–AI Interaction and Explainability": [
            "XAI",
            "explainability",
            "trust",
            "confidence measures",
            "human-AI interaction",
            "attention mechanisms",
            "SHAP",
            "interactive learning",
        ],

        "Acceptability, Responsibility, Regulation and Ethics": [
            "AI Act",
            "AI regulation",
            "responsibility",
            "bias",
            "AI ethics",
            "law",
            "organizations",
            "AI use",
            "acceptability",
        ],

        "Signal, Image and Language": [
            "signal processing",
            "computer vision",
            "natural language processing",
            "NLP",
            "large language models",
            "LLMs",
            "diffusion models",
            "multimodal AI",
            "vision-language models",
            "image compression",
        ],
    },


    "AI, Cybersecurity and Defense": {

        "AI for Cybersecurity": [
            "intrusion detection",
            "anomaly detection",
            "vulnerabilities",
            "obfuscation",
            "cyber defense",
            "threat intelligence",
            "logs",
            "security monitoring",
        ],

        "AI Security": [
            "adversarial attacks",
            "data poisoning",
            "model inversion",
            "model privacy",
            "model confidentiality",
            "hardware attacks",
            "side-channel attacks",
            "federated learning",
        ],

        "Intelligence and Information Security": [
            "information extraction",
            "disinformation",
            "deepfakes",
            "information manipulation",
            "OSINT",
            "open-source intelligence",
            "fake news",
            "digital influence",
        ],

        "Intelligent and Secure Networks": [
            "computer networks",
            "5G",
            "software-defined networking",
            "SDN",
            "network functions virtualization",
            "NFV",
            "graph neural networks for networks",
            "network orchestration",
            "QoS",
            "quality of service",
        ],

        "Autonomous Robotics and Interaction": [
            "robotics",
            "shared control",
            "robot perception",
            "planning",
            "robot swarms",
            "human-robot interaction",
            "drones",
            "UAVs",
            "underwater robots",
            "autonomous robots",
        ],
    },


    "AI, Environment and Ocean": {

        "Physics-Informed AI": [
            "physics-informed AI",
            "physics-informed neural networks",
            "PINN",
            "data assimilation",
            "conformal prediction",
            "uncertainty quantification",
            "surrogate models",
        ],

        "Observation, Remote Sensing and Data Integration": [
            "remote sensing",
            "SAR",
            "synthetic aperture radar",
            "hyperspectral imaging",
            "sonar",
            "sensors",
            "sensor fusion",
            "data fusion",
            "AIS",
            "weak signals",
            "Earth observation",
        ],

        "Ocean Digital Twin": [
            "digital twin",
            "ocean digital twin",
            "ocean dynamics",
            "oceanography",
            "ocean modeling",
            "ocean-atmosphere modeling",
            "marine extremes",
            "underwater mapping",
            "subsea mapping",
        ],
    },
}










DEFAULT_FIELDS = [
    "halId_s", "title_s", "authFullName_s", "docType_s", "producedDate_s",
    "publicationDate_s", "journalTitle_s", "abstract_s", "keyword_s", "doiId_s",
    "uri_s", "structName_s", "labStructName_s", "domainAllCode_s",
]
logger = logging.getLogger(__name__)







def fetch_page(start: int, rows: int, fields: list[str] | None = None, query: str = "*:*", timeout: int = 30) -> dict[str, Any]:
    response = requests.get(HAL_API_URL, params={"q": query, "wt": "json", "start": start, "rows": rows, "fl": ",".join(fields or DEFAULT_FIELDS)}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_all(fields: list[str] | None = None, query: str = "*:*", max_docs: int | None = None) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    start = 0
    while True:
        rows = min(MAX_ROWS_PER_PAGE, max_docs - len(docs)) if max_docs is not None else MAX_ROWS_PER_PAGE
        if rows <= 0:
            break
        response = fetch_page(start, rows, fields, query).get("response", {})
        page = response.get("docs", [])
        if not page:
            break
        docs.extend(page)
        start += len(page)
        logger.info("Fetched %d/%d notices", len(docs), response.get("numFound", 0))
        if start >= response.get("numFound", 0) or (max_docs is not None and len(docs) >= max_docs):
            break
    return docs


def compute_axis_similarities(texts: list[str], axes: dict[str, str] | None = None) -> tuple[Any, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    axes = axes or SEQUOIA_AXES
    names = list(axes)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(list(texts) + list(axes.values()))
    return cosine_similarity(matrix[:len(texts)], matrix[len(texts):]), names


def labels_from_similarities(similarities: Any, axis_names: list[str], threshold: float = 0.05, no_class_label: str = "no class") -> list[str]:
    labels = []
    for row in similarities:
        index = row.argmax()
        labels.append(axis_names[index] if row[index] >= threshold else no_class_label)
    return labels
