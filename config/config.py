from pathlib import Path

WORKSPACE = Path('./')
DATA_PATH = WORKSPACE / 'data' / 'manual_labels_with_hal_metadata.csv'
ARTIFACT_DIR = WORKSPACE / 'mpnet_sgd_artifacts'
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
CLASSIFIER_PATH = ARTIFACT_DIR / 'sgd_classifier.joblib'
ENCODER_PATH = ARTIFACT_DIR / 'mpnet_encoder'
SUBAXIS_PATH = ARTIFACT_DIR / 'subaxis_references.joblib'
METADATA_PATH = ARTIFACT_DIR / 'metadata.json'
EMBEDDING_CACHE_PATH = ARTIFACT_DIR / 'training_embeddings.npz'
BEST_METRIC_PATH = ARTIFACT_DIR / 'best_metric.json'

REGISTERED_MODEL_NAME = "sequoia-mpnet-sgd-classifier"
RANDOM_STATE = 42


PILLAR_MAP = {
    "core ai": "Core AI",
    "ai, cybersecurity and defense": "AI, Cybersecurity and Defense",
    "ai, cybersecurity & defense": "AI, Cybersecurity and Defense",
    "ia & cybersécurité": "AI, Cybersecurity and Defense",
    "ai, environment and ocean": "AI, Environment and Ocean",
    "ai, environment & ocean": "AI, Environment and Ocean",
    "ia & environnement": "AI, Environment and Ocean",
    "no class": "No class",
    "non classifié": "No class",
}



CANONICAL_LABELS = [
    'AI, cybersecurity and defense',
    'AI, environment and ocean',
    'Core AI',
    'No class',
]


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
