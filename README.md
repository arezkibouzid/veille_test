# SequoIA — veille des publications

Tableau de bord et pipeline de classification des publications de la collection
HAL [SEQUOIA](https://hal.science/SEQUOIA). Chaque publication est rattachée à
un pilier (Core AI ; AI, Cybersecurity and Defense ; AI, Environment and Ocean ;
No class) puis à un axe, proposée à la validation humaine, et les validations
servent à réentraîner le modèle.

## Fonctionnement

```
API HAL ──► SQLite (articles « new ») ──► OpenAlex ──► prédiction ──► validation humaine
                                                                            │
                          modèle actif ◄── dvc repro (réentraînement) ◄─────┘
```

- **Ingestion** : chaque jour, GitHub Actions demande à la VM de récupérer les
  nouvelles notices HAL, de les enrichir (OpenAlex) et de les classer. Rien ne
  transite par GitHub : les données restent sur la VM.
- **Validation** : la page `validation` du dashboard permet de confirmer ou
  corriger le pilier et l'axe proposés.
- **Réentraînement** : le bouton « Réentraîner le modèle » lance `dvc repro` sur
  la VM. Le nouveau modèle n'est activé que s'il fait mieux que le modèle actif
  sur les mêmes articles de test, hors articles déjà vus par ce dernier.

## Organisation du dépôt

| Chemin | Rôle |
|---|---|
| `config/config.py` | Chemins, taxonomie des piliers et des axes |
| `src/` | Préparation des données, encodeur, entraînement, évaluation, prédiction |
| `main.py` | Entraîne et évalue un modèle candidat |
| `dvc.yaml` | Étapes du réentraînement (snapshot → préparation → encodeur → entraînement) |
| `dashboard/*.qmd` | Pages du site Quarto |
| `dashboard/scripts/validation_api.py` | API FastAPI ; sert aussi le site rendu |
| `dashboard/scripts/refresh_hal.py` | Extraction HAL vers SQLite |
| `dashboard/scripts/pipeline_jobs.py` | Traitements en tâche de fond : `predict`, `retrain`, `citations`, `report` |
| `dashboard/scripts/trigger_pipeline.py` | Déclenche un traitement à distance (utilisé par GitHub Actions) |
| `dashboard/scripts/import_manual_labels.py` | Importe les labels historiques du CSV |
| `data/manual_labels_with_hal_metadata.csv` | 1 404 labels humains historiques ; seul moyen de reconstruire une base depuis zéro |
| `Dockerfile`, `deploy/deploy.sh` | Image et déploiement du conteneur sur la VM |
| `.github/workflows/` | Ingestion quotidienne (`hal-sync.yml`) et tests (`tests.yml`) |

Ne sont **jamais** dans Git : la base SQLite (`dashboard/data/`), les modèles
(`mpnet_sgd_artifacts/`) et l'état DVC (`.dvc/`, `dvc.lock`).

## Installation locale

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-optional.txt
```

Variables d'environnement :

| Variable | Rôle | Défaut |
|---|---|---|
| `SQLITE_DB_PATH` | Base SQLite | `dashboard/data/validation/sequoia_v2.db` |
| `SEQUOIA_ARTIFACT_DIR` | Dossier des modèles | `mpnet_sgd_artifacts/` |
| `VALIDATION_USER`, `VALIDATION_PASSWORD` | Compte administrateur de l'API | aucun (API d'administration désactivée) |
| `SEQUOIA_ENABLE_PIPELINE_JOBS` | `1` pour autoriser les traitements | `0` |
| `OPENALEX_API_KEY` | Clé OpenAlex, facultative | vide |

## Reconstruire une base depuis zéro

```bash
python -m dashboard.scripts.refresh_hal            # notices HAL → articles « new »
python -m dashboard.scripts.import_manual_labels   # labels historiques du CSV → validations
python main.py                                     # premier modèle (plusieurs minutes sur CPU)
python -m dashboard.scripts.predict_new_articles   # classe les articles restants
```

Le CSV ne fournit que l'identifiant HAL et le label ; toutes les métadonnées
viennent de HAL. Deux identifiants du CSV ne sont plus dans la collection et
sont ignorés.

Lancer le dashboard :

```bash
quarto render dashboard
uvicorn dashboard.scripts.validation_api:app --port 10000
```

## Tests

```bash
pip install -r requirements-ingest.txt pytest
python -m pytest tests
```

Les tests n'utilisent ni le réseau ni le modèle.

## Production (VM)

Le site public est servi par le conteneur `sequoia-dashboard`, derrière un
proxy nginx. Tout ce qui est volumineux vit dans `~/sequoia-dashboard/` sur la VM :

| Dossier de la VM | Monté dans le conteneur sur | Contenu |
|---|---|---|
| `data/` | `dashboard/data/` | Base SQLite, snapshots, rapports |
| `models/` | `mpnet_sgd_artifacts/` | Encodeur, versions de modèles, cache d'embeddings |
| `dvc/` | `.dvc/` | Configuration DVC, `dvc.lock`, cache DVC |
| `sequoia.env` | (variables) | Identifiants administrateur |

### Déployer

```bash
SEQUOIA_VM=ubuntu@<ip> SEQUOIA_VM_KEY=~/clé.pem deploy/deploy.sh
```

Le script envoie l'arbre de travail, construit l'image sur la VM (une dizaine
de minutes) et recrée uniquement le conteneur `sequoia-dashboard`. L'image
précédente est conservée pour un retour arrière.

### DVC

DVC fonctionne sur la VM sans Git (`no_scm`) et sans stockage distant : le
cache est sur le disque de la VM.

- `dvc.lock` enregistre, pour chaque réentraînement, l'empreinte du snapshot de
  la base, du manifeste d'entraînement et des modèles produits.
- Les snapshots de la base sont conservés dans le cache DVC ; l'encodeur et les
  versions de modèles sont suivis par empreinte seulement, pour ne pas doubler
  plusieurs Go.

Commandes utiles, sur la VM :

```bash
sudo docker exec sequoia-dashboard dvc status         # ce qui a changé depuis le dernier entraînement
sudo docker exec sequoia-dashboard dvc metrics show   # métriques du dernier entraînement
sudo docker exec sequoia-dashboard dvc repro          # réentraîner à la main
sudo docker exec sequoia-dashboard dvc gc --workspace # purger les anciens snapshots du cache
```

Ne pas lancer `dvc repro` à la main pendant qu'un traitement du dashboard est
en cours.

### Ingestion quotidienne

`.github/workflows/hal-sync.yml` s'exécute chaque jour à 04:17 UTC, ou à la
demande depuis l'onglet Actions. Secrets du dépôt requis :

| Secret | Valeur |
|---|---|
| `SEQUOIA_API_URL` | Adresse publique du dashboard |
| `SEQUOIA_API_USER`, `SEQUOIA_API_PASSWORD` | Compte administrateur |

Le résumé du run indique le nombre de nouvelles publications et de
classifications. Le site n'étant pas en HTTPS, le mot de passe circule en clair.
