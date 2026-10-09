#!/usr/bin/env bash
# Déploie le dashboard SequoIA dans son propre conteneur sur la VM.
#
#   SEQUOIA_VM=ubuntu@<ip> SEQUOIA_VM_KEY=~/clé.pem deploy/deploy.sh
#
# Le code de l'arbre de travail est envoyé sur la VM, l'image y est construite,
# puis le conteneur SEQUOIA_CONTAINER est recréé. Les données ne quittent jamais
# la VM : ~/sequoia-dashboard/{data,models} sont montés dans le conteneur et
# ~/sequoia-dashboard/sequoia.env fournit les identifiants. Aucun autre
# conteneur n'est modifié.
set -euo pipefail

: "${SEQUOIA_VM:?définir SEQUOIA_VM, par exemple ubuntu@<ip>}"
KEY=${SEQUOIA_VM_KEY:+-i $SEQUOIA_VM_KEY}
REMOTE_DIR=${SEQUOIA_REMOTE_DIR:-sequoia-dashboard}
NAME=${SEQUOIA_CONTAINER:-sequoia-dashboard}
PUBLISH=${SEQUOIA_PUBLISH:-127.0.0.1:8090}
# Image contenant déjà Quarto et les dépendances Python (la VM a peu de disque).
BASE_IMAGE=${SEQUOIA_BASE_IMAGE:-sequoia-dbfirst-test:20261004}
MEMORY=${SEQUOIA_MEMORY:-1400m}
MEMORY_SWAP=${SEQUOIA_MEMORY_SWAP:-2800m}

cd "$(git rev-parse --show-toplevel)"
TAG=$(date -u +%Y%m%dT%H%M%S)-$(git rev-parse --short HEAD)
git diff --quiet HEAD || TAG+=-dirty

echo "→ Envoi du code ($TAG)"
git ls-files -co --exclude-standard -z | tar -czf - --null -T - --ignore-failed-read 2>/dev/null \
  | ssh $KEY "$SEQUOIA_VM" "set -e; cd ~/$REMOTE_DIR; rm -rf code.new; mkdir code.new; tar -xzf - -C code.new; rm -rf code; mv code.new code"

ssh $KEY "$SEQUOIA_VM" NAME="$NAME" TAG="$TAG" PUBLISH="$PUBLISH" BASE_IMAGE="$BASE_IMAGE" \
    MEMORY="$MEMORY" MEMORY_SWAP="$MEMORY_SWAP" REMOTE_DIR="$REMOTE_DIR" bash -s <<'REMOTE'
set -euo pipefail
DIR=$HOME/$REMOTE_DIR
[ -f "$DIR/data/sequoia_v2.db" ] || { echo "Base absente : $DIR/data/sequoia_v2.db" >&2; exit 1; }
[ -f "$DIR/sequoia.env" ] || { echo "Identifiants absents : $DIR/sequoia.env" >&2; exit 1; }

echo "→ Construction de l'image $NAME:$TAG"
sudo docker build -q --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$NAME:$TAG" "$DIR/code" >/dev/null

echo "→ Remplacement du conteneur $NAME"
PREVIOUS=$(sudo docker inspect "$NAME" --format '{{.Config.Image}}' 2>/dev/null || true)
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo docker run -d --name "$NAME" --restart unless-stopped \
  -p "$PUBLISH:10000" --memory "$MEMORY" --memory-swap "$MEMORY_SWAP" --cpus 1 \
  --env-file "$DIR/sequoia.env" \
  -e SQLITE_DB_PATH=/sequoia/dashboard/data/sequoia_v2.db \
  -e SEQUOIA_ARTIFACT_DIR=/sequoia/mpnet_sgd_artifacts \
  -e SEQUOIA_ENABLE_PIPELINE_JOBS=1 -e OMP_NUM_THREADS=1 \
  -v "$DIR/data:/sequoia/dashboard/data" \
  -v "$DIR/models:/sequoia/mpnet_sgd_artifacts" \
  "$NAME:$TAG" >/dev/null

for attempt in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://${PUBLISH/0.0.0.0/127.0.0.1}/index.html"; then
    echo "✓ $NAME répond sur $PUBLISH (image $NAME:$TAG)"
    # Ne garder que l'image courante et la précédente (retour arrière possible).
    sudo docker images "$NAME" --format '{{.Repository}}:{{.Tag}}' \
      | grep -vxF -e "$NAME:$TAG" -e "${PREVIOUS:-none}" | xargs -r sudo docker rmi >/dev/null 2>&1 || true
    exit 0
  fi
  sleep 2
done
echo "✗ $NAME ne répond pas ; derniers journaux :" >&2
sudo docker logs --tail 40 "$NAME" >&2
exit 1
REMOTE
