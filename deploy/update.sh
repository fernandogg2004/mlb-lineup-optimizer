#!/usr/bin/env bash
# =============================================================================
# deploy/update.sh
# Actualiza el stack MLB AI en el VPS con el último código del repositorio.
#
# Uso (desde /opt/mlb-ai):
#   bash deploy/update.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}"; }
step() { echo -e "\n${YELLOW}=== $* ===${NC}"; }

COMPOSE="docker compose -f docker-compose.vps.yml --env-file .env"

step "Descargando cambios del repositorio"
git pull
git lfs pull   # descarga nuevos modelos si los hay

step "Reconstruyendo imágenes y reiniciando contenedores"
$COMPOSE up -d --build --remove-orphans

step "Estado del stack"
$COMPOSE ps

log "Actualización completada."
