#!/usr/bin/env bash
# =============================================================================
# deploy/setup_vps.sh
# Configura un servidor Ubuntu 22.04 limpio para el stack MLB AI.
#
# Ejecutar UNA SOLA VEZ como root o con sudo:
#   curl -fsSL https://raw.githubusercontent.com/fernandogg2004/mlb-lineup-optimizer/main/deploy/setup_vps.sh | bash
#   -- o --
#   bash deploy/setup_vps.sh
#
# Qué hace:
#   1. Actualiza el sistema
#   2. Instala Docker + Docker Compose
#   3. Instala Git + Git LFS
#   4. Configura firewall UFW
#   5. Clona el repositorio en /opt/mlb-ai
#   6. Descarga los modelos via Git LFS
#   7. Crea servicio systemd para auto-arranque en reboot
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/fernandogg2004/mlb-lineup-optimizer.git"
APP_DIR="/opt/mlb-ai"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN: $*${NC}"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $*${NC}"; exit 1; }

[[ $EUID -eq 0 ]] || die "Ejecuta este script como root: sudo bash setup_vps.sh"

# ---------------------------------------------------------------------------
log "[1/7] Actualizando sistema..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq ca-certificates curl gnupg lsb-release ufw

# ---------------------------------------------------------------------------
log "[2/7] Instalando Docker..."
if ! command -v docker &>/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
  log "Docker instalado: $(docker --version)"
else
  warn "Docker ya instalado, saltando."
fi

# ---------------------------------------------------------------------------
log "[3/7] Instalando Git y Git LFS..."
apt-get install -y -qq git
if ! command -v git-lfs &>/dev/null; then
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
  apt-get install -y -qq git-lfs
fi
git lfs install --system
log "Git LFS $(git lfs version) instalado."

# ---------------------------------------------------------------------------
log "[4/7] Configurando firewall..."
ufw --force reset > /dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp   comment 'SSH'
ufw allow 8000/tcp comment 'MLB API'
ufw allow 8080/tcp comment 'Airflow UI'
ufw --force enable
log "Firewall activo. Puertos abiertos: 22, 8000, 8080."

# ---------------------------------------------------------------------------
log "[5/7] Clonando repositorio en $APP_DIR..."
if [ -d "$APP_DIR/.git" ]; then
  warn "Repositorio ya existe. Ejecutando git pull..."
  cd "$APP_DIR" && git pull
else
  mkdir -p "$APP_DIR"
  # Si el repo es privado, Git pedirá usuario/PAT aquí.
  # Para evitar el prompt: git clone https://<TOKEN>@github.com/fernandogg2004/mlb-lineup-optimizer.git
  git clone "$REPO_URL" "$APP_DIR"
fi

# ---------------------------------------------------------------------------
log "[6/7] Descargando modelos via Git LFS..."
cd "$APP_DIR"
git lfs pull
log "Modelos descargados: $(ls models/*.pkl 2>/dev/null | wc -l) archivos .pkl"

# ---------------------------------------------------------------------------
log "[7/7] Creando servicio systemd mlb-ai..."
cat > /etc/systemd/system/mlb-ai.service << UNIT
[Unit]
Description=MLB AI Stack (API + Airflow)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=/usr/bin/docker compose -f docker-compose.vps.yml --env-file .env up -d
ExecStop=/usr/bin/docker compose  -f docker-compose.vps.yml down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable mlb-ai.service
log "Servicio mlb-ai.service registrado y habilitado en boot."

# ---------------------------------------------------------------------------
PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "<IP_DEL_SERVIDOR>")

echo ""
echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}  Setup completado.  Próximos pasos:${NC}"
echo ""
echo -e "  1. Crea el archivo de entorno:"
echo -e "     ${YELLOW}cp $APP_DIR/.env.vps.example $APP_DIR/.env${NC}"
echo -e "     ${YELLOW}nano $APP_DIR/.env${NC}   ← edita passwords y claves"
echo ""
echo -e "  2. Primer arranque (construye imágenes + configura Airflow):"
echo -e "     ${YELLOW}cd $APP_DIR && bash deploy/first_run.sh${NC}"
echo ""
echo -e "  Cuando esté listo:"
echo -e "     API:     http://${PUBLIC_IP}:8000/docs"
echo -e "     Airflow: http://${PUBLIC_IP}:8080"
echo -e "${GREEN}=================================================================${NC}"
