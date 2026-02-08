#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONTAINER_NAME="searxng"
IMAGE="docker.io/searxng/searxng:latest"
PORT="${SEARXNG_PORT:-8080}"
CONFIG_DIR="$PROJECT_ROOT/scripts/searxng-config"

usage() {
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 1
}

ensure_config() {
    mkdir -p "$CONFIG_DIR"

    if [[ ! -f "$CONFIG_DIR/settings.yml" ]]; then
        cat > "$CONFIG_DIR/settings.yml" << 'EOF'
use_default_settings: true

general:
  instance_name: "Search Agent SearXNG"
  privacypolicy_url: false
  donation_url: false
  contact_url: false
  enable_metrics: false

search:
  safe_search: 0
  autocomplete: ""
  default_lang: "auto"
  formats:
    - html
    - json

server:
  secret_key: "change-me-in-production"
  limiter: false
  image_proxy: true
  http_protocol_version: "1.1"

ui:
  static_use_hash: true
  default_theme: simple
  results_on_new_tab: false

outgoing:
  request_timeout: 5.0
  max_request_timeout: 15.0
  pool_connections: 100
  pool_maxsize: 20

engines:
  - name: google
    engine: google
    shortcut: g
    disabled: false

  - name: bing
    engine: bing
    shortcut: b
    disabled: false

  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false

  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    disabled: false
EOF
        echo "Created default settings.yml"
    fi
}

start() {
    ensure_config

    if podman ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '$CONTAINER_NAME' is already running"
        exit 0
    fi

    # Remove stopped container if exists
    podman rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo "Starting SearXNG on port $PORT..."
    podman run -d \
        --name "$CONTAINER_NAME" \
        -p "$PORT:8080" \
        -v "$CONFIG_DIR/settings.yml:/etc/searxng/settings.yml:ro,Z" \
        "$IMAGE"

    echo "SearXNG started: http://localhost:$PORT"
}

stop() {
    echo "Stopping SearXNG..."
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true
    echo "SearXNG stopped"
}

restart() {
    stop
    start
}

status() {
    if podman ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "SearXNG is running"
        podman ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        echo "SearXNG is not running"
    fi
}

logs() {
    podman logs -f "$CONTAINER_NAME"
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    logs)    logs ;;
    *)       usage ;;
esac
