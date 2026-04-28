#!/usr/bin/env bash
#
# Mesh Routing Bench - Universal Setup Script
#
# Этот скрипт автоматизирует развёртку стенда для разработки ML-сервера маршрутизации
# поверх Meshtastic-mesh в двух режимах:
#   1. Single-host (всё на одной машине) — быстрая отладка ML
#   2. Distributed (RPi-узлы + ноут) — реалистичный mesh
#
# Использование:
#   ./setup.sh [single-host|distributed] [setup|start|stop|reset|status|test|clean]
#
# Примеры:
#   ./setup.sh single-host setup   # полный стенд на одной машине
#   ./setup.sh distributed setup   # брокер+UI на ноуте, узлы на RPi
#   ./setup.sh single-host start   # просто запустить контейнеры
#   ./setup.sh single-host reset   # полный сброс с сохранением конфигов
#   ./setup.sh single-host clean   # полный сброс со стиранием volume
#

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Настройки ---

COMPOSE_DIR="$SCRIPT_DIR/compose"
WEB_DIR="$SCRIPT_DIR/web"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# Порты по умолчанию
BROKER_HOST="localhost"
BROKER_PORT=1883
UI_PORT=5173

# --- Функции проверки окружения ---

check_docker() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker не установлен"
        log_info "Установите Docker:"
        log_info "  Linux/RPi: curl -fsSL https://get.docker.com | sh"
        log_info "  Windows/macOS: Docker Desktop"
        return 1
    fi
    if ! docker info &>/dev/null; then
        log_error "Docker daemon не запущен или прав недостаточно"
        return 1
    fi
    log_ok "Docker доступен"
    return 0
}

check_python() {
    if ! command -v python3 &>/dev/null; then
        log_error "Python 3 не найден"
        return 1
    fi
    local py_ver
    py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ $(echo "$py_ver < 3.10" | bc -l 2>/dev/null || echo 1) -eq 1 ]]; then
        log_warn "Python 3.10+ рекомендуется, найден $py_ver"
    fi
    log_ok "Python $py_ver доступен"
    return 0
}

check_make() {
    # Проверяем, что docker compose работает
    if docker compose version &>/dev/null; then
        DOCKER_COMPOSE="docker compose"
        log_ok "docker compose доступен"
    elif docker-compose version &>/dev/null; then
        DOCKER_COMPOSE="docker-compose"
        log_ok "docker-compose доступен"
    else
        log_error "docker-compose не установлен"
        return 1
    fi
    return 0
}

# --- Утилиты ---

get_lan_ip() {
    # Получить IP адрес в локальной сети
    local ip
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ip=$(ip -4 addr show scope global | grep inet | head -1 | awk '{print $2}' | cut -d/ -f1)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        ip=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        ip=$(ipconfig | grep -A1 "IPv4" | tail -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+')
    fi
    echo "$ip"
}

ensure_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Создаём виртуальное окружение для meshtastic CLI..."
        python3 -m venv "$VENV_DIR"
    fi
    # Активируем в текущей сессии
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
    elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
        source "$VENV_DIR/Scripts/activate"
    fi
    
    if [[ -f "$REQUIREMENTS" ]]; then
        pip install -q -r "$REQUIREMENTS" 2>/dev/null || true
    fi
    log_ok "Виртуальное окружение готово"
}

ensure_web_deps() {
    if [[ -d "$WEB_DIR" ]] && [[ -f "$WEB_DIR/package.json" ]]; then
        if [[ ! -d "$WEB_DIR/node_modules" ]]; then
            log_info "Устанавливаем зависимости веб-UI..."
            (cd "$WEB_DIR" && npm install)
        fi
        log_ok "Зависимости UI готовы"
    else
        log_warn "Директория web/ не найдена, UI будет пропущено"
    fi
}

check_port() {
    local port=$1
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        netstat -ano | findstr :$port &>/dev/null
    else
        nc -z localhost "$port" 2>/dev/null
    fi
}

# --- Режимы развёртки ---

setup_single_host() {
    log_info "=== РЕЖИМ: Single-host (всё на одной машине) ==="
    
    # 1. Проверки
    check_docker || return 1
    check_python || return 1
    check_make || return 1
    
    # 2. Запуск контейнеров
    log_info "Шаг 1: Запуск контейнеров (Docker)..."
    cd "$COMPOSE_DIR"
    $DOCKER_COMPOSE --profile single-host up -d --build
    
    # Ожидание готовности узлов
    log_info "Ожидание запуска узлов (до 60 сек)..."
    for i in $(seq 1 30); do
        if $DOCKER_COMPOSE ps nodeA | grep -q "Up"; then
            log_ok "Узлы запущены"
            break
        fi
        sleep 2
    done
    
    # 3. Настройка MQTT в узлах
    log_info "Шаг 2: Настройка узлов (MQTT-конфигурация)..."
    ensure_venv
    
    for HOST in localhost:4403 localhost:4404; do
        log_info "  Настройка узла $HOST..."
        # Попытки с таймаутом
        for attempt in $(seq 1 10); do
            if meshtastic --host "$HOST" --set mqtt.address=mosquitto \
                --set mqtt.username="" \
                --set mqtt.password="" \
                --set mqtt.enabled=true \
                --set mqtt.encryption_enabled=false \
                --set mqtt.json_enabled=true \
                --set mqtt.tls_enabled=false \
                --ch-index 0 --ch-set uplink_enabled=true --ch-set downlink_enabled=true 2>/dev/null; then
                meshtastic --host "$HOST" --ch-add mqtt 2>/dev/null || true
                meshtastic --host "$HOST" --ch-index 1 --ch-set uplink_enabled=true --ch-set downlink_enabled=true 2>/dev/null || true
                log_ok "  Узел $HOST настроен"
                break
            fi
            if [[ $attempt -eq 10 ]]; then
                log_warn "  Не удалось настроить $HOST после 10 попыток"
            else
                sleep 2
            fi
        done
    done
    
    # 4. Рестарт узлов для применения конфига
    log_info "Шаг 3: Перезапуск узлов для применения конфигурации..."
    $DOCKER_COMPOSE restart nodeA nodeB
    sleep 3
    
    # 5. UI
    log_info "Шаг 4: Проверка веб-UI..."
    if ensure_web_deps; then
        if [[ -f "$WEB_DIR/package.json" ]]; then
            log_info "UI доступна по адресу: http://localhost:$UI_PORT"
            log_info "Чтобы запустить UI, выполните: cd web && npm run dev"
        fi
    fi
    
    # 6. Информация о подключении
    log_ok "Стенд готов!"
    log_info "Брокер MQTT:  $BROKER_HOST:$BROKER_PORT"
    log_info "NodeA (CLI):  localhost:4403"
    log_info "NodeB (CLI):  localhost:4404"
    if [[ -n "$(get_lan_ip)" ]]; then
        log_info "Из сети:       http://$(get_lan_ip):$UI_PORT"
    fi
    
    if check_port "$UI_PORT"; then
        log_info "UI уже запущена на порту $UI_PORT"
    fi
    
    log_info "Смотри README.md для подробностей тестирования"
    cd "$SCRIPT_DIR"
}

setup_distributed() {
    log_info "=== РЕЖИМ: Distributed (RPi-узлы + ноут) ==="
    
    # На ноуте: брокер + UI
    log_info "Настройка брокера и UI на ноуте..."
    check_docker || return 1
    check_make || return 1
    
    cd "$COMPOSE_DIR"
    $DOCKER_COMPOSE --profile host up -d
    ensure_web_deps
    
    # Получаем IP ноута
    LAPTOP_IP=$(get_lan_ip)
    log_ok "IP ноута в LAN: $LAPTOP_IP"
    
    log_info "=== ТЕПЕРЬ НА КАЖДОМ RPi ==="
    log_info "Выполните следующие команды на каждом RPi:"
    echo ""
    echo "# 1. Установите docker (если не установлен)"
    echo "curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker \$USER"
    echo ""
    echo "# 2. Склонируйте репозиторий (если ещё не сделано)"
    echo "git clone <repo-url>"
    echo "cd mesh-routing-bench/compose"
    echo ""
    echo "# 3. Запустите узел с уникальным NODE_HWID"
    echo "NODE_HWID=1001 docker compose --profile node up -d --build"
    echo ""
    echo "# 4. После сборки настройте узел"
    echo "source ../.venv/bin/activate"
    echo "meshtastic --host localhost:4403 --set mqtt.address=$LAPTOP_IP --set mqtt.enabled=true --set mqtt.encryption_enabled=false --set mqtt.json_enabled=true --set mqtt.tls_enabled=false --ch-index 0 --ch-set uplink_enabled=true --ch-set downlink_enabled=true"
    echo "meshtastic --host localhost:4403 --ch-add mqtt"
    echo "meshtastic --host localhost:4403 --ch-index 1 --ch-set uplink_enabled=true --ch-set downlink_enabled=true"
    echo "NODE_HWID=1001 docker compose --profile node restart node"
    echo ""
    echo "# 5. Запустите UI на ноуте"
    echo "cd ../web && npm run dev"
    echo ""
    log_info "UI будет доступна по адресу: http://$LAPTOP_IP:$UI_PORT"
    
    cd "$SCRIPT_DIR"
}

# --- Управление контейнерами ---

start_containers() {
    log_info "Запуск контейнеров..."
    cd "$COMPOSE_DIR"
    $DOCKER_COMPOSE up -d
    log_ok "Контейнеры запущены"
    cd "$SCRIPT_DIR"
}

stop_containers() {
    log_info "Остановка контейнеров..."
    cd "$COMPOSE_DIR"
    $DOCKER_COMPOSE stop
    log_ok "Контейнеры остановлены"
    cd "$SCRIPT_DIR"
}

reset_containers() {
    log_info "Сброс контейнеров (сохранение volume)..."
    cd "$COMPOSE_DIR"
    $DOCKER_COMPOSE down
    log_ok "Контейнеры сброшены (конфигурация сохранена)"
    cd "$SCRIPT_DIR"
}

clean_all() {
    log_warn "Полный сброс (все данные будут удалены)..."
    cd "$COMPOSE_DIR"
    $DOCKER_COMPOSE down -v
    log_ok "Полный сброс выполнен"
    cd "$SCRIPT_DIR"
}

show_status() {
    cd "$COMPOSE_DIR"
    log_info "Статус контейнеров:"
    $DOCKER_COMPOSE ps
    echo ""
    
    if check_port 1883; then
        log_ok "MQTT брокер: работает на порту 1883"
    else
        log_error "MQTT брокер: не работает"
    fi
    
    if check_port 4403; then
        log_ok "NodeA: работает на порту 4403"
    else
        log_warn "NodeA: не работает"
    fi
    
    if check_port 4404; then
        log_ok "NodeB: работает на порту 4404"
    else
        log_warn "NodeB: не работает"
    fi
    
    if check_port "$UI_PORT"; then
        log_ok "Web UI: работает на порту $UI_PORT"
    else
        log_warn "Web UI: не работает"
    fi
    
    echo ""
    log_info "Логи (последние 10 строк):"
    $DOCKER_COMPOSE logs --tail=10 2>/dev/null || true
    
    cd "$SCRIPT_DIR"
}

run_test() {
    log_info "=== Запуск теста ==="
    
    ensure_venv
    
    # Публикуем тестовое сообщение через nodeA
    log_info "Отправка тестового сообщения через NodeA..."
    meshtastic --host localhost:4403 --sendtext "ping" --ch-index 1 || log_warn "Не удалось отправить сообщение"
    
    # Проверяем поток сообщений
    log_info "Подписка на топики MQTT (5 секунд)..."
    if command -v mosquitto_sub &>/dev/null; then
        timeout 5 mosquitto_sub -h localhost -t '#' -v 2>/dev/null || true
    else
        log_warn "mosquitto_sub не установлен, пропускаем"
    fi
    
    log_ok "Тест завершён"
}

show_help() {
    cat << EOF
Mesh Routing Bench - Универсальный скрипт настройки

Использование:
    $0 <режим> <команда>

Режимы:
    single-host    Стенд на одной машине (быстрый старт)
    distributed    RPi-узлы + ноут (реалистичный стенд)

Команды:
    setup          Полная настройка стенда (по умолчанию)
    start          Запустить контейнеры
    stop           Остановить контейнеры
    reset          Сбросить контейнеры (сохранить данные)
    clean          Полный сброс (удалить всё)
    status         Показать статус
    test           Запустить тест
    help           Показать эту справку

Примеры:
    # Быстрый старт на одной машине
    $0 single-host setup
    $0 single-host start
    $0 single-host status
    
    # Тестирование
    $0 single-host test
    
    # Остановка и сброс
    $0 single-host stop
    $0 single-host reset
    
    # Полный сброс
    $0 single-host clean
    
    # Распределённый стенд (RPi + ноут)
    $0 distributed setup

URL-адреса:
    UI:        http://localhost:$UI_PORT
    MQTT:      tcp://localhost:$BROKER_PORT
    NodeA CLI: localhost:4403
    NodeB CLI: localhost:4404

Документация:
    Смотри README.md и CONTRACT.md
EOF
}

# --- Главная функция ---

main() {
    local mode="${1:-single-host}"
    local command="${2:-setup}"
    
    case "$command" in
        setup)
            case "$mode" in
                single-host) setup_single_host ;;
                distributed) setup_distributed ;;
                *) log_error "Неизвестный режим: $mode"; show_help; exit 1 ;;
            esac
            ;;
        start)   start_containers ;;
        stop)    stop_containers ;;
        reset)   reset_containers ;;
        clean)   clean_all ;;
        status)  show_status ;;
        test)    run_test ;;
        help)    show_help ;;
        *)
            log_error "Неизвестная команда: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
