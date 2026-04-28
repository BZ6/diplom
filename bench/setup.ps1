#Requires -Version 5.1
<#
.SYNOPSIS
    Mesh Routing Bench - Setup Script for Windows

.DESCRIPTION
    Развёртка стенда: 2 виртуальных Meshtastic-узла + MQTT-брокер + ML-сервер.
    Режимы: single-host (всё на одной машине) | distributed (RPi-узлы + ноут).

.PARAMETER Mode
    single-host | distributed

.PARAMETER Command
    setup | start | stop | reset | clean | status | test | help

.EXAMPLE
    .\setup.ps1 single-host setup
    .\setup.ps1 single-host status
    .\setup.ps1 single-host test
    .\setup.ps1 single-host clean
#>
param(
    [string]$Mode    = "single-host",
    [string]$Command = "setup"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ComposeDir   = Join-Path $ScriptDir "compose"
$WebDir       = Join-Path $ScriptDir "web"
$VenvDir      = Join-Path $ScriptDir ".venv"
$Requirements = Join-Path $ScriptDir "requirements.txt"
$BrokerPort   = 1883
$UiPort       = 5173
$script:UseNewCompose = $true   # docker compose vs docker-compose

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
function Log-Info  ([string]$Msg) { Write-Host "[INFO]  $Msg" -ForegroundColor Cyan    }
function Log-Ok    ([string]$Msg) { Write-Host "[OK]    $Msg" -ForegroundColor Green   }
function Log-Warn  ([string]$Msg) { Write-Host "[WARN]  $Msg" -ForegroundColor Yellow  }
function Log-Error ([string]$Msg) { Write-Host "[ERROR] $Msg" -ForegroundColor Red     }

# ---------------------------------------------------------------------------
# docker compose wrapper  (fixes the $Args / splatting bug)
# ---------------------------------------------------------------------------
function Invoke-Compose {
    param([string[]]$ComposeArgs)
    if ($script:UseNewCompose) {
        docker compose @ComposeArgs
    } else {
        docker-compose @ComposeArgs
    }
}

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
function Assert-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Log-Error "Docker не установлен. Скачайте Docker Desktop."
        return $false
    }
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Log-Error "Docker daemon не запущен (или нет прав). Запустите Docker Desktop."
        return $false
    }
    Log-Ok "Docker доступен"
    return $true
}

function Assert-Python {
    $py = (Get-Command python  -ErrorAction SilentlyContinue) ??
          (Get-Command python3 -ErrorAction SilentlyContinue)
    if (-not $py) { Log-Error "Python 3 не найден"; return $false }
    $ver = & $py.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
    Log-Ok "Python $ver доступен"
    return $true
}

function Assert-Compose {
    $null = docker compose version 2>&1
    if ($LASTEXITCODE -eq 0) { $script:UseNewCompose = $true;  Log-Ok "docker compose (plugin) доступен"; return $true }
    $null = docker-compose version 2>&1
    if ($LASTEXITCODE -eq 0) { $script:UseNewCompose = $false; Log-Ok "docker-compose (standalone) доступен"; return $true }
    Log-Error "docker compose не найден"
    return $false
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
function Get-LanIP {
    return (Get-NetIPAddress -AddressFamily IPv4 |
            Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and
                           $_.PrefixOrigin -ne 'WellKnown' } |
            Select-Object -First 1).IPAddress
}

function Test-Port ([int]$Port) {
    try {
        $tcp  = New-Object System.Net.Sockets.TcpClient
        $conn = $tcp.BeginConnect("localhost", $Port, $null, $null)
        $ok   = $conn.AsyncWaitHandle.WaitOne(500, $false)
        if ($ok) { $tcp.EndConnect($conn) }
        $tcp.Close()
        return $ok
    } catch { return $false }
}

function Get-PythonExe {
    $cmd = (Get-Command python  -ErrorAction SilentlyContinue) ??
           (Get-Command python3 -ErrorAction SilentlyContinue)
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-MeshtasticExe {
    $venvExe = Join-Path $VenvDir "Scripts\meshtastic.exe"
    if (Test-Path $venvExe) { return $venvExe }
    return (Get-Command meshtastic -ErrorAction SilentlyContinue)?.Source
}

function Ensure-Venv {
    $pyExe = Get-PythonExe
    if (-not $pyExe) { Log-Error "Python не найден, venv создать невозможно"; return }

    if (-not (Test-Path $VenvDir)) {
        Log-Info "Создаём виртуальное окружение..."
        & $pyExe -m venv $VenvDir
    }

    $activate = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $activate) { & $activate }

    if (Test-Path $Requirements) {
        $pip = Join-Path $VenvDir "Scripts\pip.exe"
        Log-Info "Устанавливаем зависимости Python..."
        & $pip install -q -r $Requirements 2>&1 | Where-Object { $_ -match "ERROR" } | ForEach-Object { Log-Warn $_ }
    }
    Log-Ok "Виртуальное окружение готово"
}

function Ensure-WebDeps {
    if (-not (Test-Path (Join-Path $WebDir "package.json"))) {
        Log-Warn "Директория web/ не найдена — UI пропущено"
        return
    }
    if (-not (Test-Path (Join-Path $WebDir "node_modules"))) {
        Log-Info "Устанавливаем зависимости веб-UI..."
        Push-Location $WebDir; npm install; Pop-Location
    }
    Log-Ok "Зависимости UI готовы"
}

# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------
function Wait-ContainerUp ([string]$Service, [int]$TimeoutSec = 120) {
    Log-Info "Ожидание запуска $Service (до $TimeoutSec сек)..."
    Push-Location $ComposeDir
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $ps = docker compose ps $Service 2>&1 | Out-String
        if ($ps -match "running|Up") {
            Pop-Location
            Log-Ok "$Service запущен"
            return $true
        }
        Start-Sleep -Seconds 3
    }
    Pop-Location
    Log-Warn "$Service не готов за $TimeoutSec сек — продолжаем"
    return $false
}

function Wait-PortOpen ([int]$Port, [int]$TimeoutSec = 60, [string]$Label = "порт $Port") {
    Log-Info "Ожидание открытия $Label (до $TimeoutSec сек)..."
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $Port) { Log-Ok "$Label доступен"; return $true }
        Start-Sleep -Seconds 2
    }
    Log-Warn "$Label так и не открылся"
    return $false
}

# ---------------------------------------------------------------------------
# Meshtastic node configuration
# ---------------------------------------------------------------------------
function Set-NodeMqttConfig ([string]$HostPort) {
    # meshtastic CLI поддерживает формат --host host:port (проверено)
    $mesh = Get-MeshtasticExe
    if (-not $mesh) { Log-Warn "meshtastic CLI не найден, пропускаем настройку $HostPort"; return }

    $port = [int]($HostPort -split ':')[1]
    Log-Info "  Настройка узла $HostPort..."

    for ($i = 1; $i -le 12; $i++) {
        if (-not (Test-Port $port)) { Start-Sleep -Seconds 5; continue }

        & $mesh --host $HostPort `
                --set mqtt.address mosquitto `
                --set mqtt.username "" `
                --set mqtt.password "" `
                --set mqtt.enabled true `
                --set mqtt.encryption_enabled false `
                --set mqtt.json_enabled true `
                --set mqtt.tls_enabled false `
                --ch-index 0 `
                --ch-set uplink_enabled true `
                --ch-set downlink_enabled true 2>&1 | Out-Null

        if ($LASTEXITCODE -eq 0) {
            & $mesh --host $HostPort --ch-add mqtt 2>&1 | Out-Null
            & $mesh --host $HostPort --ch-index 1 `
                    --ch-set uplink_enabled true `
                    --ch-set downlink_enabled true 2>&1 | Out-Null
            Log-Ok "  Узел $HostPort настроен (канал 0 + mqtt канал 1)"
            return
        }
        if ($i -lt 12) { Start-Sleep -Seconds 5 }
    }
    Log-Warn "  Не удалось настроить $HostPort после 12 попыток"
}

# ---------------------------------------------------------------------------
# setup single-host
# ---------------------------------------------------------------------------
function Initialize-SingleHost {
    Log-Info "=== РЕЖИМ: Single-host ==="

    if (-not (Assert-Docker))  { return }
    if (-not (Assert-Python))  { return }
    if (-not (Assert-Compose)) { return }

    # 1. Сборка и запуск
    Log-Info "Шаг 1: Сборка и запуск контейнеров..."
    Push-Location $ComposeDir
    Invoke-Compose @("--profile", "single-host", "up", "-d", "--build")
    if ($LASTEXITCODE -ne 0) { Pop-Location; Log-Error "docker compose up завершился с ошибкой"; return }
    Pop-Location

    # 2. Ждём MQTT и узлы
    Wait-PortOpen -Port 1883 -TimeoutSec 60  -Label "MQTT брокер"
    Wait-ContainerUp -Service "nodeA" -TimeoutSec 120
    Wait-ContainerUp -Service "nodeB" -TimeoutSec 30
    Wait-PortOpen -Port 4403 -TimeoutSec 60  -Label "NodeA (порт 4403)"
    Wait-PortOpen -Port 4404 -TimeoutSec 30  -Label "NodeB (порт 4404)"

    # 3. Ждём ML-сервисы
    Wait-ContainerUp -Service "mesh-collector" -TimeoutSec 60
    Wait-ContainerUp -Service "mesh-graph"     -TimeoutSec 30
    Wait-ContainerUp -Service "mesh-router"    -TimeoutSec 30

    # 4. Настройка MQTT в узлах
    Log-Info "Шаг 3: Настройка MQTT в узлах..."
    Ensure-Venv
    Set-NodeMqttConfig "localhost:4403"
    Set-NodeMqttConfig "localhost:4404"

    # 5. Рестарт узлов
    Log-Info "Шаг 4: Перезапуск узлов..."
    Push-Location $ComposeDir
    Invoke-Compose @("restart", "nodeA", "nodeB")
    Pop-Location
    Start-Sleep -Seconds 3

    # 6. UI
    Log-Info "Шаг 5: Проверка веб-UI..."
    Ensure-WebDeps

    # 6. Итог
    $lanIp = Get-LanIP
    Log-Ok ""
    Log-Ok "Стенд готов!"
    Log-Info "  MQTT брокер : localhost:$BrokerPort"
    Log-Info "  NodeA (CLI) : localhost:4403"
    Log-Info "  NodeB (CLI) : localhost:4404"
    if ($lanIp) { Log-Info "  LAN-адрес   : $lanIp" }
    if (Test-Path (Join-Path $WebDir "package.json")) {
        Log-Info "  Запуск UI   : cd web && npm run dev  →  http://localhost:$UiPort"
    }
    Log-Info ""
    Log-Info "Запустите тест:  .\setup.ps1 single-host test"
}

# ---------------------------------------------------------------------------
# setup distributed
# ---------------------------------------------------------------------------
function Initialize-Distributed {
    Log-Info "=== РЕЖИМ: Distributed (ноут = брокер + UI) ==="

    if (-not (Assert-Docker))  { return }
    if (-not (Assert-Compose)) { return }

    Push-Location $ComposeDir
    Invoke-Compose @("--profile", "host", "up", "-d")
    Pop-Location

    Ensure-WebDeps

    $laptopIp = Get-LanIP
    Log-Ok "Брокер запущен. IP ноута в LAN: $laptopIp"
    Log-Info ""
    Log-Info "=== На каждом RPi выполните ==="
    Write-Host @"

# 1. Установите Docker (один раз)
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker `$USER && newgrp docker

# 2. Склонируйте репо и войдите в папку
git clone <repo-url> && cd <repo>/firmware/mesh-routing-bench/compose

# 3. Запустите узел (уникальный NODE_HWID на каждом RPi)
NODE_HWID=1001 docker compose --profile node up -d --build

# 4. Настройте MQTT (замените IP на: $laptopIp)
source ../.venv/bin/activate
meshtastic --host localhost:4403 --set mqtt.address $laptopIp --set mqtt.enabled true \
  --set mqtt.encryption_enabled false --set mqtt.json_enabled true --set mqtt.tls_enabled false \
  --ch-index 0 --ch-set uplink_enabled true --ch-set downlink_enabled true
meshtastic --host localhost:4403 --ch-add mqtt
meshtastic --host localhost:4403 --ch-index 1 --ch-set uplink_enabled true --ch-set downlink_enabled true
NODE_HWID=1001 docker compose --profile node restart node
"@
    Log-Info "UI на ноуте: cd web && npm run dev  →  http://${laptopIp}:${UiPort}"
}

# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------
function Start-Stack {
    Log-Info "Запуск контейнеров..."
    Push-Location $ComposeDir; Invoke-Compose @("up", "-d"); Pop-Location
    Log-Ok "Готово"
}

function Stop-Stack {
    Log-Info "Остановка контейнеров..."
    Push-Location $ComposeDir; Invoke-Compose @("stop"); Pop-Location
    Log-Ok "Контейнеры остановлены"
}

function Invoke-NodeConfigure {
    Log-Info "=== Настройка MQTT на узлах (без пересборки) ==="
    if (-not (Assert-Compose)) { return }
    Ensure-Venv

    Set-NodeMqttConfig "localhost:4403"
    Set-NodeMqttConfig "localhost:4404"

    Log-Info "Перезапуск узлов для применения конфигурации..."
    Push-Location $ComposeDir
    Invoke-Compose @("restart", "nodeA", "nodeB")
    Pop-Location
    Start-Sleep -Seconds 5

    Log-Ok "Конфигурация применена — проверьте:"
    Log-Info "  docker logs mesh-nodeA | grep -i mqtt"
    Log-Info "  docker exec mesh-mosquitto mosquitto_sub -h localhost -t '#' -v"
}

function Reset-Stack {
    Log-Info "Сброс контейнеров (volumes сохранены)..."
    Push-Location $ComposeDir; Invoke-Compose @("down"); Pop-Location
    Log-Ok "Готово"
}

function Remove-Stack {
    Log-Warn "Полный сброс — удаляем контейнеры И volumes (конфиг узлов сотрётся)..."
    $ans = Read-Host "Продолжить? [y/N]"
    if ($ans -notmatch '^[Yy]') { Log-Info "Отменено"; return }
    Push-Location $ComposeDir; Invoke-Compose @("down", "-v"); Pop-Location
    Log-Ok "Готово"
}

function Show-Status {
    if (-not (Assert-Compose)) { return }
    Push-Location $ComposeDir

    Log-Info "--- Контейнеры ---"
    Invoke-Compose @("ps")

    Write-Host ""
    Log-Info "--- Порты ---"
    @(
        @{ Port = 1883;    Label = "MQTT broker";  Fatal = $true  }
        @{ Port = 9001;    Label = "MQTT WS";      Fatal = $false }
        @{ Port = 4403;    Label = "NodeA gRPC";   Fatal = $false }
        @{ Port = 4404;    Label = "NodeB gRPC";   Fatal = $false }
        @{ Port = $UiPort; Label = "Web UI";       Fatal = $false }
    ) | ForEach-Object {
        if (Test-Port $_.Port) {
            Log-Ok  "$($_.Label) — открыт (localhost:$($_.Port))"
        } elseif ($_.Fatal) {
            Log-Error "$($_.Label) — НЕ РАБОТАЕТ (localhost:$($_.Port))"
        } else {
            Log-Warn "$($_.Label) — закрыт (localhost:$($_.Port))"
        }
    }

    Write-Host ""
    Log-Info "--- Последние логи (10 строк) ---"
    Invoke-Compose @("logs", "--tail=10")

    Pop-Location
}

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
function Run-Test {
    Log-Info "=== Тест стенда ==="
    $ok = $true

    # 1. Порты
    Log-Info "1. Проверка портов..."
    foreach ($p in @(1883, 4403, 4404)) {
        if (Test-Port $p) { Log-Ok  "  порт $p — OK"
        } else             { Log-Warn "  порт $p — недоступен"; $ok = $false }
    }

    # 2. Контейнеры
    Log-Info "2. Статус контейнеров..."
    Push-Location $ComposeDir
    $ps = Invoke-Compose @("ps", "--format", "table") 2>&1 | Out-String
    Pop-Location
    $ps -split "`n" | Where-Object { $_ -match '\S' } | ForEach-Object { Write-Host "  $_" }

    # 3. MQTT: подписка в фоне → sendtext broadcast → смотрим что пришло
    Log-Info "3. Тест MQTT + meshtastic sendtext (broadcast)..."
    Ensure-Venv
    $mesh = Get-MeshtasticExe

    if (-not (Test-Port 1883)) {
        Log-Warn "  MQTT брокер недоступен (порт 1883 закрыт) — пропускаем"
    } else {
        $tmpOut = [System.IO.Path]::GetTempFileName()
        $tmpErr = [System.IO.Path]::GetTempFileName()

        $subProc = Start-Process docker `
            -ArgumentList @("exec", "mesh-mosquitto", "mosquitto_sub",
                            "-h", "localhost", "-t", "#", "-v") `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $tmpOut `
            -RedirectStandardError  $tmpErr

        Start-Sleep -Seconds 1

        if ($mesh -and (Test-Port 4403)) {
            Log-Info "  Отправка 'bench-test' через NodeA (порт 4403)..."
            $sendOut = & $mesh --host localhost:4403 --sendtext "bench-test" --ch-index 0 2>&1 | Out-String
            if ($LASTEXITCODE -eq 0) { Log-Ok  "  sendtext OK" }
            else                     { Log-Warn "  sendtext вернул ошибку: $($sendOut.Trim())" }
        } else {
            Log-Warn "  meshtastic CLI не найден или NodeA недоступна — sendtext пропущен"
        }

        Log-Info "  Ожидание MQTT-сообщений (8 сек)..."
        Start-Sleep -Seconds 8
        $subProc | Stop-Process -Force -ErrorAction SilentlyContinue
        [void]$subProc.WaitForExit(2000)

        $lines = Get-Content $tmpOut -ErrorAction SilentlyContinue |
                 Where-Object { $_ -match '\S' }
        Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue

        if ($lines) {
            Log-Ok "  Получено MQTT-сообщений: $($lines.Count)"
            $lines | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
            if ($lines.Count -gt 10) { Write-Host "    ... (и ещё $($lines.Count - 10))" }
        } else {
            Log-Warn "  Сообщений не получено — проверьте: docker exec mesh-mosquitto mosquitto_sub -h localhost -t '#' -v"
        }
    }

    # -----------------------------------------------------------------------
    # 4. ML-сервисы: collector → graph_service → recommendation_engine
    # -----------------------------------------------------------------------
    Log-Info "4. Тест ML-сервисов..."

    # 4a. Collector — проверяем, что узлы появились в SQLite-БД.
    #     nodeinfo-пакеты рассылаются при старте и периодически (раз в 30 мин),
    #     поэтому ждём до 30 сек прежде чем сдаться.
    Log-Info "  4a. Collector — ожидание данных в БД (до 30 сек)..."
    $collectorOk = $false
    for ($i = 1; $i -le 6; $i++) {
        try {
            $nodeCount = docker exec mesh-collector python -c `
                "import os,sqlite3; db=os.getenv('MESHTASTIC_DB','/var/lib/meshtastic/mesh_network.db'); c=sqlite3.connect(db); print(c.execute('SELECT COUNT(*) FROM nodes').fetchone()[0])" 2>&1
            if ($LASTEXITCODE -eq 0 -and $nodeCount -match '^\d+' -and [int]$nodeCount -gt 0) {
                Log-Ok "  Collector: $([int]$nodeCount) узел(ов) сохранено в БД"
                $collectorOk = $true; break
            }
        } catch {}
        if ($i -lt 6) { Start-Sleep -Seconds 5 }
    }
    if (-not $collectorOk) {
        Log-Warn "  Collector: узлы пока не появились (nodeinfo рассылаются при старте и раз в 30 мин)"
        Log-Info "             Проверьте вручную: docker logs mesh-collector --tail=20"
    }

    # 4b. Graph service — ищем строку «Graph rebuilt» в логах контейнера.
    Log-Info "  4b. Graph service — проверка логов..."
    $graphLog = docker logs mesh-graph 2>&1 | Select-String "Graph rebuilt" | Select-Object -Last 1
    if ($graphLog) {
        Log-Ok "  Graph service: $($graphLog.Line.Trim())"
    } else {
        Log-Warn "  Graph service: пересборка ещё не зафиксирована"
        Log-Info "             Проверьте: docker logs mesh-graph --tail=20"
    }

    # 4c. Recommendation engine
    #     Шаги:
    #       1) вливаем тестовое ребро nodeA↔nodeB прямо в SQLite через mesh-collector
    #       2) перезапускаем mesh-router, чтобы граф подхватил новое ребро
    #       3) подписываемся на routing/recommendation/#
    #       4) публикуем фейковое text-сообщение nodeA→nodeB через mosquitto_pub
    #       5) ждём рекомендацию
    Log-Info "  4c. Recommendation engine — тест рекомендации маршрута..."

    # Шаг 1: инжект тестового ребра (!000003e9 = nodeA hwid=1001, !000003ea = nodeB hwid=1002)
    $ts = [int]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())
    $pyInject = "import os,sqlite3; db=os.getenv('MESHTASTIC_DB','/var/lib/meshtastic/mesh_network.db'); conn=sqlite3.connect(db); [conn.execute('INSERT OR REPLACE INTO edges VALUES(?,?,?,?)',t) for t in [('!000003e9','!000003ea',9.0,$ts),('!000003ea','!000003e9',9.0,$ts)]]; conn.commit(); conn.close(); print('ok')"
    $injRes = docker exec mesh-collector python -c $pyInject 2>&1
    if ($injRes -match '\bok\b') {
        Log-Ok "  Тестовое ребро nodeA↔nodeB (!000003e9 ↔ !000003ea) добавлено"
    } else {
        Log-Warn "  Не удалось добавить тестовое ребро: $injRes"
    }

    # Шаг 2: перезапускаем роутер — он подхватит ребро из БД при старте
    Log-Info "  Перезапуск mesh-router (подхватит новые рёбра)..."
    Push-Location $ComposeDir
    Invoke-Compose @("restart", "mesh-router") 2>&1 | Out-Null
    Pop-Location
    Start-Sleep -Seconds 8   # ждём переподключения к MQTT

    # Шаг 3: подписываемся на routing/recommendation/#
    $tmpRec    = [System.IO.Path]::GetTempFileName()
    $tmpRecErr = [System.IO.Path]::GetTempFileName()
    $recProc   = Start-Process docker `
        -ArgumentList @("exec", "mesh-mosquitto", "mosquitto_sub",
                        "-h", "localhost", "-t", "routing/recommendation/#", "-v") `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput $tmpRec `
        -RedirectStandardError  $tmpRecErr
    Start-Sleep -Seconds 1

    # Шаг 4: публикуем text-сообщение через mosquitto_pub напрямую в брокер.
    #         from=1001 (nodeA !000003e9), to=1002 (nodeB !000003ea)
    $msgJson = '{"from":1001,"to":1002,"type":"text","sender":"!000003e9","payload":{"text":"route-test"}}'
    Log-Info "  Публикация тестового text-сообщения nodeA→nodeB..."
    docker exec mesh-mosquitto mosquitto_pub `
        -h localhost `
        -t "msh/2/json/LongFast/!000003e9" `
        -m $msgJson 2>&1 | Out-Null

    # Шаг 5: ждём рекомендацию
    Log-Info "  Ожидание рекомендации маршрута (10 сек)..."
    Start-Sleep -Seconds 10
    $recProc | Stop-Process -Force -ErrorAction SilentlyContinue
    [void]$recProc.WaitForExit(2000)

    $recLines = Get-Content $tmpRec -ErrorAction SilentlyContinue |
                Where-Object { $_ -match '\S' }
    Remove-Item $tmpRec, $tmpRecErr -ErrorAction SilentlyContinue

    if ($recLines) {
        Log-Ok "  Recommendation engine: рекомендация получена!"
        $recLines | ForEach-Object { Write-Host "    $_" }
    } else {
        Log-Warn "  Рекомендация не получена — проверьте вручную:"
        Log-Info "    docker logs mesh-router --tail=30"
        Log-Info "    docker exec mesh-mosquitto mosquitto_sub -h localhost -t 'routing/#' -v"
    }

    Write-Host ""
    if ($ok) { Log-Ok  "Тест завершён — стенд работает" }
    else      { Log-Warn "Тест завершён — есть предупреждения (см. выше)" }
}

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
function Show-Help {
    Write-Host @"
Mesh Routing Bench  —  setup.ps1 (Windows/PowerShell)

Использование:
    .\setup.ps1 [<режим>] [<команда>]

Режимы:          single-host (по умолчанию) | distributed
Команды:         setup* | start | stop | reset | clean | status | test | help

Быстрый старт:
    .\setup.ps1 single-host setup    # сборка + запуск + настройка
    .\setup.ps1 single-host test     # проверка портов + MQTT
    .\setup.ps1 single-host status   # статус контейнеров

Управление:
    .\setup.ps1 single-host start    # запустить (без пересборки)
    .\setup.ps1 single-host stop     # остановить
    .\setup.ps1 single-host reset    # down (volumes сохранены)
    .\setup.ps1 single-host clean    # down -v (volumes удалены!)

Адреса:
    MQTT TCP    tcp://localhost:$BrokerPort
    MQTT WS     ws://localhost:9001
    NodeA gRPC  localhost:4403
    NodeB gRPC  localhost:4404
    Web UI      http://localhost:$UiPort  (после: cd web && npm run dev)

Первый запуск требует политики выполнения скриптов:
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
"@
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
switch ($Command) {
    "setup" {
        switch ($Mode) {
            "single-host" { Initialize-SingleHost }
            "distributed" { Initialize-Distributed }
            default       { Log-Error "Неизвестный режим: $Mode"; Show-Help; exit 1 }
        }
    }
    "configure" { Invoke-NodeConfigure }
    "start"     { if (Assert-Compose) { Start-Stack }  }
    "stop"      { if (Assert-Compose) { Stop-Stack }   }
    "reset"     { if (Assert-Compose) { Reset-Stack }  }
    "clean"     { if (Assert-Compose) { Remove-Stack } }
    "status"    { Show-Status }
    "test"      { Run-Test }
    "help"      { Show-Help }
    default  { Log-Error "Неизвестная команда: $Command"; Show-Help; exit 1 }
}
