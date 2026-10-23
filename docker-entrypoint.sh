#!/bin/sh
# serve   — запустить API (если модели нет — сначала скачать данные и обучить)
# refresh — перекачать текущий сезон и переобучить модель (затем перезапустите контейнер с serve)
# любая другая команда — передаётся в CLI: docker run ... football-predictor predict --league ...
set -e

train() {
    python -m football_predictor download
    python -m football_predictor train --no-market --rebuild
}

case "$1" in
    serve)
        if [ ! -f "$FOOTBALL_MODEL_PATH" ]; then
            echo "Модель не найдена — скачиваю данные и обучаю (несколько минут)..."
            train
        fi
        exec uvicorn football_predictor.api:app --host 0.0.0.0 --port "${PORT:-8000}"
        ;;
    refresh)
        train
        ;;
    *)
        exec python -m football_predictor "$@"
        ;;
esac
