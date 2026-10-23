# Образ сервиса прогнозов. Данные и модель хранятся на томах /app/data и /app/models:
# при первом запуске контейнер сам скачает данные и обучит модель (~5 минут),
# дальше стартует сразу. Обновить данные и модель: команда `refresh` (см. README).
FROM python:3.12-slim

# libgomp1 — OpenMP-рантайм, без него не грузится LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY football_predictor ./football_predictor
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# sed убирает CRLF: git на Windows может переписать окончания строк, и тогда скрипт не запустится
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && useradd --create-home --uid 1000 app && mkdir -p data models && chown app:app data models
USER app

ENV FOOTBALL_MODEL_PATH=/app/models/predictor.pkl \
    PYTHONUNBUFFERED=1
VOLUME ["/app/data", "/app/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["serve"]
