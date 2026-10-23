# Образ API. Модель собирается при сборке образа: данные качаются с football-data.co.uk.
# Для обновления прогнозов пересобирайте образ (или монтируйте свежий models/predictor.pkl).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY football_predictor ./football_predictor
RUN python -m football_predictor download && python -m football_predictor train --no-market \
    && rm -rf data/features.pkl
ENV FOOTBALL_MODEL_PATH=/app/models/predictor.pkl
EXPOSE 8000
CMD ["uvicorn", "football_predictor.api:app", "--host", "0.0.0.0", "--port", "8000"]
