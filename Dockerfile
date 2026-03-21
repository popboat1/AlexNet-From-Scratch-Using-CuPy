FROM node:20-alpine AS builder

WORKDIR /app/frontend

COPY frontend/3d-visualizer/package*.json ./
RUN npm install

COPY frontend/3d-visualizer/ ./

RUN npm run build

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY --from=builder /app/frontend/out ./static

EXPOSE 7860

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "7860"]