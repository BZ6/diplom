FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p saved_model

ENV PORT=8000
ENV MODEL_PATH=/app/saved_model/routing_model.pt

EXPOSE 8000

CMD ["python", "serve.py"]
