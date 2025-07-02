FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use environment variable PORT or fallback to 8001 in the CMD
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}
