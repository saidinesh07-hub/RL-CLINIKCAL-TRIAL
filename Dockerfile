FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y \
  libfreetype6-dev \
  libpng-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose both common API/UI ports
EXPOSE 7860
EXPOSE 8000

# Environment variables for configuration
ENV API_BASE_URL=http://localhost:7860
ENV MODEL_NAME=clinical-trial-rl
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
