FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y \
  libfreetype6-dev \
  libpng-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose both Gradio (7860) and FastAPI (8000) ports
EXPOSE 7860 8000

# Environment variables for configuration
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV API_BASE_URL=http://localhost:8000
ENV MODEL_NAME=clinical-trial-rl
ENV PYTHONUNBUFFERED=1

# Default: run Gradio app
# To run inference instead: docker run --entrypoint "python inference.py" ...
CMD ["python", "app.py"]
