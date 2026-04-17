FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Copy application
COPY . .

# Ensure data directory exists
RUN mkdir -p data

EXPOSE ${PORT:-5001}

ENV FLASK_ENV=production

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT','5001') + '/health')" || exit 1

CMD ["python", "app.py"]
