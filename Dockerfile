FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY neo_sentinel ./neo_sentinel
COPY models ./models
COPY streamlit_app.py .
COPY .streamlit ./.streamlit

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import json,os,urllib.request; r=json.load(urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')); exit(0 if r['model_loaded'] else 1)"

# PORT is injected by Render, Railway, Fly and most container platforms.
CMD ["sh", "-c", "uvicorn neo_sentinel.api.app:app --host 0.0.0.0 --port ${PORT}"]
