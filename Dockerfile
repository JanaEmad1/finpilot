FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY finpilot ./finpilot
COPY eval ./eval

# Build the synthetic database and the baseline intent model inside the image, so the
# container works with no extra steps. (Mount ./models to use the fine-tuned DistilBERT.)
RUN python -m finpilot.data.generate && python -m finpilot.intent.baseline && python -m eval.eval_intent

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "finpilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
