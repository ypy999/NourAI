FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install minimal runtime dependencies (CPU PyTorch + NumPy)
COPY requirements/base.txt ./requirements/base.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/base.txt

# Copy source code and create non-root runtime user
COPY . .
RUN useradd --create-home --shell /bin/bash jebrex \
    && mkdir -p /app/data/production /app/data/cache \
    && chown -R jebrex:jebrex /app

USER jebrex

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:7860/api/v1/health', timeout=3); assert r.status == 200"

CMD ["python", "run_nourai.py", "--no-browser", "--host", "0.0.0.0"]
