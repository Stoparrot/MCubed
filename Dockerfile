# CPU container for portability checks on Linux or Docker Desktop.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534
WORKDIR /workspace
COPY requirements.txt .
# Install the CPU wheel explicitly, including when built on an x86-64 cloud host.
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check
COPY LICENSE .
COPY mcubed mcubed
COPY vendor vendor
COPY configs configs
COPY dashboard dashboard
COPY datasets datasets
COPY tools tools
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "mcubed.doctor"]
