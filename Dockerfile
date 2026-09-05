# CPU container for portability checks on Linux or Docker Desktop.
FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mcubed mcubed
COPY vendor vendor
COPY configs configs
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "mcubed.doctor"]
