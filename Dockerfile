FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_EXTRA_INDEX_URL=${TORCH_INDEX_URL} \
	PIP_NO_CACHE_DIR=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
