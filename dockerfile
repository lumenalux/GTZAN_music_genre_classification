FROM python:3.10-slim-bullseye as builder

WORKDIR /usr/src/app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim-bullseye

WORKDIR /usr/src/app

COPY --from=builder /install /usr/local
COPY . .

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  NAME=World

EXPOSE 5000

CMD ["python", "predict.py"]
