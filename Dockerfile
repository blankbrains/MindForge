FROM docker.m.daocloud.io/library/node:22-alpine@sha256:16e22a550f3863206a3f701448c45f7912c6896a62de43add43bb9c86130c3e2 AS frontend-builder

WORKDIR /build/mindforge-web
COPY mindforge-web/package.json mindforge-web/package-lock.json ./
RUN npm ci
COPY mindforge-web/ ./
ARG VITE_API_TIMEOUT_MS=30000
ARG VITE_RESEARCH_TIMEOUT_MS=900000
ENV VITE_API_TIMEOUT_MS=${VITE_API_TIMEOUT_MS} \
    VITE_RESEARCH_TIMEOUT_MS=${VITE_RESEARCH_TIMEOUT_MS}
RUN npm run build


FROM docker.m.daocloud.io/library/python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS runtime

WORKDIR /app

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY src/ src/

COPY --from=frontend-builder /build/mindforge-web/dist /app/mindforge-web/dist

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --create-home --uid "${APP_UID}" --gid "${APP_GID}" appuser \
    && mkdir -p /app/data /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent "http://127.0.0.1:${API_PORT}/api/v1/ready" > /dev/null || exit 1

CMD ["sh", "-c", "uvicorn mindforge.api.server:app --host \"${API_HOST}\" --port \"${API_PORT}\""]
