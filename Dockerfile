FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev --frozen --no-install-project

COPY . .
RUN uv sync --extra dev --frozen

CMD ["uv", "run", "pytest"]
