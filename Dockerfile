FROM python:3.12-slim-bookworm

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG PYTORCH_PACKAGES="torch==2.6.0+cu124 torchvision==0.21.0+cu124"

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash build-essential ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project \
      --no-install-package torch \
      --no-install-package torchvision \
    && uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
      --index-url "${PYTORCH_INDEX_URL}" \
      ${PYTORCH_PACKAGES} \
    && uv pip check --python "${UV_PROJECT_ENVIRONMENT}/bin/python"

COPY . .

RUN python scripts/runpod_cuda_smoke.py --allow-cpu

CMD ["python", "scripts/runpod_cuda_smoke.py"]
