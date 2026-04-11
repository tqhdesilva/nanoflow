FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install uv
RUN pip install uv

WORKDIR /app

# Copy project and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Default: print help. Override with actual training command.
ENTRYPOINT ["uv", "run", "python"]
CMD ["train.py", "--help"]
