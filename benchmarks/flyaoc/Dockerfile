FROM python@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
COPY --from=ghcr.io/astral-sh/uv@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc /uv /usr/local/bin/uv
WORKDIR /runtime
COPY pyproject.toml uv.lock ./
ENV UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_PYTHON_DOWNLOADS=never
RUN uv sync --frozen --extra baselines --no-install-project --no-dev && uv cache clean
ENV PATH="/runtime/.venv/bin:$PATH"
WORKDIR /output
CMD ["sleep", "infinity"]
