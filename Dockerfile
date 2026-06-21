FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（利用 docker layer 缓存）
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# 复制源码
COPY src/ src/
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "mindforge.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
