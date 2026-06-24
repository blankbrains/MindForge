FROM python:3.11-slim

WORKDIR /app

# ── 统一 UTF-8 编码（防止中文乱码）──
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（利用 docker layer 缓存）
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

# 复制源码
COPY src/ src/
RUN pip install --no-cache-dir -e .

# 创建非 root 用户运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "mindforge.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
