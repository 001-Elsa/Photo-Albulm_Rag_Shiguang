FROM python:3.14-slim-bookworm

WORKDIR /app

# 先装依赖,充分利用镜像层缓存
COPY requirements-core.txt requirements-enterprise.txt ./
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip wheel \
    && python -m pip install --no-cache-dir -r requirements-enterprise.txt \
    && python -m pip uninstall --yes pip setuptools wheel msgpack

# 语义模型按需启用(镜像体积换能力):
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir transformers rapidocr-onnxruntime onnxruntime

COPY shiguang/ shiguang/
COPY run.py .
COPY LICENSE .

# 数据库和缩略图落在 /data 卷；多人部署的密钥必须由环境变量注入
ENV SHIGUANG_DATA=/data \
    SHIGUANG_HOST=0.0.0.0
VOLUME /data

EXPOSE 8626
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8626/healthz')" || exit 1

CMD ["python", "run.py"]
