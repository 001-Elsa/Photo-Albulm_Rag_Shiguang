FROM python:3.11-slim

WORKDIR /app

# 先装依赖,充分利用镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pillow numpy watchdog

# 语义模型按需启用(镜像体积换能力):
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir transformers rapidocr-onnxruntime onnxruntime

COPY shiguang/ shiguang/
COPY run.py .

# 数据(库/缩略图/密钥)全部落在 /data 卷
ENV SHIGUANG_DATA=/data \
    SHIGUANG_HOST=0.0.0.0
VOLUME /data

EXPOSE 8626
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8626/healthz')" || exit 1

CMD ["python", "run.py"]
