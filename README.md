# 拾光——离线多模态相册语义检索系统

基于 Chinese-CLIP、OCR、EXIF 和人脸特征构建的离线多模态照片检索系统，支持通过自然语言描述、时间地点、图片文字和人物信息搜索本地照片。

本项目的核心是**多模态检索**，不是完整 RAG：检索结果由排序器直接返回，没有交给大模型生成回答。

## 核心能力

- Chinese-CLIP 图文跨模态检索。
- OCR 全文检索与 EXIF 时间、地点、截图等结构化过滤。
- 人脸聚类、人物命名检索和感知哈希近重复检测。
- 语义向量与 OCR 结果通过 RRF 融合，避免直接比较不同通道的原始分数。
- 默认使用 NumPy 完成单机向量检索。
- 提供 pgvector-HNSW 实验性适配，但尚未完成大规模容量与并发验证。
- FastAPI API、后台索引、身份认证、审计、指标和限流。

## 系统架构

```text
照片目录
   ↓
文件扫描 / watchdog 监听
   ↓
EXIF / Thumbnail / pHash
   ├─ Chinese-CLIP 图像向量
   ├─ RapidOCR 文字
   └─ InsightFace 人脸特征
   ↓
SQLite Metadata + FTS5 + NumPy Vector Store
                    └─ pgvector-HNSW（实验性）
   ↓
Query Parser → Metadata Filter → Semantic/OCR Recall
   ↓
RRF Fusion → FastAPI
```

## 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements-core.txt
python run.py
```

打开 `http://127.0.0.1:8626`，配置相册目录后启动索引。仅安装核心依赖时会使用 `demo` 伪向量，它只能验证程序链路，**不具备真实语义检索能力**。

完整 CPU AI 依赖：

```bash
pip install -r requirements-ai-cpu.txt
```

模型首次加载可能需要联网下载；照片数据和模型推理均保留在本地。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
python -m compileall -q shiguang
```

默认 CI 运行不下载大型模型的自动化测试和 Docker 构建。真实 AI 模型测试需要在具备相应模型和算力的环境单独执行。

## 检索评测

```bash
python eval/build_testset.py --sample 100
python eval/run_eval.py --tag baseline
```

评测脚本输出 Recall@1/5/10、MRR 和 P50/P95 延迟。仓库不提供虚构指标；公开结果前应同时记录照片数量、人工查询数量、硬件和模型版本。

## 项目边界

- 当前是检索系统，不是包含答案生成环节的完整 RAG。
- 尚未通过真实测试证明“百万级照片”“企业级稳定性”或“高并发部署”。
- pgvector 是实验性适配，默认后端仍是 NumPy 单机检索。
- Docker 默认构建核心服务，不包含大型 AI 权重。
- 人脸聚类采用内存中的贪心算法，尚未针对大规模图库验证。

更多说明见 [架构设计](docs/架构设计.md)、[评估指南](docs/评估指南.md)和[安装运行指南](docs/安装运行指南.md)。

## License

[MIT](LICENSE)
