# 拾光——离线多模态相册语义检索系统

基于 Chinese-CLIP、OCR、EXIF 和人脸特征构建的离线多模态照片检索系统，支持通过自然语言描述、时间地点、图片文字和人物信息搜索本地照片。

本项目的核心是**多模态检索**，不是完整 RAG：检索结果由排序器直接返回，没有交给大模型生成回答。

## 核心能力

- Chinese-CLIP 图文跨模态检索。
- OCR 原文与中文 2/3-gram 联合索引，支持“高铁”“G1024”等局部文字查询。
- EXIF 时间、GPS 城市、截图和已命名人物等结构化过滤。
- 规则式查询意图识别，按 `scene/document/person/time_location/hybrid` 动态调整 RRF 权重。
- 返回 `matched_by`、各通道排名和 OCR 片段，便于解释检索结果。
- SQLite 持久化索引任务，支持状态机、失败重试、重启恢复、模型版本触发重建和幂等覆盖。
- 默认使用 NumPy 单机向量检索；提供未经大规模压测的 pgvector-HNSW **实验性适配**。

## 系统架构

```text
照片目录
   ↓
文件扫描 / watchdog 监听
   ↓
SQLite index_jobs（pending/running/succeeded/failed/skipped）
   ↓
EXIF / Thumbnail / pHash
   ├─ Chinese-CLIP 图像向量
   ├─ RapidOCR 原文 + 中文 n-gram
   └─ InsightFace 人脸特征
   ↓
SQLite Metadata + FTS5 + NumPy Vector Store
                    └─ pgvector-HNSW（实验性）
   ↓
Query Parser → Metadata Filter → Multi-channel Recall
   ↓
Intent-aware weighted RRF
   ↓
FastAPI + 可解释检索结果
```

索引结果写入和任务状态更新处于同一短事务。重复执行时，向量和 OCR 使用 upsert，人脸先删除旧记录再写入；任务通过照片内容哈希、处理器版本和唯一约束合并。

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

## 健康与降级

`GET /healthz` 会报告真实组件状态：

```json
{
  "status": "degraded",
  "database": "ready",
  "embedder": "demo",
  "semantic_search_ready": false,
  "ocr_ready": false,
  "face_search_ready": false
}
```

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
python -m compileall -q shiguang
```

默认 CI 运行无需下载大模型的单元/集成测试、Docker 构建，并在独立 `pgvector/pgvector:pg16` 服务中验证向量插入、更新、HNSW 查询。真实 AI 模型测试仍需在具备相应模型和算力的环境单独执行。

## 检索评测

```bash
python eval/build_testset.py --sample 100
python eval/run_eval.py --mode clip_only --tag clip_only
python eval/run_eval.py --mode ocr_only --tag ocr_only
python eval/run_eval.py --mode fixed --tag fixed
python eval/run_eval.py --mode dynamic --tag dynamic
python eval/run_eval.py --compare fixed dynamic
```

| 方案 | Recall@1 | Recall@5 | Recall@10 | MRR | P95 |
|---|---:|---:|---:|---:|---:|
| CLIP only | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| OCR only | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 固定权重融合 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 查询意图动态融合 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

仓库不提供虚构指标。发布简历前应记录照片数量、查询数量、硬件、模型名称与版本，再填入真实结果。

## 项目边界

- 当前是检索系统，不是包含答案生成环节的完整 RAG。
- NumPy 后端适合个人单机图库，但尚未声明或证明具体规模上限。
- pgvector 目前是实验性旁路同步适配，已覆盖基础集成测试，但尚未完成大规模容量与并发验证。
- Docker 默认构建核心服务，不包含大型 AI 权重。
- 人脸聚类采用内存中的贪心算法，尚未针对大规模图库验证。

更多说明见 [架构设计](docs/架构设计.md)、[评估指南](docs/评估指南.md)和[安装运行指南](docs/安装运行指南.md)。

## License

[MIT](LICENSE)
