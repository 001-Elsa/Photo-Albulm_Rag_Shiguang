# 拾光——离线多模态相册语义检索系统

基于 Chinese-CLIP、OCR、EXIF 和人脸特征构建的多模态照片检索系统。
核心是**多模态检索**，不是完整 RAG：结果由排序器直接返回，不交给大模型生成回答。

项目提供两种部署模式，能力边界不同，简历表述也必须分开写。

| 能力 | Personal Mode | Enterprise Mode |
|---|---|---|
| 业务库 | SQLite | PostgreSQL + pgvector + RLS |
| 对象存储 | 本地 `data/` | MinIO |
| 任务执行 | 进程内 Indexer | Celery Worker + Beat + Redis |
| 租户模型 | 单机相册 | Organization / Collection / Asset |
| OCR 检索 | FTS5 + 中文 n-gram | `simple` tsvector + `ILIKE` + `pg_trgm` |
| 人脸 | 检测、聚类、命名、按人物过滤 | 特征提取与存储；聚类/人物检索未完成 |
| 可观测性 | 基础 metrics / healthz | Prometheus、Grafana、OpenTelemetry |
| 验证程度 | 本地单测 + Demo 冒烟 | Postgres / Redis / MinIO / Celery 的上传索引搜索 E2E CI |

## Personal Mode

适合本地相册和简历中的检索算法主线。

```text
照片目录
  → 扫描 / watchdog
  → SQLite index_jobs
  → Thumbnail / Embedding / OCR / Face
  → SQLite + FTS5 + NumPy（可选旁路 pgvector）
  → Query Parser → 多通道召回 → Intent-aware RRF
```

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements-core.txt
python run.py
```

打开 `http://127.0.0.1:8626`。默认关闭认证并只监听本机。
仅安装核心依赖时使用 `demo` 伪向量，只能验证链路，不具备真实语义检索能力。

完整 CPU AI 依赖：

```bash
pip install -r requirements-ai-cpu.txt
SHIGUANG_MODEL_DOWNLOAD_ENABLED=true python run.py
```

## Enterprise Mode

适合展示后端工程能力：多租户、异步任务、对象存储、观测与迁移隔离。

```text
Client
  → Nginx → FastAPI
  → PostgreSQL / Redis / MinIO
  → Celery Worker / Beat
  → Embedding / OCR / Face 结果回写
  → Enterprise Search + Explainable Reranker
```

```bash
# 先把 deploy/secrets/*.example 复制为 *.txt 并填入真实密钥
for f in deploy/secrets/*.example; do cp "$f" "${f%.example}.txt"; done
# PowerShell: Get-ChildItem deploy/secrets/*.example | ForEach-Object {
#   Copy-Item $_ "$($_.FullName -replace '\.example$', '.txt')"
# }

docker compose up -d --build
curl http://127.0.0.1:8626/readyz
```

Compose 会启动：API、独立 Migration、Worker、Beat、PostgreSQL/pgvector、
Redis、MinIO、Nginx、Prometheus、Grafana、OTel Collector。

数据库迁移由独立管理员账号执行；API/Worker 只使用 `NOBYPASSRLS` 应用账号。

## 安全说明

- `.gitignore` 已排除 `data/`、数据库、日志、密钥和缓存。
- 仓库历史已清理旧的运行数据与密钥文件；若你持有旧克隆，请重新拉取或重建。
- 旧密钥/旧密码一律视为已泄露，必须轮换。
- 企业模式通过环境变量或 `_FILE` 注入密钥，不默认写明文密码文件。
- `Config.save()` 不会把数据库/MinIO/指标等敏感字段写入 `data/config.json`。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
python -m compileall -q shiguang
ruff check shiguang eval scripts tests run.py
mypy shiguang --ignore-missing-imports
```

CI 默认跑：

- Python 3.10/3.11/3.12 单元测试与覆盖率门槛
- Ruff / Mypy / compileall
- Personal Demo API 冒烟
- Docker 构建
- pgvector 旁路集成
- 企业栈组件与 HTTP→Worker E2E（Postgres / Redis / MinIO / Celery）
- Gitleaks、`pip-audit`、CodeQL 与镜像 Trivy 扫描

CI 覆盖“上传图片 → MinIO → Celery → embedding → 搜索返回”和跨组织 API
访问拒绝。真实故障注入、Locust 报告、人工标注 Recall 结果，仍需在目标环境
补齐后才能写进简历。

## 检索评测

```bash
python eval/build_testset.py --sample 100
# 人工标注 expected_paths 后：
python eval/run_eval.py --mode clip_only --tag clip_only
python eval/run_eval.py --mode ocr_only --tag ocr_only
python eval/run_eval.py --mode fixed --tag fixed
python eval/run_eval.py --mode dynamic --tag dynamic
python eval/run_eval.py --compare fixed dynamic
```

精排器是可配置、可解释的线性模型；可用 `eval/train_reranker.py` 从标注特征训练权重。
**在提交真实标注集和训练报告之前，不要声称“已基于标注数据训练学习排序模型”。**

| 方案 | Recall@1 | Recall@5 | Recall@10 | MRR | P95 |
|---|---:|---:|---:|---:|---:|
| CLIP only | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| OCR only | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 固定权重融合 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 查询意图动态融合 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

仓库不提供虚构指标。发布简历前应记录照片数量、查询数量、硬件、模型名称与版本，再填入真实结果。

## 明确未完成 / 不能夸大的边界

- Enterprise `FACE_CLUSTER` / `VECTOR_SYNC` 枚举存在，但 Worker 会对未实现处理器直接失败，不会假成功。
- Enterprise 尚未提供完整人物聚类、命名、合并拆分和按人物检索通道。
- 没有提交真实人工标注测试集，因此不能写具体 Recall / P95 / 十万级支持数字。
- 压测脚本（Locust、pgvector benchmark）已提供，但仓库不包含真实压测报告。
- Alembic 未采用；当前是独立 Migration 进程 + 版本化 SQL。这是可接受设计，不是硬伤。
- `PostgresRepository` 仍然偏大，后续可拆成 Identity / Asset / Job / Search 等仓库。

更多说明见 [架构设计](docs/架构设计.md)、[企业级改造说明](docs/企业级改造说明.md)、
[评估指南](docs/评估指南.md) 和 [安装运行指南](docs/安装运行指南.md)。

## License

[MIT](LICENSE)
