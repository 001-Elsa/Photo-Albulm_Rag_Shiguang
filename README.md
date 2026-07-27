# 拾光 —— 离线多模态语义检索平台

**企业模式**:组织内共享媒资库(商品图/质检照片/设计素材)的自然语言检索——多用户登录、RBAC、审计、指标、Docker 部署。
**个人模式**:`auth_enabled=false`,就是你自己的全离线相册第二大脑。

几万张照片不用打一个标签,直接用自然语言搜:

> "去年在海边拍的日落" · "那张高铁票截图" · "和妈妈的合影" · "2024年3月的樱花"

**数据一字节不出电脑**:图文向量、OCR、人脸聚类、查询解析全部本地推理。

## 功能

| 功能 | 说明 |
|---|---|
| 自然语言搜图 | Chinese-CLIP 图文向量,"搜日落出日落" |
| 截图文字搜索 | OCR 全文索引(FTS5 BM25),"高铁票""账单"直接命中 |
| 时间/条件过滤 | "去年""2024年3月""冬天""上个月""截图" 自动解析成结构化条件 |
| 地点检索 | GPS → 城市名全离线("在杭州拍的"),v0.9 新增 |
| 以图搜图 | 点开照片一键找视觉相似,v0.9 新增 |
| 人物检索 | 人脸聚类,命名后可搜"和妈妈的合影" |
| 混合排序 | 语义 + 文字 + 条件三路 RRF 融合 |
| 相似去重 | 感知哈希找近重复,释放磁盘 |
| 增量索引 | 新照片自动入库;中途关机,下次从断点继续 |
| 多用户/RBAC | 登录认证、admin/viewer 角色、审计日志(v1.0) |
| 可插拔向量库 | 内存矩阵(单机)/ pgvector-HNSW(百万级),自动降级(v1.0) |
| 可观测/运维 | /healthz、Prometheus /metrics、JSON 日志、限流、Docker、CI(v1.0) |
| 性能优化 | ONNX int8 量化,CPU 也能跑;评估脚本量化 Recall@K/MRR |

## 快速开始

```bash
# 1. 安装核心依赖
pip install -r requirements.txt

# 2. 安装语义模型(推荐)
pip install torch transformers    # 首次运行自动下载 Chinese-CLIP(约 700MB)

# 3. 启动
python run.py                     # Windows 可双击 start.bat
# 打开 http://127.0.0.1:8626 → 设置 → 填相册目录 → 开始索引
```

可选增强:

```bash
pip install rapidocr-onnxruntime   # 截图文字搜索
pip install insightface onnxruntime  # 人脸聚类
pip install pillow-heif            # iPhone HEIC
```

详细步骤见 [docs/安装运行指南.md](docs/安装运行指南.md)。

## 架构

```
Web UI (原生JS)
   │ REST / SSE
FastAPI 服务
   ├─ 查询解析  自然语言 → {语义, 关键词, 年/月, 人物, 截图}   (规则引擎 / 本地Ollama)
   ├─ 检索引擎  语义向量(内存矩阵) + OCR FTS5 + 结构化过滤 → RRF 融合
   └─ 索引管线  扫描 → 缩略图/EXIF/pHash → CLIP向量 → OCR → 人脸 → 聚类
                 ↑ watchdog 增量监听          ↑ 三阶段断点标记,可随时中断续建
SQLite(元数据 + FTS5 + 向量BLOB) / 本地模型(ONNX int8 或 PyTorch)
```

设计说明见 [docs/架构设计.md](docs/架构设计.md),与八周开发计划的对照见 [docs/八周功能对照.md](docs/八周功能对照.md)。

## 评估与性能

```bash
python eval/build_testset.py          # 生成测试集模板并抽样出题
python eval/run_eval.py --tag v1      # Recall@1/5/10、MRR、延迟 P50/P95
python eval/run_eval.py --compare v1 v2

python scripts/export_onnx.py --int8  # 导出 + int8 量化
python scripts/benchmark.py --backend transformers --tag fp32
python scripts/benchmark.py --backend onnx --tag int8
python scripts/benchmark.py --compare fp32 int8
```

评估口径与调优建议见 [docs/评估指南.md](docs/评估指南.md)。

## 测试

```bash
python -m pytest tests/ -q   # 36 个单元测试,无重依赖即可运行
```

版本变更见 [docs/升级说明.md](docs/升级说明.md);企业级架构与部署见 [docs/企业级改造说明.md](docs/企业级改造说明.md)。

## 隐私

- 无任何外网请求(模型下载除外,也可手动离线放置)
- 所有数据落在项目 `data/` 目录,删掉即彻底清除
