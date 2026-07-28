"""W3:查询解析——自然语言 → 结构化检索条件。

"去年在海边拍的日落" → {semantic:"海边 日落", year_from:2025, year_to:2025}
"那张高铁票截图"     → {semantic:"高铁票", ocr:"高铁", screenshot:true}
"和妈妈的合影"       → {semantic:"合影", person:"妈妈"}

两种模式:
- rules  : 纯规则(默认,零依赖、毫秒级、可单测)
- ollama : 本地小模型 JSON 输出,失败自动回落 rules
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import date

log = logging.getLogger("shiguang.query")

CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
          "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}


@dataclass
class ParsedQuery:
    semantic: str = ""            # 语义检索文本
    keywords: list = field(default_factory=list)  # OCR 关键词
    year_from: int | None = None
    year_to: int | None = None
    months: list = field(default_factory=list)  # 命中任一月份;空=不限
    person: str | None = None     # 人物名(匹配已命名的人物簇)
    place: str | None = None      # 城市名(离线逆地理,v0.9)
    screenshot: bool | None = None  # True=只要截图 / None=不限
    intent: str = "scene"          # scene | document | person | time_location | hybrid

    def to_dict(self):
        return asdict(self)


# ---------- 规则解析 ----------

_RE_YEAR = re.compile(r"(?<!\d)(20\d{2})\s*年?(?!\d)")
_RE_MONTH = re.compile(r"(\d{1,2}|[一二三四五六七八九十]|十[一二])月")
_RE_PERSON = re.compile(r"(?:和|跟|与)\s*([一-龥A-Za-z]{1,6}?)\s*(?:的)?(?:合影|合照|一起)")

_SEASONS = {"春天": [3, 4, 5], "夏天": [6, 7, 8], "秋天": [9, 10, 11], "冬天": [12, 1, 2],
            "春季": [3, 4, 5], "夏季": [6, 7, 8], "秋季": [9, 10, 11], "冬季": [12, 1, 2]}

_STOPWORDS = ("的照片", "的图片", "照片", "图片", "拍的", "拍摄的", "那张", "这张",
              "一张", "帮我找", "找一下", "搜一下", "搜索", "查找")


def _relative_year(q: str, today: date) -> tuple[int | None, int | None, str]:
    pairs = [("大前年", -3), ("前年", -2), ("去年", -1), ("今年", 0)]
    for word, delta in pairs:
        if word in q:
            y = today.year + delta
            return y, y, q.replace(word, " ")
    return None, None, q


def parse_rules(query: str, today: date | None = None) -> ParsedQuery:
    today = today or date.today()
    q = query.strip()
    p = ParsedQuery()

    # 截图意图
    if "截图" in q or "截屏" in q:
        p.screenshot = True
        q = q.replace("截图", " ").replace("截屏", " ")

    # 人物:"和妈妈的合影"
    m = _RE_PERSON.search(q)
    if m:
        p.person = m.group(1)
        q = q.replace(m.group(0), " 合影 ")

    # 相对年份
    yf, yt, q = _relative_year(q, today)
    if yf:
        p.year_from, p.year_to = yf, yt

    # 绝对年份
    m = _RE_YEAR.search(q)
    if m:
        y = int(m.group(1))
        p.year_from, p.year_to = y, y
        q = q.replace(m.group(0), " ")

    # 相对月份:上个月 / 这个月
    if "上个月" in q or "上月" in q:
        y, mth = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        p.year_from = p.year_from or y
        p.year_to = p.year_to or y
        p.months = [mth]
        q = q.replace("上个月", " ").replace("上月", " ")
    elif "这个月" in q or "本月" in q:
        p.year_from = p.year_from or today.year
        p.year_to = p.year_to or today.year
        p.months = [today.month]
        q = q.replace("这个月", " ").replace("本月", " ")

    # 月份
    m = _RE_MONTH.search(q)
    if m and not p.months:
        raw = m.group(1)
        parsed_month: int | None = int(raw) if raw.isdigit() else CN_NUM.get(raw)
        if parsed_month and 1 <= parsed_month <= 12:
            p.months = [parsed_month]
            q = q.replace(m.group(0), " ")

    # 季节 → 月份集合(季节词保留在语义里,画面本身有季节特征)
    for season, months in _SEASONS.items():
        if season in q:
            if not p.months:
                p.months = list(months)
            break

    # 地点:已知城市名(在库,离线)
    from .geo import find_city_in_text

    city = find_city_in_text(q)
    if city:
        p.place = city
        q = re.sub(rf"在?\s*{re.escape(city)}\s*(?:拍的|拍|玩|旅游|旅行)?", " ", q)

    # 清理口水词
    for w in _STOPWORDS:
        q = q.replace(w, " ")
    p.semantic = re.sub(r"\s+", " ", q).strip() or query.strip()

    # OCR 关键词:语义文本里的连续中英文/数字片段都作为候选关键词
    p.keywords = [w for w in re.split(r"[\s,，。/]+", p.semantic) if len(w) >= 2][:5]
    p.intent = classify_intent(query, p)
    return p


_DOCUMENT_HINTS = (
    "订单", "票", "账单", "发票", "金额", "支付", "快递", "编号", "号码",
    "截图", "收据", "合同", "证件",
)


def classify_intent(query: str, parsed: ParsedQuery | None = None) -> str:
    """可解释、低延迟的查询意图分类，不依赖在线大模型。"""
    p = parsed or ParsedQuery()
    has_document = any(x in query for x in _DOCUMENT_HINTS) or bool(
        re.search(r"[A-Za-z]*\d{4,}", query)
    )
    has_metadata = bool(p.year_from or p.months or p.place)
    has_person = bool(p.person)
    active = sum((has_document, has_metadata, has_person))
    if active > 1:
        return "hybrid"
    if has_document:
        return "document"
    if has_person:
        return "person"
    if has_metadata:
        return "time_location"
    return "scene"


# ---------- Ollama 解析(可选) ----------

_SYSTEM = """你是照片搜索查询解析器。把用户口语查询解析成 JSON,字段:
semantic(字符串,描述画面内容的检索词)、keywords(字符串数组,可能出现在图中文字里的关键词)、
year_from/year_to(整数或null)、months(1-12整数数组或空)、person(人名或null)、
place(城市名或null)、screenshot(布尔或null)。
今天是{today}。只输出 JSON,不要任何其他文字。"""


def parse_ollama(query: str, host: str, model: str, today: date | None = None) -> ParsedQuery:
    import urllib.request

    today = today or date.today()
    body = json.dumps({
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": _SYSTEM.format(today=today.isoformat())},
            {"role": "user", "content": query},
        ],
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    raw = json.loads(data["message"]["content"])
    p = ParsedQuery()
    p.semantic = str(raw.get("semantic") or query)
    p.keywords = [str(k) for k in (raw.get("keywords") or [])][:5]
    for k in ("year_from", "year_to"):
        v = raw.get(k)
        setattr(p, k, int(v) if isinstance(v, (int, float)) else None)
    p.months = [int(x) for x in (raw.get("months") or []) if isinstance(x, (int, float))]
    p.person = raw.get("person") or None
    p.place = raw.get("place") or None
    p.screenshot = raw.get("screenshot") if isinstance(raw.get("screenshot"), bool) else None
    p.intent = classify_intent(query, p)
    return p


def parse(query: str, cfg, today: date | None = None) -> ParsedQuery:
    """入口:按配置选择解析器,ollama 失败自动回落规则。"""
    if cfg.query_parser == "ollama":
        try:
            return parse_ollama(query, cfg.ollama_host, cfg.ollama_model, today)
        except Exception as e:
            log.warning("ollama 解析失败,回落规则: %s", e)
    return parse_rules(query, today)
