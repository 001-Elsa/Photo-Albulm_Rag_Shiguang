import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.query_parser import parse_rules

TODAY = date(2026, 7, 26)


def test_relative_year_and_semantic():
    p = parse_rules("去年在海边拍的日落", today=TODAY)
    assert p.year_from == 2025 and p.year_to == 2025
    assert "海边" in p.semantic and "日落" in p.semantic
    assert "去年" not in p.semantic


def test_absolute_year_month():
    p = parse_rules("2024年3月的樱花", today=TODAY)
    assert p.year_from == 2024 and p.months == [3]
    assert "樱花" in p.semantic


def test_cn_month():
    p = parse_rules("十月的红叶", today=TODAY)
    assert p.months == [10]


def test_season_months():
    p = parse_rules("2024年冬天的雪", today=TODAY)
    assert p.year_from == 2024 and sorted(p.months) == [1, 2, 12]
    assert "冬天" in p.semantic  # 季节词保留在语义里


def test_last_month():
    p = parse_rules("上个月的聚餐", today=TODAY)
    assert p.year_from == 2026 and p.months == [6]


def test_place_extraction():
    p = parse_rules("在杭州拍的西湖", today=TODAY)
    assert p.place == "杭州"
    assert "西湖" in p.semantic and "杭州" not in p.semantic


def test_screenshot():
    p = parse_rules("那张高铁票截图", today=TODAY)
    assert p.screenshot is True
    assert "高铁票" in p.semantic
    assert any("高铁" in k for k in p.keywords)


def test_person():
    p = parse_rules("和妈妈的合影", today=TODAY)
    assert p.person == "妈妈"
    assert "合影" in p.semantic


def test_person_variant():
    p = parse_rules("跟小李一起", today=TODAY)
    assert p.person == "小李"


def test_plain_semantic_fallback():
    p = parse_rules("火锅", today=TODAY)
    assert p.semantic == "火锅"
    assert p.year_from is None and p.screenshot is None


def test_stopwords_stripped():
    p = parse_rules("帮我找一张雪山的照片", today=TODAY)
    assert "帮我找" not in p.semantic and "照片" not in p.semantic
    assert "雪山" in p.semantic
