"""v1.0:数据库 schema 版本迁移框架。

企业环境里"删库重建"不可接受——所有 schema 变更走有序迁移:
- kv 表记录 schema_version
- MIGRATIONS 按版本号排列,启动时把落后的依次补齐,幂等可重入
- 新变更 = 在列表末尾追加一个 (版本号, SQL或函数),禁止修改历史项
"""
from __future__ import annotations

import logging

log = logging.getLogger("shiguang.migrations")

# 001:v0.8/v0.9 的基础表(全部 IF NOT EXISTS,对老库幂等)
_M001_BASE = """
CREATE TABLE IF NOT EXISTS photos (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    sha1        TEXT,
    size        INTEGER,
    mtime       REAL,
    width       INTEGER,
    height      INTEGER,
    taken_at    TEXT,
    year        INTEGER,
    month       INTEGER,
    lat         REAL,
    lon         REAL,
    place       TEXT,
    camera      TEXT,
    is_screenshot INTEGER DEFAULT 0,
    phash       TEXT,
    thumb       TEXT,
    status      TEXT DEFAULT 'scanned',
    embedded    INTEGER DEFAULT 0,
    ocr_done    INTEGER DEFAULT 0,
    faces_done  INTEGER DEFAULT 0,
    added_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_photos_year ON photos(year, month);
CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);

CREATE TABLE IF NOT EXISTS embeddings (
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    dim      INTEGER,
    vec      BLOB
);

CREATE TABLE IF NOT EXISTS ocr_text (
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    text     TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
    text, content='ocr_text', content_rowid='photo_id',
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS ocr_ai AFTER INSERT ON ocr_text BEGIN
    INSERT INTO ocr_fts(rowid, text) VALUES (new.photo_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS ocr_ad AFTER DELETE ON ocr_text BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, text) VALUES ('delete', old.photo_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS ocr_au AFTER UPDATE ON ocr_text BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, text) VALUES ('delete', old.photo_id, old.text);
    INSERT INTO ocr_fts(rowid, text) VALUES (new.photo_id, new.text);
END;

CREATE TABLE IF NOT EXISTS persons (
    id    INTEGER PRIMARY KEY,
    name  TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS faces (
    id        INTEGER PRIMARY KEY,
    photo_id  INTEGER REFERENCES photos(id) ON DELETE CASCADE,
    person_id INTEGER REFERENCES persons(id),
    bbox      TEXT,
    vec       BLOB
);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);

CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""

# 002:v1.0 企业化——用户 / 审计
_M002_AUTH = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    pwd_hash   TEXT NOT NULL,      -- pbkdf2$iterations$salt_hex$hash_hex
    role       TEXT NOT NULL DEFAULT 'viewer',  -- admin | viewer
    created_at REAL,
    disabled   INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY,
    ts      REAL,
    user    TEXT,
    action  TEXT,       -- login / search / view / download / index / settings
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""

MIGRATIONS: list[tuple[int, str]] = [
    (1, _M001_BASE),
    (2, _M002_AUTH),
]

LATEST = max(v for v, _ in MIGRATIONS)


def migrate(conn) -> int:
    """把连接指向的库迁移到最新版本,返回最终版本号。"""
    conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
    row = conn.execute("SELECT v FROM kv WHERE k='schema_version'").fetchone()
    current = int(row[0]) if row else 0
    for version, sql in MIGRATIONS:
        if version <= current:
            continue
        log.info("迁移 schema v%d → v%d", current, version)
        conn.executescript(sql)
        conn.execute(
            "INSERT OR REPLACE INTO kv (k, v) VALUES ('schema_version', ?)",
            (str(version),),
        )
        conn.commit()
        current = version
    return current
