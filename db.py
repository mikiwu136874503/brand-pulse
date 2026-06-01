import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from content_fetcher import detect_source_type

DB_PATH = Path(__file__).parent / "data" / "brand_pulse.db"

ARTICLE_COLUMNS = (
    "id, brand_name, title, link, summary, published, source, fetched_at, "
    "category, ai_summary, is_favorite"
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_article_columns(conn: sqlite3.Connection) -> bool:
    """补齐 articles 表字段，返回 is_favorite 字段是否可用。"""
    columns = _table_columns(conn, "articles")
    if "category" not in columns:
        conn.execute("ALTER TABLE articles ADD COLUMN category TEXT")
    if "ai_summary" not in columns:
        conn.execute("ALTER TABLE articles ADD COLUMN ai_summary TEXT")
    if "is_favorite" not in columns:
        try:
            conn.execute(
                "ALTER TABLE articles ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            # 字段已存在时跳过（兼容并发迁移或重复执行）
            pass
    conn.commit()
    return "is_favorite" in _table_columns(conn, "articles")


def init_db() -> dict[str, bool | int]:
    """初始化数据库，返回 schema 状态供前端提示。"""
    DB_PATH.parent.mkdir(exist_ok=True)
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS brands (
                id TEXT PRIMARY KEY,
                brand_name TEXT NOT NULL UNIQUE,
                rss_url TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'rss'
            );

            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                brand_name TEXT NOT NULL,
                title TEXT NOT NULL,
                link TEXT NOT NULL UNIQUE,
                summary TEXT,
                published TEXT,
                source TEXT,
                fetched_at TEXT NOT NULL,
                category TEXT,
                ai_summary TEXT,
                is_favorite INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_articles_brand_name
                ON articles(brand_name);
            CREATE INDEX IF NOT EXISTS idx_articles_published
                ON articles(published);
            CREATE INDEX IF NOT EXISTS idx_articles_category
                ON articles(category);
            CREATE INDEX IF NOT EXISTS idx_articles_is_favorite
                ON articles(is_favorite);
            """
        )
        _ensure_brand_columns(conn)
        has_favorite = _ensure_article_columns(conn)
        favorite_count = 0
        if has_favorite:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM articles WHERE COALESCE(is_favorite, 0) = 1"
            ).fetchone()
            favorite_count = int(row["cnt"]) if row else 0
        conn.commit()
    return {
        "has_is_favorite_column": has_favorite,
        "favorite_count": favorite_count,
    }


def _normalize_article_row(row: sqlite3.Row) -> dict:
    """统一文章字段，确保 is_favorite 为整数 0/1。"""
    article = dict(row)
    article["is_favorite"] = 1 if int(article.get("is_favorite") or 0) else 0
    return article


def _ensure_brand_columns(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "brands")
    if "source_type" not in columns:
        conn.execute("ALTER TABLE brands ADD COLUMN source_type TEXT")
        rows = conn.execute("SELECT id, rss_url FROM brands").fetchall()
        for row in rows:
            source_type = detect_source_type(row["rss_url"] or "")
            conn.execute(
                "UPDATE brands SET source_type = ? WHERE id = ?",
                (source_type, row["id"]),
            )


def list_brands() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, brand_name, rss_url, source_type FROM brands ORDER BY brand_name"
        ).fetchall()
    return [dict(row) for row in rows]


def brand_name_exists(brand_name: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM brands WHERE brand_name = ?",
            (brand_name.strip(),),
        ).fetchone()
    return row is not None


def add_brand(brand_name: str, source_url: str, source_type: str | None = None) -> str:
    brand_id = uuid4().hex
    url = source_url.strip()
    resolved_type = (source_type or detect_source_type(url)).lower()
    if resolved_type not in ("rss", "web"):
        resolved_type = detect_source_type(url)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO brands (id, brand_name, rss_url, source_type)
            VALUES (?, ?, ?, ?)
            """,
            (brand_id, brand_name.strip(), url, resolved_type),
        )
    return brand_id


def delete_brand(brand_id: str) -> str | None:
    """级联删除品牌及其文章，返回被删除的品牌名称。"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT brand_name FROM brands WHERE id = ?",
            (brand_id,),
        ).fetchone()
        if not row:
            return None
        brand_name = row["brand_name"]
        conn.execute("DELETE FROM articles WHERE brand_name = ?", (brand_name,))
        conn.execute("DELETE FROM brands WHERE id = ?", (brand_id,))
        return brand_name


def get_article_by_id(article_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT {ARTICLE_COLUMNS} FROM articles WHERE id = ?",
            (article_id,),
        ).fetchone()
    return _normalize_article_row(row) if row else None


def get_articles_by_ids(article_ids: list[str | int]) -> list[dict]:
    """
    根据 ID 列表批量返回文章完整字段。
    - 空列表返回 []
    - 不存在的 ID 被忽略
    - 结果顺序与传入 ID 顺序一致（去重后）
    """
    if not article_ids:
        return []

    normalized_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in article_ids:
        article_id = str(raw_id).strip()
        if not article_id or article_id in seen:
            continue
        seen.add(article_id)
        normalized_ids.append(article_id)

    if not normalized_ids:
        return []

    try:
        placeholders = ",".join("?" * len(normalized_ids))
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT {ARTICLE_COLUMNS} FROM articles WHERE id IN ({placeholders})",
                normalized_ids,
            ).fetchall()
    except sqlite3.Error:
        return []

    by_id: dict[str, dict] = {}
    for row in rows:
        article = _normalize_article_row(row)
        by_id[article["id"]] = article
    return [by_id[aid] for aid in normalized_ids if aid in by_id]


def _published_range_conditions(
    published_start: str | None,
    published_end: str | None,
) -> tuple[list[str], list]:
    """构建 published 时间范围 SQL 条件（ISO 格式字符串比较）。"""
    if not published_start or not published_end:
        return [], []
    return (
        [
            "published IS NOT NULL",
            "TRIM(published) != ''",
            "published GLOB '????-??-??*'",
            "published >= ?",
            "published <= ?",
        ],
        [published_start, published_end],
    )


def list_articles(
    brand_name: str | None = None,
    favorites_only: bool = False,
    published_start: str | None = None,
    published_end: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """列出文章，可按品牌、主题、收藏状态与发布时间范围筛选。"""
    conditions: list[str] = []
    params: list = []
    if brand_name:
        conditions.append("brand_name = ?")
        params.append(brand_name)
    if category == "未分类":
        conditions.append("(category IS NULL OR TRIM(category) = '')")
    elif category:
        conditions.append("category = ?")
        params.append(category)
    if favorites_only:
        conditions.append("COALESCE(is_favorite, 0) = 1")
    pub_conds, pub_params = _published_range_conditions(published_start, published_end)
    conditions.extend(pub_conds)
    params.extend(pub_params)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT {ARTICLE_COLUMNS}
            FROM articles
            {where_clause}
            ORDER BY published DESC, fetched_at DESC
            """,
            params,
        ).fetchall()
    return [_normalize_article_row(row) for row in rows]


def count_articles_by_brand() -> dict[str, int]:
    """按品牌统计文章数量，避免在 UI 循环中重复查询。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT brand_name, COUNT(*) AS cnt FROM articles GROUP BY brand_name"
        ).fetchall()
    return {row["brand_name"]: int(row["cnt"]) for row in rows}


def count_uncategorized_articles(brand_name: str | None = None) -> int:
    conditions = ["(category IS NULL OR TRIM(category) = '')"]
    params: list = []
    if brand_name:
        conditions.append("brand_name = ?")
        params.append(brand_name)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM articles WHERE {where_clause}",
            params,
        ).fetchone()
    return int(row["cnt"]) if row else 0


def count_articles_without_ai_summary(brand_name: str | None = None) -> int:
    conditions = ["(ai_summary IS NULL OR TRIM(ai_summary) = '')"]
    params: list = []
    if brand_name:
        conditions.append("brand_name = ?")
        params.append(brand_name)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM articles WHERE {where_clause}",
            params,
        ).fetchone()
    return int(row["cnt"]) if row else 0


def list_uncategorized_articles(brand_name: str | None = None) -> list[dict]:
    with get_connection() as conn:
        if brand_name:
            rows = conn.execute(
                f"""
                SELECT {ARTICLE_COLUMNS}
                FROM articles
                WHERE brand_name = ?
                  AND (category IS NULL OR TRIM(category) = '')
                ORDER BY published DESC, fetched_at DESC
                """,
                (brand_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT {ARTICLE_COLUMNS}
                FROM articles
                WHERE category IS NULL OR TRIM(category) = ''
                ORDER BY published DESC, fetched_at DESC
                """
            ).fetchall()
    return [dict(row) for row in rows]


def list_articles_without_ai_summary(brand_name: str | None = None) -> list[dict]:
    with get_connection() as conn:
        condition = "ai_summary IS NULL OR TRIM(ai_summary) = ''"
        if brand_name:
            rows = conn.execute(
                f"""
                SELECT {ARTICLE_COLUMNS}
                FROM articles
                WHERE brand_name = ? AND ({condition})
                ORDER BY published DESC, fetched_at DESC
                """,
                (brand_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT {ARTICLE_COLUMNS}
                FROM articles
                WHERE {condition}
                ORDER BY published DESC, fetched_at DESC
                """
            ).fetchall()
    return [dict(row) for row in rows]


def update_article_category(article_id: str, category: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE articles SET category = ? WHERE id = ?",
            (category, article_id),
        )


def update_article_ai_summary(article_id: str, ai_summary: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE articles SET ai_summary = ? WHERE id = ?",
            (ai_summary, article_id),
        )


def toggle_article_favorite(article_id: str) -> bool | None:
    """
    切换文章收藏状态。
    返回切换后的收藏状态（True=已收藏）；文章不存在时返回 None。
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(is_favorite, 0) AS is_favorite FROM articles WHERE id = ?",
            (article_id,),
        ).fetchone()
        if not row:
            return None
        new_value = 0 if int(row["is_favorite"]) else 1
        conn.execute(
            "UPDATE articles SET is_favorite = ? WHERE id = ?",
            (new_value, article_id),
        )
        conn.commit()
    return bool(new_value)


def count_favorite_articles(brand_name: str | None = None) -> int:
    """统计收藏文章数量，可与品牌筛选组合。"""
    conditions = ["COALESCE(is_favorite, 0) = 1"]
    params: list = []
    if brand_name:
        conditions.append("brand_name = ?")
        params.append(brand_name)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM articles WHERE {where_clause}",
            params,
        ).fetchone()
    return int(row["cnt"]) if row else 0


def save_articles(articles: list[dict]) -> tuple[int, int]:
    """保存文章，按 link 去重。返回 (新增数量, 跳过数量)。"""
    fetched_at = datetime.now().isoformat(timespec="seconds")
    new_count = 0
    skipped_count = 0
    with get_connection() as conn:
        for article in articles:
            exists = conn.execute(
                "SELECT 1 FROM articles WHERE link = ?",
                (article["link"],),
            ).fetchone()
            if exists:
                skipped_count += 1
                continue
            conn.execute(
                """
                INSERT INTO articles
                    (id, brand_name, title, link, summary, published, source,
                     fetched_at, category, ai_summary, is_favorite)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0)
                """,
                (
                    uuid4().hex,
                    article["brand_name"],
                    article["title"],
                    article["link"],
                    article.get("summary", ""),
                    article.get("published", ""),
                    article.get("source", ""),
                    fetched_at,
                ),
            )
            new_count += 1
    return (new_count, skipped_count)
