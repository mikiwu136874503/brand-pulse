import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from content_fetcher import detect_source_type

DB_PATH = Path(__file__).parent / "data" / "brand_pulse.db"

ARTICLE_COLUMNS = (
    "id, brand_name, title, link, summary, published, source, fetched_at, "
    "category, ai_summary"
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_article_columns(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "articles")
    if "category" not in columns:
        conn.execute("ALTER TABLE articles ADD COLUMN category TEXT")
    if "ai_summary" not in columns:
        conn.execute("ALTER TABLE articles ADD COLUMN ai_summary TEXT")


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


def init_db() -> None:
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
                ai_summary TEXT
            );
            """
        )
        _ensure_brand_columns(conn)
        _ensure_article_columns(conn)


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
    return dict(row) if row else None


def list_articles(brand_name: str | None = None) -> list[dict]:
    with get_connection() as conn:
        if brand_name:
            rows = conn.execute(
                f"""
                SELECT {ARTICLE_COLUMNS}
                FROM articles
                WHERE brand_name = ?
                ORDER BY published DESC, fetched_at DESC
                """,
                (brand_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT {ARTICLE_COLUMNS}
                FROM articles
                ORDER BY published DESC, fetched_at DESC
                """
            ).fetchall()
    return [dict(row) for row in rows]


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
                     fetched_at, category, ai_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
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
