"""
load_nsina_corpus_postgres.py

Bootstraps a working corpus for the AI-enhanced Sinhala corpus project
by pulling a sample of articles from the NSina dataset (Hettiarachchi et al.,
2024) and storing them in a PostgreSQL database with the metadata fields
needed for downstream annotation and semantic search.

Setup:
    pip install huggingface_hub pandas psycopg2-binary

    # Create the database and user first (see PostgreSQL install steps):
    #   sudo -i -u postgres
    #   psql
    #   CREATE DATABASE sinhala_corpus;
    #   CREATE USER corpus_user WITH ENCRYPTED PASSWORD 'your_password_here';
    #   GRANT ALL PRIVILEGES ON DATABASE sinhala_corpus TO corpus_user;

Usage:
    python load_nsina_corpus_postgres.py \\
        --sample_size 1000 \\
        --dbname sinhala_corpus \\
        --user corpus_user \\
        --password your_password_here \\
        --host localhost \\
        --port 5432
"""

import argparse
import json
import random
from datetime import datetime, timezone

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from huggingface_hub import hf_hub_download


def create_schema(conn) -> None:
    """Create the articles table if it doesn't already exist."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id            TEXT PRIMARY KEY,
                title         TEXT,
                content       TEXT,
                source        TEXT,
                category      TEXT,
                published_at  TEXT,
                ingested_at   TIMESTAMP,
                embedding     BYTEA
            )
            """
        )
    conn.commit()


def load_nsina_sample(sample_size: int, seed: int = 42) -> pd.DataFrame:
    """
    Load a sample of articles from the NSina dataset on Hugging Face.

    NSina is published as one large concatenated JSON file rather than
    line-delimited JSON. The `datasets` library's streaming JSON reader has a
    known bug on files like this (it keeps doubling its internal read buffer
    looking for a complete record and eventually overflows a 32-bit int), so
    we bypass `load_dataset(..., streaming=True)` entirely and instead
    download the raw file once via huggingface_hub and parse it directly.

    Important: NSina was built by concatenating separate per-source JSON
    files, so the first N records are NOT a representative sample (e.g. the
    first ~1000 records are almost entirely one source and one category).
    We take a random sample across the whole file instead.
    """
    print("Downloading NSINA.json from the Hugging Face Hub (one-time download)...")
    file_path = hf_hub_download(
        repo_id="sinhala-nlp/NSINA",
        filename="NSINA.json",
        repo_type="dataset",
    )

    print("Reading full dataset (this stays cached locally after the first run)...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Full dataset has {len(data)} articles. Drawing a random sample of {sample_size}...")
    random.seed(seed)
    rows = random.sample(data, min(sample_size, len(data)))

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} articles.")
    print("Available columns:", list(df.columns))
    print("Category distribution in sample:")
    print(df["Category"].value_counts())
    print("Source distribution in sample:")
    print(df["Source"].value_counts())
    return df


def normalize_record(record: dict) -> dict:
    """
    Map an NSina record to our schema.

    NSina's published JSON structure uses these field names (per the
    dataset card's example record):
        Source, Timestamp, Headline, News Content, URL, Category, Parent URL

    We fall back to a few alternate keys just in case a future version of
    the dataset renames fields.
    """
    return {
        "id": str(record.get("URL", record.get("url", ""))),
        "title": record.get("Headline", record.get("title", "")),
        "content": record.get("News Content", record.get("content", "")),
        "source": record.get("Source", record.get("source", "")),
        "category": record.get("Category", record.get("category", "")),
        "published_at": str(record.get("Timestamp", record.get("date", ""))),
        "ingested_at": datetime.now(timezone.utc),
    }


def insert_articles(conn, df: pd.DataFrame) -> None:
    """Normalize and bulk-insert articles into the PostgreSQL corpus table."""
    records = []

    for _, row in df.iterrows():
        record = normalize_record(row.to_dict())

        # Skip records with no usable text
        if not record["content"]:
            continue

        records.append(
            (
                record["id"],
                record["title"],
                record["content"],
                record["source"],
                record["category"],
                record["published_at"],
                record["ingested_at"],
            )
        )

    if not records:
        print("No valid articles to insert.")
        return

    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO articles
                (id, title, content, source, category, published_at, ingested_at)
            VALUES %s
            ON CONFLICT (id) DO NOTHING
            """,
            records,
        )
    conn.commit()
    print(f"Inserted (up to) {len(records)} articles into the database.")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap corpus from NSina into PostgreSQL."
    )
    parser.add_argument(
        "--sample_size", type=int, default=1000,
        help="Number of articles to pull from NSina (default: 1000)",
    )
    parser.add_argument("--dbname", type=str, default="sinhala_corpus")
    parser.add_argument("--user", type=str, default="corpus_user")
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="5432")
    args = parser.parse_args()

    df = load_nsina_sample(args.sample_size)

    conn = psycopg2.connect(
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
    )

    try:
        create_schema(conn)
        insert_articles(conn, df)

        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
        print(f"Total articles now in '{args.dbname}': {count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
