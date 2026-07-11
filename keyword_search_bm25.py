"""
keyword_search_bm25.py

Baseline keyword search using BM25 over the same PostgreSQL corpus used for
semantic search, so results can be directly compared against
semantic_search.py using the same queries and the same article set.

Setup:
    pip install rank-bm25 psycopg2-binary

Usage:
    python keyword_search_bm25.py \\
        --query "ක්‍රිකට් තරගය" \\
        --dbname sinhala_corpus \\
        --user corpus_user \\
        --password root \\
        --top_k 5
"""

import argparse

import psycopg2
from rank_bm25 import BM25Okapi


def fetch_all_articles(conn):
    """Fetch id, title, source, category, published_at, content for every article."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, title, source, category, published_at, content
            FROM articles
            WHERE content IS NOT NULL AND content <> ''
            """
        )
        rows = cursor.fetchall()

    columns = ["id", "title", "source", "category", "published_at", "content"]
    return [dict(zip(columns, row)) for row in rows]


def tokenize(text: str):
    """
    Simple whitespace tokenizer. Sinhala doesn't have the same word-boundary
    conventions as English, but whitespace splitting is a standard, reasonable
    baseline choice for BM25 comparisons in the literature on low-resource
    languages without a mature word segmenter.
    """
    return text.split()


def build_bm25_index(articles):
    corpus_tokens = [tokenize(article["content"]) for article in articles]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25


def search(bm25, articles, query: str, top_k: int):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(articles[i], scores[i]) for i in ranked_indices]


def main():
    parser = argparse.ArgumentParser(description="BM25 keyword search over the Sinhala corpus.")
    parser.add_argument("--query", type=str, required=True, help="Search query text")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dbname", type=str, default="sinhala_corpus")
    parser.add_argument("--user", type=str, default="corpus_user")
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="5432")
    args = parser.parse_args()

    conn = psycopg2.connect(
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
    )
    try:
        print("Fetching all articles from PostgreSQL...")
        articles = fetch_all_articles(conn)
        print(f"Building BM25 index over {len(articles)} articles...")
        bm25 = build_bm25_index(articles)
    finally:
        conn.close()

    results = search(bm25, articles, args.query, args.top_k)

    print(f"\nTop {len(results)} BM25 results for query: \"{args.query}\"\n")
    for rank, (article, score) in enumerate(results, start=1):
        snippet = article["content"][:150].replace("\n", " ")
        print(f"{rank}. [{score:.4f}] {article['title']}")
        print(f"   source={article['source']} category={article['category']} published={article['published_at']}")
        print(f"   {snippet}...")
        print()


if __name__ == "__main__":
    main()
