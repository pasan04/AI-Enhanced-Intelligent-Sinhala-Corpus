"""
semantic_search.py

Takes a text query, embeds it with the same multilingual model used to build
the corpus, searches the FAISS index for the most similar articles, and
retrieves their full metadata (title, source, category) from PostgreSQL.

Setup:
    pip install sentence-transformers faiss-cpu numpy psycopg2-binary

Usage:
    python semantic_search.py \\
        --query "ශ්‍රී ලංකා ක්‍රිකට් තරගය" \\
        --dbname sinhala_corpus \\
        --user corpus_user \\
        --password root \\
        --top_k 5
"""

import argparse
import json

import numpy as np
import psycopg2
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/LaBSE"


def load_index_and_ids(index_path: str, id_map_path: str):
    index = faiss.read_index(index_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return index, ids


def embed_query(model, query: str) -> np.ndarray:
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector.astype("float32")


def search(index, query_vector, top_k: int):
    """Returns (scores, positions) for the top_k nearest vectors."""
    scores, positions = index.search(query_vector, top_k)
    return scores[0], positions[0]


def fetch_articles_by_ids(conn, article_ids):
    """Fetch title, source, category, content for a list of article ids,
    preserving the order given."""
    if not article_ids:
        return {}
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, title, source, category, content, published_at
            FROM articles
            WHERE id = ANY(%s)
            """,
            (article_ids,),
        )
        rows = cursor.fetchall()

    columns = ["id", "title", "source", "category", "content", "published_at"]
    return {row[0]: dict(zip(columns, row)) for row in rows}


def main():
    parser = argparse.ArgumentParser(description="Semantic search over the Sinhala corpus.")
    parser.add_argument("--query", type=str, required=True, help="Search query text (Sinhala or English)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--faiss_index_path", type=str, default="corpus.index")
    parser.add_argument("--id_map_path", type=str, default="corpus_ids.json")
    parser.add_argument("--dbname", type=str, default="sinhala_corpus")
    parser.add_argument("--user", type=str, default="corpus_user")
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="5432")
    args = parser.parse_args()

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index and id mapping...")
    index, ids = load_index_and_ids(args.faiss_index_path, args.id_map_path)

    query_vector = embed_query(model, args.query)
    scores, positions = search(index, query_vector, args.top_k)

    # Map FAISS row positions back to article ids, skipping any -1 (no match)
    matched_ids = [ids[pos] for pos in positions if pos != -1]

    conn = psycopg2.connect(
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
    )
    try:
        articles = fetch_articles_by_ids(conn, matched_ids)
    finally:
        conn.close()

    print(f"\nTop {len(matched_ids)} results for query: \"{args.query}\"\n")
    for rank, (article_id, score) in enumerate(zip(matched_ids, scores), start=1):
        article = articles.get(article_id)
        if not article:
            continue
        snippet = article["content"][:150].replace("\n", " ")
        print(f"{rank}. [{score:.4f}] {article['title']}")
        print(f"   source={article['source']} category={article['category']} published={article['published_at']}")
        print(f"   {snippet}...")
        print()


if __name__ == "__main__":
    main()
