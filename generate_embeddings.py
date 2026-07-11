"""
generate_embeddings.py

Reads articles from the PostgreSQL corpus table, generates dense vector
embeddings for each article's content using a multilingual sentence
transformer (supports Sinhala), stores the embeddings back into PostgreSQL,
and builds a FAISS index for fast semantic similarity search.

Setup:
    pip install sentence-transformers faiss-cpu numpy psycopg2-binary

Usage:
    python generate_embeddings.py \\
        --dbname sinhala_corpus \\
        --user corpus_user \\
        --password root \\
        --faiss_index_path corpus.index \\
        --id_map_path corpus_ids.json
"""

import argparse
import json

import numpy as np
import psycopg2
import faiss
from sentence_transformers import SentenceTransformer

# LaBSE supports 100+ languages, including Sinhala, and produces
# 768-dimensional embeddings. paraphrase-multilingual-mpnet-base-v2 is a
# lighter alternative if LaBSE is too slow on your machine.
MODEL_NAME = "sentence-transformers/LaBSE"


def fetch_articles(conn):
    """Fetch id and content for every article that doesn't have an embedding yet."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, content
            FROM articles
            WHERE embedding IS NULL
              AND content IS NOT NULL
              AND content <> ''
            """
        )
        return cursor.fetchall()


def embed_articles(model, articles, batch_size: int = 32):
    """
    Generate embeddings for a list of (id, content) tuples.
    Returns (ids, embeddings) where embeddings is a numpy array of shape
    (n_articles, embedding_dim).
    """
    ids = [row[0] for row in articles]
    texts = [row[1] for row in articles]

    print(f"Encoding {len(texts)} articles in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return ids, embeddings.astype("float32")


def store_embeddings_in_postgres(conn, ids, embeddings):
    """Write each embedding back into the articles table as raw bytes."""
    with conn.cursor() as cursor:
        for article_id, vector in zip(ids, embeddings):
            cursor.execute(
                "UPDATE articles SET embedding = %s WHERE id = %s",
                (vector.tobytes(), article_id),
            )
    conn.commit()
    print(f"Stored {len(ids)} embeddings in PostgreSQL.")


def build_faiss_index(embeddings, dim: int):
    """Build a flat inner-product FAISS index (cosine similarity, since
    embeddings are normalized)."""
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for articles and build a FAISS index."
    )
    parser.add_argument("--dbname", type=str, default="sinhala_corpus")
    parser.add_argument("--user", type=str, default="corpus_user")
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="5432")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--faiss_index_path", type=str, default="corpus.index")
    parser.add_argument("--id_map_path", type=str, default="corpus_ids.json")
    args = parser.parse_args()

    conn = psycopg2.connect(
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
    )

    try:
        articles = fetch_articles(conn)
        if not articles:
            print("No articles found needing embeddings. Nothing to do.")
            return

        print(f"Loading model '{MODEL_NAME}' (this may take a minute on first run)...")
        model = SentenceTransformer(MODEL_NAME)

        ids, embeddings = embed_articles(model, articles, args.batch_size)
        store_embeddings_in_postgres(conn, ids, embeddings)

        dim = embeddings.shape[1]
        index = build_faiss_index(embeddings, dim)

        faiss.write_index(index, args.faiss_index_path)
        with open(args.id_map_path, "w", encoding="utf-8") as f:
            json.dump(ids, f)

        print(f"FAISS index saved to {args.faiss_index_path} ({index.ntotal} vectors, dim={dim}).")
        print(f"ID mapping saved to {args.id_map_path}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
