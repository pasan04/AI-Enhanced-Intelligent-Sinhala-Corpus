"""
app.py

Flask web app exposing a single search box that runs the same query through
both the semantic search pipeline (LaBSE + FAISS) and the keyword baseline
(BM25), returning both result sets side by side.

Setup:
    pip install flask sentence-transformers faiss-cpu rank-bm25 psycopg2-binary numpy

Usage:
    python app.py \\
        --dbname sinhala_corpus \\
        --user corpus_user \\
        --password root

Then open http://localhost:5000 in your browser.
"""

import argparse
import json

import numpy as np
import psycopg2
import faiss
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

MODEL_NAME = "sentence-transformers/LaBSE"

app = Flask(__name__)

# Populated once at startup in main(), then reused across requests.
STATE = {
    "model": None,
    "faiss_index": None,
    "faiss_ids": None,
    "bm25": None,
    "bm25_articles": None,
    "db_config": None,
}


def get_connection():
    return psycopg2.connect(**STATE["db_config"])


def load_semantic_index(faiss_index_path: str, id_map_path: str):
    index = faiss.read_index(faiss_index_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return index, ids


def load_bm25_index(conn):
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
    articles = [dict(zip(columns, row)) for row in rows]
    corpus_tokens = [article["content"].split() for article in articles]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, articles


def fetch_articles_by_ids(conn, article_ids):
    if not article_ids:
        return {}
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, title, source, category, published_at, content
            FROM articles
            WHERE id = ANY(%s)
            """,
            (article_ids,),
        )
        rows = cursor.fetchall()
    columns = ["id", "title", "source", "category", "published_at", "content"]
    return {row[0]: dict(zip(columns, row)) for row in rows}


def semantic_search(query: str, top_k: int):
    vector = STATE["model"].encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, positions = STATE["faiss_index"].search(vector, top_k)
    scores, positions = scores[0], positions[0]

    matched_ids = [STATE["faiss_ids"][pos] for pos in positions if pos != -1]

    conn = get_connection()
    try:
        articles = fetch_articles_by_ids(conn, matched_ids)
    finally:
        conn.close()

    results = []
    for article_id, score in zip(matched_ids, scores):
        article = articles.get(article_id)
        if not article:
            continue
        results.append({
            "title": article["title"],
            "source": article["source"],
            "category": article["category"],
            "published_at": article["published_at"],
            "snippet": article["content"][:200],
            "score": round(float(score), 4),
        })
    return results


def keyword_search(query: str, top_k: int):
    query_tokens = query.split()
    scores = STATE["bm25"].get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in ranked_indices:
        article = STATE["bm25_articles"][i]
        results.append({
            "title": article["title"],
            "source": article["source"],
            "category": article["category"],
            "published_at": article["published_at"],
            "snippet": article["content"][:200],
            "score": round(float(scores[i]), 4),
        })
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 5))

    if not query:
        return jsonify({"error": "Query is empty"}), 400

    semantic_results = semantic_search(query, top_k)
    keyword_results = keyword_search(query, top_k)

    return jsonify({
        "query": query,
        "semantic": semantic_results,
        "keyword": keyword_results,
    })


def main():
    parser = argparse.ArgumentParser(description="Run the Sinhala corpus search demo web app.")
    parser.add_argument("--dbname", type=str, default="sinhala_corpus")
    parser.add_argument("--user", type=str, default="corpus_user")
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="5432")
    parser.add_argument("--faiss_index_path", type=str, default="corpus.index")
    parser.add_argument("--id_map_path", type=str, default="corpus_ids.json")
    parser.add_argument("--web_port", type=int, default=5000)
    args = parser.parse_args()

    STATE["db_config"] = {
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password,
        "host": args.host,
        "port": args.port,
    }

    print(f"Loading model '{MODEL_NAME}'...")
    STATE["model"] = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index...")
    STATE["faiss_index"], STATE["faiss_ids"] = load_semantic_index(
        args.faiss_index_path, args.id_map_path
    )

    print("Building BM25 index...")
    conn = get_connection()
    try:
        STATE["bm25"], STATE["bm25_articles"] = load_bm25_index(conn)
    finally:
        conn.close()

    print(f"Ready. Open http://localhost:{args.web_port} in your browser.")
    app.run(host="0.0.0.0", port=args.web_port, debug=False)


if __name__ == "__main__":
    main()
