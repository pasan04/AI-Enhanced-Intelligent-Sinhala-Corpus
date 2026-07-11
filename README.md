# AI-Enhanced-Intelligent-Sinhala-Corpus

Sinhala is spoken by millions of people in
Sri Lanka, yet it remains a low-resource language for
Natural Language Processing (NLP) because relatively
few high-quality linguistic resources exist for it. The
corpora that do exist, including the well-known Sinmin
project, were built around storage schemas that only
support exact keyword matching or n-gram frequency
counts, so there is currently no easy way to look up
Sinhala documents by what they mean rather than
which words they contain. In this paper I build a
small working system that tries to close that gap:
an AI-enhanced Sinhala corpus that pairs a relational
document store with transformer-based sentence embeddings and vector similarity search. The corpus is
bootstrapped from NSina.

# AI-Enhanced Intelligent Sinhala Corpus Framework

A Sinhala news corpus system that supports both traditional keyword search (BM25) and semantic search (multilingual sentence embeddings + FAISS), built on top of [NSina](https://huggingface.co/datasets/sinhala-nlp/NSINA).

This repository accompanies the research paper *"An AI-Enhanced Intelligent Sinhala Corpus Framework for Semantic Search and Automated Natural Language Processing."*

---

## Project structure

```
sinhala_corpus/
├── load_nsina_corpus_postgres.py   # Step 1: bootstrap the corpus into PostgreSQL
├── generate_embeddings.py          # Step 2: generate LaBSE embeddings + FAISS index
├── semantic_search.py              # Step 3a: query via semantic search (CLI)
├── keyword_search_bm25.py          # Step 3b: query via BM25 keyword search (CLI)
├── app.py                          # Step 4: web demo (both methods side by side)
├── templates/
│   └── index.html                  # frontend for app.py
├── sinhala_corpus_paper.tex        # the paper (compile with XeLaTeX)
├── demo_interface.png              # screenshot used as a figure in the paper
├── corpus.index                    # generated FAISS index (created by step 2)
└── corpus_ids.json                 # generated FAISS id mapping (created by step 2)
```

---

## Prerequisites

- Python 3.10+
- PostgreSQL installed and running locally
- A free [Hugging Face](https://huggingface.co) account (NSina is a gated dataset)
- ~2 GB free disk space (NSina's raw file is ~1.9 GB, cached after first download)

---

## 1. Environment setup

```bash
python3 -m venv myvenv
source myvenv/bin/activate

pip install datasets pandas psycopg2-binary huggingface_hub \
            sentence-transformers faiss-cpu rank-bm25 flask numpy
```

## 2. PostgreSQL setup

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

sudo -i -u postgres
psql
```

Inside `psql`:

```sql
CREATE DATABASE sinhala_corpus;
CREATE USER corpus_user WITH ENCRYPTED PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE sinhala_corpus TO corpus_user;
\c sinhala_corpus
GRANT ALL ON SCHEMA public TO corpus_user;
GRANT CREATE ON SCHEMA public TO corpus_user;
\q
exit
```

## 3. Hugging Face access to NSina

NSina is a gated dataset — you must accept its terms before downloading it.

1. Visit [huggingface.co/datasets/sinhala-nlp/NSINA](https://huggingface.co/datasets/sinhala-nlp/NSINA) while logged in, and accept the access conditions on the page.
2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — use the **Read** preset (not Fine-grained, which excludes gated-repo access by default).
3. Authenticate:

```bash
hf auth login
```

Paste the token when prompted.

---

## Step 1 — Load the corpus into PostgreSQL

`load_nsina_corpus_postgres.py` downloads NSina (first run only, then cached), draws a **random** sample of articles (random sampling matters — see script docstring for why), and inserts them into PostgreSQL.

```bash
python load_nsina_corpus_postgres.py \
    --sample_size 1000 \
    --dbname sinhala_corpus \
    --user corpus_user \
    --password your_password_here
```

**Options:**
| Flag | Default | Description |
|---|---|---|
| `--sample_size` | 1000 | Number of articles to randomly sample from NSina |
| `--dbname` | sinhala_corpus | PostgreSQL database name |
| `--user` | corpus_user | PostgreSQL user |
| `--password` | *(required)* | PostgreSQL password |
| `--host` | localhost | PostgreSQL host |
| `--port` | 5432 | PostgreSQL port |

This creates an `articles` table with columns: `id`, `title`, `content`, `source`, `category`, `published_at`, `ingested_at`, `embedding`.

---

## Step 2 — Generate embeddings and build the FAISS index

`generate_embeddings.py` reads every article without an embedding yet, encodes it with **LaBSE** (a multilingual sentence transformer that supports Sinhala), stores the vector back in PostgreSQL, and builds a FAISS index.

```bash
python generate_embeddings.py \
    --dbname sinhala_corpus \
    --user corpus_user \
    --password your_password_here
```

**Options:**
| Flag | Default | Description |
|---|---|---|
| `--batch_size` | 32 | Encoding batch size |
| `--faiss_index_path` | corpus.index | Output path for the FAISS index |
| `--id_map_path` | corpus_ids.json | Output path for the FAISS-position-to-article-ID mapping |

Note: the LaBSE model (~1.9 GB) downloads once on first run and is cached afterward. Encoding speed is roughly 10 seconds per batch of 32 articles on CPU — budget time accordingly for larger samples.

---

## Step 3a — Query via semantic search (command line)

```bash
python semantic_search.py \
    --query "ක්‍රිකට් තරගය" \
    --dbname sinhala_corpus \
    --user corpus_user \
    --password your_password_here \
    --top_k 5
```

Embeds the query with LaBSE, searches the FAISS index for the nearest article vectors, and prints the top-k results with similarity scores.

## Step 3b — Query via keyword search (command line)

```bash
python keyword_search_bm25.py \
    --query "ක්‍රිකට් තරගය" \
    --dbname sinhala_corpus \
    --user corpus_user \
    --password your_password_here \
    --top_k 5
```

Builds a BM25 index over all article content (whitespace-tokenized) and prints the top-k results ranked by BM25 score.

---

## Step 4 — Run the web demo (both methods side by side)

```bash
python app.py \
    --dbname sinhala_corpus \
    --user corpus_user \
    --password your_password_here
```

Then open **http://localhost:5000**. Type a query and see semantic search and BM25 results rendered in two columns for direct comparison.

On startup, the app loads the LaBSE model, the FAISS index, and builds the BM25 index once — this takes a little time, but only happens once per server start, not per search.

**Options:**
| Flag | Default | Description |
|---|---|---|
| `--faiss_index_path` | corpus.index | Path to the FAISS index built in Step 2 |
| `--id_map_path` | corpus_ids.json | Path to the ID mapping built in Step 2 |
| `--web_port` | 5000 | Port to serve the web app on |

---

## Compiling the paper

The paper contains Sinhala text and TikZ diagrams, so it **must** be compiled with **XeLaTeX**, not pdfLaTeX (which cannot render Sinhala script at all).

**Locally:**
```bash
xelatex sinhala_corpus_paper.tex
```

**On Overleaf:**
1. Upload `sinhala_corpus_paper.tex` **and** `demo_interface.png` into the same project folder.
2. Open the project menu → Compiler → select **XeLaTeX** (it defaults to pdfLaTeX, which will fail with a fontspec error).
3. Recompile.

---

## Typical end-to-end run

```bash
source myvenv/bin/activate

python load_nsina_corpus_postgres.py --sample_size 1000 --dbname sinhala_corpus --user corpus_user --password your_password_here
python generate_embeddings.py --dbname sinhala_corpus --user corpus_user --password your_password_here
python app.py --dbname sinhala_corpus --user corpus_user --password your_password_here
```

Then open `http://localhost:5000` and start searching.

---

## Known limitations

- BM25 baseline uses whitespace tokenization, since no mature, widely adopted Sinhala word segmenter currently exists.
- The current sample (998 articles after cleaning) is a development-scale sample, not the full NSina corpus (506,932 articles) — intended for validating the pipeline before scaling up.
- Quantitative evaluation (Precision@k, Recall@k, MRR) using paraphrased queries is planned but not yet implemented in this repository.
