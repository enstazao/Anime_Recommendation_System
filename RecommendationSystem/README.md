# TF-IDF Anime Recommendation System

This folder implements a simple content-based recommender for the anime
dataset. It uses TF-IDF vectors built from anime metadata and evaluates the
recommendations against the GPT-OSS annotated test set.

## Files

- `config.py`: central file paths and model settings.
- `src/data_loader.py`: loads anime metadata and the annotated test set.
- `src/preprocessing.py`: builds weighted text used by TF-IDF.
- `src/tfidf_recommender.py`: TF-IDF vectorizer and recommendation logic.
- `src/bm25f_recommender.py`: field-weighted BM25F recommender.
- `src/embedding_recommender.py`: sentence-transformer embedding recommender.
- `src/hybrid_recommender.py`: combines BM25F, TF-IDF, embeddings, and
  metadata overlap.
- `src/evaluate.py`: Precision@10 and Recall@10 evaluation.
- `src/evaluate_bm25f.py`: BM25F recommender evaluation.
- `src/evaluate_embeddings.py`: embedding recommender evaluation.
- `src/evaluate_hybrid.py`: hybrid recommender evaluation.
- `app.py`: Flask API and static app server.
- `index.html`: one-page UI for searching anime, choosing TF-IDF or embeddings,
  and viewing top 10 results.
- `reports/tfidf_test_report.md`: final test report.

## Run Evaluation

Install the recommender dependencies in your active environment:

```bash
pip install -r RecommendationSystem/requirements.txt
```

```bash
python3 RecommendationSystem/src/evaluate.py
```

Run the sentence-embedding recommender evaluation:

```bash
python3 RecommendationSystem/src/evaluate_embeddings.py
```

Run the BM25F recommender evaluation:

```bash
python3 RecommendationSystem/src/evaluate_bm25f.py
```

Run the hybrid recommender evaluation:

```bash
python3 RecommendationSystem/src/evaluate_hybrid.py
```

The first embedding run downloads `sentence-transformers/all-MiniLM-L6-v2` and
creates cached embeddings in `RecommendationSystem/models/`. Later runs reuse
that cache.

## Run App

```bash
python3 RecommendationSystem/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The app runs on Flask's default port `5000`. The UI lets the user choose the
recommendation method:

- `Hybrid`: blended model using BM25F, TF-IDF, embeddings, tag overlap, and
  genre overlap.
- `TF-IDF`: exact metadata/tag text matching baseline.
- `BM25F`: field-weighted search ranking over title, genres, tags, studios,
  format, source, and description.
- `Embeddings`: sentence-transformer semantic similarity.

If you are already inside the `RecommendationSystem` folder, run:

```bash
pip install -r requirements.txt
python app.py
```

## How It Works

The recommender combines each anime's genres, tags, studios, title, and
description into one weighted text field. Genres and tags are repeated more
often than description so they have stronger influence.

TF-IDF converts each anime into a sparse vector. For a selected anime, the
system computes cosine similarity between that anime vector and all other anime
vectors, then returns the top 10 highest-scoring anime.

The embedding model uses `sentence-transformers/all-MiniLM-L6-v2` to turn the
same anime metadata text into dense semantic vectors. The frontend sends
`model=hybrid`, `model=tfidf`, `model=bm25f`, or `model=embedding` to
`/api/recommend`.

BM25F is a field-aware version of BM25. Instead of merging all anime metadata
into one text blob, it scores separate fields with different weights:

- genres: strongest signal
- tags: strong signal
- title: medium signal
- description: supporting signal
- studios, format, source: light supporting signals

Evaluation compares the TF-IDF top 10 against the GPT-OSS annotated test set:

- Precision@10: how many of the model's top 10 are annotated as relevant.
- Recall@10: how many annotated relevant anime were recovered by the model.

## Vector Database

A vector database is not required for this project size. The catalog has about
22k anime, and MiniLM embeddings are small enough to keep in memory. A simple
NumPy matrix plus cosine similarity is easier to understand, easier to run, and
fast enough for a university project.

Use a vector database only if the project grows to hundreds of thousands or
millions of items, needs multi-user production traffic, or requires persistent
indexed approximate nearest-neighbor search.
