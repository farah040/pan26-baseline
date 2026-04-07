#!/usr/bin/env python3
import gzip
import json
import numpy as np
import faiss
import click
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from pathlib import Path
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tira.third_party_integrations import ir_datasets

# ── constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 6
STRIDE      = 2
BATCH_SIZE  = 64
MODEL_NAME  = "intfloat/e5-base-v2"

# ── chunking ──────────────────────────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in sent_tokenize(text) if s.strip()]

def sliding_window(
    sentences: list[str],
    win: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[tuple[int, str]]:
    if len(sentences) <= win:
        return [(0, " ".join(sentences))]
    return [
        (i, " ".join(sentences[i : i + win]))
        for i in range(0, len(sentences) - win + 1, stride)
    ]

def preprocess_document(doc_id: str, text: str) -> list[tuple[str, str]]:
    sentences = split_sentences(text)
    windows   = sliding_window(sentences)
    return [(doc_id, chunk_text) for _, chunk_text in windows]

def preprocess_corpus(ir_dataset) -> list[tuple[str, str]]:
    chunks = []
    for doc in ir_dataset.docs_iter():
        chunks.extend(preprocess_document(doc.doc_id, doc.default_text()))
    return chunks

def preprocess_queries(ir_dataset) -> list[tuple[str, str]]:
    chunks = []
    for query in ir_dataset.queries_iter():
        chunks.extend(preprocess_document(query.query_id, query.default_text()))
    return chunks

# ── encoding ──────────────────────────────────────────────────────────────────
def encode_chunks(
    chunks: list[tuple[str, str]],
    model: SentenceTransformer,
    prefix: str = "passage: ",
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, list[str]]:
    doc_ids    = [doc_id for doc_id, _ in chunks]
    texts      = [prefix + text for _, text in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32), doc_ids

# ── FAISS index ───────────────────────────────────────────────────────────────
def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index

def save_index(index: faiss.Index, doc_ids: list[str], index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "doc_ids.json", "w") as f:
        json.dump(doc_ids, f)
    print(f"Index saved to {index_dir}")

def load_index(index_dir: Path) -> tuple[faiss.Index, list[str]]:
    index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "doc_ids.json") as f:
        doc_ids = json.load(f)
    print(f"Index loaded: {index.ntotal} vectors")
    return index, doc_ids

def get_index(
    corpus_chunks: list[tuple[str, str]],
    model: SentenceTransformer,
    index_dir: Path,
) -> tuple[faiss.Index, list[str]]:
    if (index_dir / "faiss.index").exists() and (index_dir / "doc_ids.json").exists():
        print("Found existing index, loading...")
        return load_index(index_dir)

    print("Building index from scratch...")
    embeddings, doc_ids = encode_chunks(corpus_chunks, model, prefix="passage: ")
    index = build_index(embeddings)
    save_index(index, doc_ids, index_dir)
    return index, doc_ids

# ── search ────────────────────────────────────────────────────────────────────
def search(
    query_chunks: list[tuple[str, str]],
    index: faiss.Index,
    corpus_doc_ids: list[str],
    model: SentenceTransformer,
    top_k: int = 1000,
) -> dict[str, list[tuple[str, float]]]:
    query_embeddings, query_ids = encode_chunks(query_chunks, model, prefix="query: ")

    scores, indices = index.search(query_embeddings, top_k)

    # aggregate chunk-level scores to document level (max score per doc)
    results: dict[str, dict[str, float]] = {}
    for qid, chunk_scores, chunk_indices in zip(query_ids, scores, indices):
        if qid not in results:
            results[qid] = {}
        for score, corpus_idx in zip(chunk_scores, chunk_indices):
            doc_id = corpus_doc_ids[corpus_idx]
            if doc_id not in results[qid] or score > results[qid][doc_id]:
                results[qid][doc_id] = float(score)

    # sort each query's results by score descending
    return {
        qid: sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for qid, doc_scores in results.items()
    }

# ── output ────────────────────────────────────────────────────────────────────
def write_trec_run(
    results: dict[str, list[tuple[str, float]]],
    output_path: Path,
    tag: str = "e5_chunk_faiss",
    top_k: int = 1000,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt") as f:
        for qid, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(doc_scores[:top_k], start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score} {tag}\n")
    print(f"Run file written to {output_path}")

# ── main ──────────────────────────────────────────────────────────────────────
@click.command()
@click.option("--dataset", type=str, required=True, help="The dataset id or a local directory.")
@click.option("--output",  type=Path, required=True, help="The output directory.")
@click.option("--index",   type=Path, required=True, help="The index directory.")
def main(dataset, output, index):
    ir_dataset = ir_datasets.load(dataset)

    print("Preprocessing corpus...")
    corpus_chunks = preprocess_corpus(ir_dataset)
    print(f"  {len(corpus_chunks)} corpus chunks")

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building/loading index...")
    faiss_index, corpus_doc_ids = get_index(corpus_chunks, model, index)

    print("Preprocessing queries...")
    query_chunks = preprocess_queries(ir_dataset)
    print(f"  {len(query_chunks)} query chunks")

    print("Searching...")
    results = search(query_chunks, faiss_index, corpus_doc_ids, model)

    print("Writing run file...")
    write_trec_run(results, Path(output) / "run.txt.gz")
    print("Done.")

if __name__ == "__main__":
    main()
