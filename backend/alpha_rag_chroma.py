#!/usr/bin/env python3
"""Build a simple CSV-based RAG pipeline and store embeddings in ChromaDB.

Usage:
    export GEMINI_API_KEY="your_google_api_key"
    pip install chromadb google-genai

    0) Default behavior (no command):
         python alpha_rag_chroma.py

  1) Ingest CSV rows into ChromaDB:
     python alpha_rag_chroma.py ingest --csv alpha_factors.csv --db ./chroma_db

  2) Query the vector store:
     python alpha_rag_chroma.py query --query "momentum alpha with volume" -k 5 --db ./chroma_db
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List


DEFAULT_COLLECTION = "alpha_factors"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
# Put your Gemini API key here if you want to avoid environment variables.
GEMINI_API_KEY = "AIzaSyB1dlz_SePTDmWrL7Q_YFAikvARJRIpjdA"


@dataclass
class Record:
    alpha: str
    formula: str
    definition: str
    tags: str


def load_csv_records(csv_path: str) -> List[Record]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    records: List[Record] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"Alpha", "Formula", "Definition", "Tags"}
        if not reader.fieldnames or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must include headers: {sorted(expected)}. Got: {reader.fieldnames}"
            )

        for row in reader:
            records.append(
                Record(
                    alpha=(row.get("Alpha") or "").strip(),
                    formula=(row.get("Formula") or "").strip(),
                    definition=(row.get("Definition") or "").strip(),
                    tags=(row.get("Tags") or "").strip(),
                )
            )
    return records


def record_to_document(record: Record) -> str:
    return (
        f"Alpha: {record.alpha}\n"
        f"Formula: {record.formula}\n"
        f"Definition: {record.definition}\n"
        f"Tags: {record.tags}"
    )


class GeminiEmbeddingFunction:
    """Chroma-compatible embedding function backed by Gemini embeddings."""

    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
        # google-genai expects plain model IDs like "text-embedding-004".
        self._normalized_model = self.model_name.replace("models/", "")
        self._resolved_model = self._normalized_model

    def name(self) -> str:
        # Chroma expects custom embedding functions to provide a stable name.
        return f"gemini:{self.model_name}"

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        # Chroma may call this explicitly on newer versions.
        return self.__call__(input)

    def embed_query(self, input: Any) -> List[List[float]]:
        # Chroma query path may pass either a string or list of strings.
        if isinstance(input, str):
            return self.__call__([input])
        if isinstance(input, list):
            return self.__call__(input)
        raise TypeError(f"Unsupported query input type: {type(input)}")

    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Missing Gemini SDK. Install with: pip install google-genai"
            ) from exc

        client = genai.Client(api_key=self.api_key)

        # Different Gemini projects/accounts may expose different embedding model IDs.
        model_candidates = [
            self._resolved_model,
            "gemini-embedding-001",
            "text-embedding-004",
            "embedding-001",
        ]
        deduped_candidates: List[str] = []
        seen = set()
        for m in model_candidates:
            if m and m not in seen:
                seen.add(m)
                deduped_candidates.append(m)

        last_exc: Exception | None = None
        response = None
        for model in deduped_candidates:
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=input,
                )
                self._resolved_model = model
                break
            except Exception as exc:
                if "not found" in str(exc).lower() or "not supported" in str(exc).lower():
                    last_exc = exc
                    continue
                raise

        if response is None:
            if last_exc is not None:
                raise RuntimeError(
                    "No available Gemini embedding model found. "
                    "Tried: gemini-embedding-001, text-embedding-004, embedding-001"
                ) from last_exc
            raise RuntimeError("Failed to get embedding response from Gemini.")

        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise RuntimeError("Gemini embedding response did not contain embeddings.")

        vectors: List[List[float]] = []
        for emb in embeddings:
            values = getattr(emb, "values", None)
            if values is None and isinstance(emb, dict):
                values = emb.get("values")
            if values is None:
                raise RuntimeError("Unexpected Gemini embedding payload format.")
            vectors.append([float(v) for v in values])

        return vectors


def get_collection(db_path: str, collection_name: str) -> Any:
    try:
        import chromadb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Install with: pip install chromadb google-genai"
        ) from exc

    api_key = GEMINI_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY in code or export GEMINI_API_KEY."
        )

    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = GeminiEmbeddingFunction(api_key=api_key, model_name=DEFAULT_EMBED_MODEL)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"source": "alpha_factors_csv"},
    )


def ingest(csv_path: str, db_path: str, collection_name: str, batch_size: int = 64) -> None:
    records = load_csv_records(csv_path)
    if not records:
        print("No records found in CSV. Nothing to ingest.")
        return

    collection = get_collection(db_path, collection_name)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, str]] = []

    for idx, r in enumerate(records, start=1):
        alpha_id = r.alpha if r.alpha else f"row_{idx}"
        ids.append(alpha_id)
        docs.append(record_to_document(r))
        metas.append(
            {
                "alpha": r.alpha,
                "tags": r.tags,
                "formula": r.formula,
                "source_file": os.path.basename(csv_path),
            }
        )

    # Upsert in batches to avoid very large single requests.
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
        )

    print(f"Ingested {len(ids)} records into '{collection_name}' at '{db_path}'.")


def query(db_path: str, collection_name: str, query_text: str, top_k: int) -> None:
    collection = get_collection(db_path, collection_name)
    total = collection.count()
    if total == 0:
        print("Collection is empty. Run ingest first.")
        return

    result = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    if not documents:
        print("No similar records found.")
        return

    print(f"Query: {query_text}")
    print(f"Top {len(documents)} matches from '{collection_name}':\n")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        alpha = (meta or {}).get("alpha", "N/A")
        tags = (meta or {}).get("tags", "")
        print(f"[{i}] alpha={alpha} | distance={dist:.4f} | tags={tags}")
        print(doc)
        print("-" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSV -> ChromaDB RAG utility")
    parser.add_argument("--db", default="./chroma_db", help="ChromaDB persistence directory")
    parser.add_argument(
        "--csv",
        default="alpha_factors.csv",
        help="Input CSV path for default mode and ingest",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name"
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Gemini embedding model, e.g. gemini-embedding-001",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    ingest_cmd = subparsers.add_parser("ingest", help="Ingest CSV rows into ChromaDB")
    ingest_cmd.add_argument("--csv", default="alpha_factors.csv", help="Input CSV path")

    query_cmd = subparsers.add_parser("query", help="Run retrieval query against ChromaDB")
    query_cmd.add_argument("--query", required=True, help="Query text")
    query_cmd.add_argument("-k", "--top-k", type=int, default=1, help="Number of results")

    return parser


def main() -> None:
    global DEFAULT_EMBED_MODEL

    parser = build_parser()
    args = parser.parse_args()
    DEFAULT_EMBED_MODEL = args.embed_model

    if args.command in (None, "ingest"):
        ingest(csv_path=args.csv, db_path=args.db, collection_name=args.collection)
    elif args.command == "query":
        query(
            db_path=args.db,
            collection_name=args.collection,
            query_text=args.query,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
