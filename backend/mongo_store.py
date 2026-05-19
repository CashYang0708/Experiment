from __future__ import annotations

from datetime import datetime, timezone
import os

from pymongo import MongoClient


_mongo_client = None


def _get_collection():
    global _mongo_client
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        return None
    if _mongo_client is None:
        _mongo_client = MongoClient(uri)
    db_name = os.getenv("MONGODB_DB", "quant_evaluation").strip() or "quant_evaluation"
    collection_name = os.getenv("MONGODB_COLLECTION", "evaluation_reports").strip() or "evaluation_reports"
    return _mongo_client[db_name][collection_name]


def save_report(user_query: str, report: str, label: str) -> None:
    collection = _get_collection()
    if collection is None:
        return
    payload = {
        "user_query": user_query,
        "evaluation_report": report,
        "label": label,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        collection.insert_one(payload)
    except Exception as exc:
        print(f"MongoDB insert failed: {exc}")
