"""
Statement Copilot - Chroma Vector DB Setup Script (OpenRouter Embeddings + Hybrid BM25)
====================================================================================

Bu script DuckDB'deki transaction verilerini embedding'e evirip Chroma'ya ykler.

 Gncel yaklam
- Embedding: OpenRouter /embeddings (OpenAI-compatible)
- Varsaylan model: google/gemini-embedding-001 (multilingual ok gl)
- Chroma: collection embedding_function zerinden (query_texts + upsert docs)
- Metadata: str/int/float/bool dndaki tipler sanitize edilir
- Tenant filtreleme: app-level tenant_id zorunlu (where={"tenant_id": ...})
- Hybrid retrieval: Vector + BM25 (rank-bm25) + RRF (rank fusion)

Kullanm:
    pip install chromadb duckdb requests rank-bm25

    export OPENROUTER_API_KEY="..."
    python scripts/setup/chroma_setup.py
    TENANT_ID=acme python scripts/setup/chroma_setup.py
    python scripts/setup/chroma_setup.py --reset
    python scripts/setup/chroma_setup.py --only-test
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import duckdb
import requests
from rank_bm25 import BM25Okapi

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chroma_setup")

# ---------------------------
# Paths & Config
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "db" / "statement_copilot.duckdb"
CHROMA_PATH = PROJECT_ROOT / "vector_store" / "chroma_db"

DEFAULT_TENANT_ID = os.getenv("TENANT_ID", "default_tenant").strip() or "default_tenant"
DEFAULT_USER_ID = os.getenv("USER_ID", "default_user").strip() or "default_user"

# OpenRouter Embeddings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()     # optional analytics header
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "").strip()   # optional analytics header
OPENROUTER_EMBED_URL = os.getenv("OPENROUTER_EMBED_URL", "https://openrouter.ai/api/v1/embeddings").strip()
OPENROUTER_EMBED_BATCH = int(os.getenv("OPENROUTER_EMBED_BATCH", "96"))  # safety batch size

# "En gl" default (multilingual)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-001").strip()

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "transactions_v1")
BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "256"))

# Hybrid weights (RRF)
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "30"))  # vector ve bm25 aday says

# Chroma built-in tenant/database (opsiyonel)
CHROMA_TENANT_ENV = os.getenv("CHROMA_TENANT", "").strip()
CHROMA_DB_ENV = os.getenv("CHROMA_DATABASE", "").strip()

# Merchant bilgisi gmme metninde grlt yapmasn diye
GENERIC_MERCHANTS = {
    "CURRENCY_EXCHANGE_INTERNAL",
    "AUTHORIZATION_HOLD",
    "P2P_TRANSFER",
    "ATM_WITHDRAWAL",
    "BANK_FEE_TRANSFER",
    "UTILITY_BILL",
    "INTERNAL_TRANSFER",
}

# Chroma metadata iin pratik limitler
MAX_META_STR_LEN = 256
MAX_DOC_LEN = 2000

# ---------------------------
# Types
# ---------------------------
@dataclass
class TransactionDocument:
    tx_id: str
    text: str
    metadata: Dict[str, Any]


class VectorStoreProtocol(Protocol):
    def upsert(self, documents: List[TransactionDocument]) -> None: ...
    def query(self, query_text: str, n_results: int = 10, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def delete(self, ids: List[str]) -> None: ...
    def count(self) -> int: ...


# ---------------------------
# Helpers
# ---------------------------
def _safe_str(x: Any, max_len: int = MAX_META_STR_LEN) -> str:
    if x is None:
        return ""
    s = str(x).replace("\x00", "").strip()
    if len(s) > max_len:
        return s[: max_len - 1] + ""
    return s


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def _coerce_datetime(x: Any) -> Optional[datetime]:
    if x is None or x == "":
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    if isinstance(x, (int, float)):
        try:
            return datetime.fromtimestamp(float(x))
        except Exception:
            return None
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _coerce_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            s = str(v).strip().lower()
            if s:
                out.append(s)
        return sorted(set(out))
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.split(",") if p.strip()]
        return sorted(set(parts))
    s = str(value).strip().lower()
    return [s] if s else []


def _tags_to_csv(tags: List[str], max_elems: int = 64) -> str:
    return ",".join(tags[:max_elems])


def _sanitize_metadata_for_chroma(meta: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, bool):
            clean[k] = v
            continue
        if isinstance(v, (int, float)):
            clean[k] = v
            continue
        if isinstance(v, datetime):
            clean[k] = _safe_str(v.isoformat(), MAX_META_STR_LEN)
            continue
        if isinstance(v, (list, dict, tuple, set)):
            clean[k] = _safe_str(str(v), MAX_META_STR_LEN)
            continue
        clean[k] = _safe_str(v, MAX_META_STR_LEN)
    return clean


def _build_embed_text(data: Dict[str, Any]) -> str:
    parts: List[str] = []

    merchant_norm = _safe_str(data.get("merchant_norm"), 128)
    description = _safe_str(data.get("description"), 512)
    category = _safe_str(data.get("category"), 64)
    subcategory = _safe_str(data.get("subcategory"), 64)
    direction = _safe_str(data.get("direction"), 16)
    channel = _safe_str(data.get("channel"), 64)
    tx_type = _safe_str(data.get("transaction_type"), 64)

    amount = _to_float(data.get("amount"), 0.0)

    if merchant_norm and merchant_norm.upper() not in GENERIC_MERCHANTS:
        parts.append(f"Merchant: {merchant_norm}")

    if description:
        parts.append(f"Aklama: {description}")

    if category:
        parts.append(f"Kategori: {category}")
        if subcategory:
            parts.append(f"Alt kategori: {subcategory}")

    if direction == "expense":
        parts.append(f"Gider: {abs(amount):.2f}")
    elif direction == "income":
        parts.append(f"Gelir: {amount:.2f}")
    elif direction == "transfer":
        parts.append(f"Transfer: {abs(amount):.2f}")

    if channel:
        parts.append(f"Kanal: {channel}")

    if tx_type:
        parts.append(f"lem tipi: {tx_type}")

    text = " | ".join([p for p in parts if p]).strip()
    if not text:
        text = f"Transaction {data.get('tx_id', '')} | {direction} | {amount:.2f}"

    if len(text) > MAX_DOC_LEN:
        text = text[: MAX_DOC_LEN - 1] + ""

    return text


def _tokenize(text: str) -> List[str]:
    # Unicode word tokenize (Trke dahil)
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


# ---------------------------
# OpenRouter Embedding Function (for Chroma)
# ---------------------------
class OpenRouterEmbeddingFunction:
    """
    Chroma embedding_function interface: callable that returns List[List[float]]
    Uses OpenRouter OpenAI-compatible embeddings endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1/embeddings",
        batch_size: int = 96,
        site_url: str = "",
        site_name: str = "",
        timeout_s: int = 60,
        max_retries: int = 4,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY bo. OpenRouter embeddings kullanmak iin API key gerekli.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.batch_size = max(1, batch_size)
        self.site_url = site_url
        self.site_name = site_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional OpenRouter headers (analytics / discoverability)
        if self.site_url:
            h["HTTP-Referer"] = self.site_url
        if self.site_name:
            h["X-Title"] = self.site_name
        return h

    def _post_embeddings(self, inputs: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(
                    self.base_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_s,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"OpenRouter transient error: {r.status_code} {r.text[:200]}")
                if r.status_code >= 400:
                    raise RuntimeError(f"OpenRouter error: {r.status_code} {r.text[:500]}")
                data = r.json()
                items = data.get("data", [])
                # items: [{ "embedding": [...], "index": i }, ...]
                embeddings = [None] * len(inputs)
                for it in items:
                    idx = it.get("index")
                    emb = it.get("embedding")
                    if isinstance(idx, int) and emb is not None:
                        embeddings[idx] = emb
                if any(e is None for e in embeddings):
                    raise RuntimeError("OpenRouter embedding response eksik index dndrd.")
                return embeddings  # type: ignore[return-value]
            except Exception as e:
                last_err = e
                sleep_s = min(2 ** attempt, 10)
                logger.warning(f"OpenRouter embeddings retry {attempt}/{self.max_retries}: {e} (sleep {sleep_s}s)")
                time.sleep(sleep_s)

        raise RuntimeError(f"OpenRouter embeddings failed after retries: {last_err}")

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        out: List[List[float]] = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            out.extend(self._post_embeddings(batch))
        return out


# ---------------------------
# Chroma Store + Hybrid (BM25)
# ---------------------------
@dataclass
class _BM25TenantIndex:
    ids: List[str]
    docs: List[str]
    tokenized_docs: List[List[str]]
    bm25: BM25Okapi
    built_at: float


class ChromaVectorStore(VectorStoreProtocol):
    def __init__(
        self,
        persist_path: Path,
        collection_name: str,
        embedding_model: str,
        chroma_tenant: Optional[str] = None,
        chroma_database: Optional[str] = None,
    ):
        import chromadb
        from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

        persist_path.mkdir(parents=True, exist_ok=True)

        tenant = (chroma_tenant or DEFAULT_TENANT).strip() or DEFAULT_TENANT
        database = (chroma_database or DEFAULT_DATABASE).strip() or DEFAULT_DATABASE

        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
            tenant=tenant,
            database=database,
        )

        self.collection_name = collection_name

        logger.info(f"Embedding function (OpenRouter) hazrlanyor: {embedding_model}")
        self.embedding_fn = OpenRouterEmbeddingFunction(
            api_key=OPENROUTER_API_KEY,
            model=embedding_model,
            base_url=OPENROUTER_EMBED_URL,
            batch_size=OPENROUTER_EMBED_BATCH,
            site_url=OPENROUTER_SITE_URL,
            site_name=OPENROUTER_SITE_NAME,
        )

        # BM25 cache (tenant bazl)
        self._bm25_cache: Dict[str, _BM25TenantIndex] = {}

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
            )
            logger.info(f"Chroma collection yklendi: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_fn,
            )
            logger.info(f"Chroma collection oluturuldu: {collection_name}")

    def reset_collection(self) -> None:
        logger.warning(f"Collection reset: {self.collection_name}")
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
        self._bm25_cache.clear()

    def upsert(self, documents: List[TransactionDocument]) -> None:
        if not documents:
            return

        ids = [d.tx_id for d in documents]
        texts = [d.text for d in documents]
        metadatas = [_sanitize_metadata_for_chroma(d.metadata) for d in documents]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        # data deiti -> bm25 cache invalid
        tenant_ids = {d.metadata.get("tenant_id", "") for d in documents}
        for t in tenant_ids:
            t = (t or "").strip()
            if t and t in self._bm25_cache:
                del self._bm25_cache[t]

    def query(self, query_text: str, n_results: int = 10, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=ids)
        # cache invalidate (basite hepsini temizle)
        self._bm25_cache.clear()

    def count(self) -> int:
        return self.collection.count()

    # -------- Hybrid (BM25 + Vector) --------
    def _iter_all_docs(self, where: Dict[str, Any], page_size: int = 5000) -> Tuple[List[str], List[str]]:
        all_ids: List[str] = []
        all_docs: List[str] = []
        offset = 0
        while True:
            batch = self.collection.get(
                where=where,
                include=["documents"],
                limit=page_size,
                offset=offset,
            )
            ids = batch.get("ids", []) or []
            docs = batch.get("documents", []) or []
            if not ids:
                break
            # docs can be list[str] aligned to ids
            for _id, _doc in zip(ids, docs):
                all_ids.append(_id)
                all_docs.append(_doc or "")
            offset += page_size
        return all_ids, all_docs

    def _ensure_bm25(self, tenant_id: str) -> _BM25TenantIndex:
        tenant_id = (tenant_id or "").strip()
        if not tenant_id:
            raise ValueError("BM25 index iin tenant_id zorunlu.")

        cached = self._bm25_cache.get(tenant_id)
        if cached is not None:
            return cached

        logger.info(f"BM25 index build (tenant={tenant_id}) -> Chroma'dan dokmanlar ekiliyor...")
        ids, docs = self._iter_all_docs(where={"tenant_id": tenant_id})
        if not ids:
            # empty index
            tokenized_docs: List[List[str]] = [[]]
            bm25 = BM25Okapi(tokenized_docs)
            idx = _BM25TenantIndex(ids=[], docs=[], tokenized_docs=[], bm25=bm25, built_at=time.time())
            self._bm25_cache[tenant_id] = idx
            return idx

        tokenized_docs = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized_docs)
        idx = _BM25TenantIndex(
            ids=ids,
            docs=docs,
            tokenized_docs=tokenized_docs,
            bm25=bm25,
            built_at=time.time(),
        )
        self._bm25_cache[tenant_id] = idx
        logger.info(f"BM25 index hazr: {len(ids):,} dokman")
        return idx

    def _bm25_search(self, tenant_id: str, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        idx = self._ensure_bm25(tenant_id)
        if not idx.ids:
            return []
        qtok = _tokenize(query_text)
        scores = idx.bm25.get_scores(qtok)  # numpy array
        # top-k indices
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx.ids[i], float(s)) for i, s in ranked if s > 0]

    def hybrid_query(
        self,
        query_text: str,
        tenant_id: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        vector_weight: float = HYBRID_VECTOR_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        rrf_k: int = HYBRID_RRF_K,
        candidates: int = HYBRID_CANDIDATES,
    ) -> Dict[str, Any]:
        tenant_id = (tenant_id or "").strip()
        if not tenant_id:
            raise ValueError("hybrid_query iin tenant_id zorunlu.")

        where_final = dict(where or {})
        where_final["tenant_id"] = tenant_id

        # 1) Vector candidates
        vec = self.collection.query(
            query_texts=[query_text],
            n_results=max(candidates, n_results),
            where=where_final,
            include=["documents", "metadatas", "distances"],
        )
        vec_ids = (vec.get("ids", [[]]) or [[]])[0]
        vec_docs = (vec.get("documents", [[]]) or [[]])[0]
        vec_metas = (vec.get("metadatas", [[]]) or [[]])[0]
        vec_dists = (vec.get("distances", [[]]) or [[]])[0]

        # 2) BM25 candidates
        bm25_pairs = self._bm25_search(tenant_id=tenant_id, query_text=query_text, top_k=max(candidates, n_results))
        bm25_ids = [i for i, _ in bm25_pairs]

        # 3) RRF fusion
        scores: Dict[str, float] = {}
        meta_by_id: Dict[str, Dict[str, Any]] = {}
        doc_by_id: Dict[str, str] = {}
        dist_by_id: Dict[str, float] = {}

        for rank, _id in enumerate(vec_ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + (vector_weight / (rrf_k + rank))
        for rank, _id in enumerate(bm25_ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + (bm25_weight / (rrf_k + rank))

        # Fill from vector payload
        for _id, d, m, dist in zip(vec_ids, vec_docs, vec_metas, vec_dists):
            if _id not in doc_by_id:
                doc_by_id[_id] = d or ""
            if _id not in meta_by_id:
                meta_by_id[_id] = m or {}
            if isinstance(dist, (int, float)):
                dist_by_id[_id] = float(dist)

        # Fetch missing docs/metas for BM25-only ids
        missing = [i for i in bm25_ids if i not in doc_by_id]
        if missing:
            got = self.collection.get(ids=missing, include=["documents", "metadatas"])
            got_ids = got.get("ids", []) or []
            got_docs = got.get("documents", []) or []
            got_metas = got.get("metadatas", []) or []
            for _id, d, m in zip(got_ids, got_docs, got_metas):
                doc_by_id[_id] = d or ""
                meta_by_id[_id] = m or {}

        ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        out_ids = [i for i, _ in ranked_ids]
        out_docs = [doc_by_id.get(i, "") for i in out_ids]
        out_metas = [meta_by_id.get(i, {}) for i in out_ids]
        out_dists = [dist_by_id.get(i, None) for i in out_ids]

        return {
            "ids": [out_ids],
            "documents": [out_docs],
            "metadatas": [out_metas],
            "distances": [out_dists],   # vector dist varsa, yoksa None
            "scores": [{i: scores[i] for i in out_ids}],  # hybrid score map
        }


# ---------------------------
# Transform DuckDB row -> TransactionDocument
# ---------------------------
def create_transaction_document(row: Tuple[Any, ...], columns: List[str]) -> TransactionDocument:
    data = dict(zip(columns, row))

    dt = _coerce_datetime(data.get("date_time"))
    date_epoch = int(dt.timestamp()) if dt else 0

    tags_arr_list = _coerce_tags(data.get("tags_arr"))
    tags_final_list = _coerce_tags(data.get("tags_final"))

    tags_arr_csv = _tags_to_csv(tags_arr_list)
    tags_final_csv = _tags_to_csv(tags_final_list)

    embed_text = _build_embed_text(data)

    tx_id = _safe_str(data.get("tx_id"), 64)
    if not tx_id:
        raise ValueError("tx_id bo olamaz (DuckDB -> Chroma).")

    metadata = {
        "tx_id": tx_id,
        "file_id": _safe_str(data.get("file_id"), 32),
        "tenant_id": _safe_str(data.get("tenant_id"), 64),
        "user_id": _safe_str(data.get("user_id") or DEFAULT_USER_ID, 64),
        "currency": _safe_str(data.get("currency"), 8),

        "date_time": dt.isoformat() if dt else "",
        "date_epoch": _to_int(date_epoch, 0),
        "month_year": _safe_str(data.get("month_year"), 10),

        "amount": _to_float(data.get("amount"), 0.0),
        "balance": _to_float(data.get("balance"), 0.0),
        "confidence": _to_float(data.get("confidence"), 0.0),

        "direction": _safe_str(data.get("direction"), 16),
        "category": _safe_str(data.get("category"), 64),
        "subcategory": _safe_str(data.get("subcategory"), 64),
        "category_final": _safe_str(data.get("category_final"), 64),
        "subcategory_final": _safe_str(data.get("subcategory_final"), 64),

        "merchant_norm": _safe_str(data.get("merchant_norm"), 128),
        "merchant_key": _safe_str(data.get("merchant_key"), 128),

        "channel": _safe_str(data.get("channel"), 64),
        "transaction_type": _safe_str(data.get("transaction_type"), 64),
        "transaction_code": _safe_str(data.get("transaction_code"), 32),

        "tags": _safe_str(data.get("tags"), 256),
        "tags_arr_csv": _safe_str(tags_arr_csv, 512),
        "tags_final_csv": _safe_str(tags_final_csv, 512),
        "tags_arr_count": _to_int(len(tags_arr_list), 0),
        "tags_final_count": _to_int(len(tags_final_list), 0),
    }

    metadata = _sanitize_metadata_for_chroma(metadata)
    return TransactionDocument(tx_id=tx_id, text=embed_text, metadata=metadata)


# ---------------------------
# Load DuckDB -> Chroma
# ---------------------------
def load_transactions_to_vector_db(
    duckdb_path: Path,
    vector_store: VectorStoreProtocol,
    tenant_id: str,
    batch_size: int = BATCH_SIZE,
) -> None:
    tenant_id = (tenant_id or "").strip()
    if not tenant_id:
        raise ValueError("tenant_id zorunlu (multi-tenant filtreleme iin).")

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        total_count = con.execute(
            "SELECT COUNT(*) FROM transactions WHERE tenant_id = ?",
            [tenant_id],
        ).fetchone()[0]

        logger.info(f"Tenant={tenant_id} | Toplam {total_count:,} transaction yklenecek")
        if total_count == 0:
            logger.warning("Yklenecek veri yok.")
            return

        columns = [
            "tx_id", "file_id", "tenant_id", "user_id", "currency",
            "date_time", "value_date", "amount", "balance", "direction",
            "description", "merchant_norm", "merchant_key",
            "category", "subcategory", "category_final", "subcategory_final",
            "tags", "tags_arr", "tags_final", "confidence",
            "channel", "transaction_type", "transaction_code", "month_year",
        ]

        offset = 0
        processed = 0
        t0 = time.time()

        while offset < total_count:
            query = f"""
                SELECT {", ".join(columns)}
                FROM transactions
                WHERE tenant_id = ?
                ORDER BY tx_id
                LIMIT {batch_size} OFFSET {offset}
            """
            rows = con.execute(query, [tenant_id]).fetchall()
            if not rows:
                break

            docs: List[TransactionDocument] = []
            for row in rows:
                try:
                    docs.append(create_transaction_document(row, columns))
                except Exception as e:
                    logger.warning(f"Row skipped (offset={offset}): {e}")

            if docs:
                vector_store.upsert(docs)

            processed += len(docs)
            offset += batch_size

            progress = (processed / total_count) * 100
            elapsed = time.time() - t0
            logger.info(f"lerleme: {processed:,}/{total_count:,} ({progress:.1f}%) | {elapsed:.1f}s")

        logger.info(f" Ykleme tamamland. Toplam upsert: {processed:,}")

    finally:
        con.close()


# ---------------------------
# Test (Hybrid Search)
# ---------------------------
def test_hybrid_search(vector_store: ChromaVectorStore, tenant_id: str) -> None:
    print("\n" + "=" * 70)
    print(" HYBRID SEARCH TEST (Vector + BM25 via RRF)")
    print("=" * 70)

    test_queries = [
        "market alverii",
        "fatura demesi",
        "Netflix abonelik",
        "havale transfer",
        "ATM para ekme",
        "dviz ilemi",
        "Google Cloud deme",
        "byk harcama",
    ]

    for query in test_queries:
        print(f"\n Query: '{query}'")
        where = {"tenant_id": tenant_id}
        results = vector_store.hybrid_query(query, tenant_id=tenant_id, n_results=3, where=where)

        docs = results.get("documents", [[]])[0] if results else []
        metas = results.get("metadatas", [[]])[0] if results else []
        dists = results.get("distances", [[]])[0] if results else []
        score_map = (results.get("scores", [{}]) or [{}])[0]

        if docs and metas:
            for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                merchant = meta.get("merchant_norm", "N/A")
                amount = meta.get("amount", 0.0)
                date_str = (meta.get("date_time", "") or "")[:10]
                dist = dists[i - 1] if i - 1 < len(dists) else None
                hid = (results.get("ids", [[]])[0][i - 1]) if results.get("ids") else ""
                hscore = score_map.get(hid, 0.0)

                dist_str = f"vec_dist={dist:.4f}" if isinstance(dist, (int, float)) else "vec_dist=NA"
                print(f"   {i}. {merchant:<22} | {amount:>10.2f} | {date_str} | hybrid={hscore:.4f} | {dist_str}")
                print(f"      {doc[:120]}{'...' if len(doc) > 120 else ''}")
        else:
            print("   Sonu bulunamad")


# ---------------------------
# CLI / Main
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", type=str, default=DEFAULT_TENANT_ID, help="tenant_id override (app-level)")
    p.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Chroma collection name")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    p.add_argument("--reset", action="store_true", help="Collection' sfrla (sil + yeniden olutur)")
    p.add_argument("--only-test", action="store_true", help="Run only test queries")

    p.add_argument("--chroma-tenant", type=str, default=CHROMA_TENANT_ENV, help="Chroma tenant (opsiyonel)")
    p.add_argument("--chroma-db", type=str, default=CHROMA_DB_ENV, help="Chroma database (opsiyonel)")
    return p.parse_args()


def main() -> None:
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY bulunamad. Export edip tekrar dene.")
        return

    args = parse_args()
    tenant_id = (args.tenant or DEFAULT_TENANT_ID).strip() or "default_tenant"

    print(" Statement Copilot - Chroma Vector DB Kurulumu (OpenRouter + Hybrid)")
    print("=" * 70)
    print(f"Tenant (app):      {tenant_id}")
    print(f"Collection:        {args.collection}")
    print(f"DB:                {DB_PATH}")
    print(f"Chroma path:       {CHROMA_PATH}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Embedding model:   {EMBEDDING_MODEL}")
    print(f"OpenRouter URL:    {OPENROUTER_EMBED_URL}")
    if args.chroma_tenant:
        print(f"Chroma tenant:     {args.chroma_tenant}")
    if args.chroma_db:
        print(f"Chroma database:   {args.chroma_db}")

    if not DB_PATH.exists():
        logger.error(f"DuckDB bulunamad: {DB_PATH}")
        logger.error("nce scripts/setup/duckdb_setup.py altrn")
        return

    vector_store = ChromaVectorStore(
        persist_path=CHROMA_PATH,
        collection_name=args.collection,
        embedding_model=EMBEDDING_MODEL,
        chroma_tenant=args.chroma_tenant or None,
        chroma_database=args.chroma_db or None,
    )

    if args.reset:
        vector_store.reset_collection()

    if not args.only_test:
        existing = vector_store.count()
        logger.info(f"Mevcut dokman says: {existing:,}")

        load_transactions_to_vector_db(
            duckdb_path=DB_PATH,
            vector_store=vector_store,
            tenant_id=tenant_id,
            batch_size=args.batch_size,
        )

        final_count = vector_store.count()
        logger.info(f"Final dokman says: {final_count:,}")

    test_hybrid_search(vector_store, tenant_id=tenant_id)

    print("\n" + "=" * 70)
    print(f" Chroma setup tamam: {CHROMA_PATH} / collection={args.collection}")
    print("=" * 70)


if __name__ == "__main__":
    main()
