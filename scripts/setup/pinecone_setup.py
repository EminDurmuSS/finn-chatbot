"""
Statement Copilot - Pinecone Setup Script (Production, Pinecone HTTP SDK)
========================================================================
 OpenRouter Embeddings +  Pinecone Hybrid Search (Dense + BM25 Sparse)

Gereksinimler:
    pip install -U pinecone duckdb requests pinecone-text

Environment Variables:
    # Pinecone
    PINECONE_API_KEY=your_api_key
    PINECONE_INDEX_HOST="your-index-host"          # (recommended) Pinecone Console > Index > HOST

    # Index creation (optional):
    # Serverless:
    PINECONE_CLOUD=aws|gcp|azure
    PINECONE_REGION=us-east-1
    # Pod-based (if your account allows):
    PINECONE_ENVIRONMENT=us-west1-gcp
    PINECONE_POD_TYPE=p1.x1
    PINECONE_PODS=1

    # OpenRouter
    OPENROUTER_API_KEY=your_openrouter_key
    OPENROUTER_EMBED_MODEL="qwen/qwen3-embedding-8b"
    OPENROUTER_EMBED_DIMENSIONS=4096               # Qwen3-Embedding-8B supports 32..4096; default 4096
    OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
    # Opsiyonel (OpenRouter telemetry):
    OPENROUTER_SITE_URL="https://your-site.com"
    OPENROUTER_APP_NAME="StatementCopilot"

    # Hybrid / BM25
    HYBRID_ALPHA=0.70                              # 0=sparse only(BM25), 1=dense only(embedding)
    BM25_FIT_MAX_DOCS=200000                       # max doc count for initial BM25 fit (reduce if corpus is large)
    BM25_REFIT=0                                   # 1 ise BM25 params yeniden fit edilir

Notlar:
- For hybrid search, set the Pinecone index metric to "dotproduct".
- If the existing index has a different dimension/metric, use a new index name (PINECONE_INDEX_NAME).
"""

import os
import time
import json
import math
import duckdb
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# -----------------------------------------------------------------------------
from pinecone import Pinecone, ServerlessSpec, PodSpec

# -----------------------------------------------------------------------------
from pinecone_text.sparse import BM25Encoder
from pinecone_text.hybrid import hybrid_convex_scale

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "db" / "statement_copilot.duckdb"
HOST_CACHE_PATH = PROJECT_ROOT / "db" / "pinecone_index_host.json"
BM25_PARAMS_PATH = PROJECT_ROOT / "db" / "bm25_params.json"

# ---------------------------
# .env Loader (lightweight, no deps)
# ---------------------------
def load_dotenv(dotenv_path: Optional[Path] = None, override: bool = False) -> Optional[Path]:
    candidates: List[Path] = []
    if dotenv_path:
        candidates.append(dotenv_path)
    else:
        candidates.append(Path.cwd())
        candidates.append(Path(__file__).resolve().parent)

    seen = set()
    resolved: List[Path] = []
    for start in candidates:
        for parent in [start, *start.parents]:
            if parent in seen:
                continue
            seen.add(parent)
            resolved.append(parent)

    env_file: Optional[Path] = None
    for parent in resolved:
        if parent.is_file() and parent.name == ".env":
            env_file = parent
            break
        candidate = parent / ".env"
        if candidate.is_file():
            env_file = candidate
            break

    if not env_file:
        return None

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value

    return env_file


# Load .env early so module-level os.getenv picks it up
load_dotenv()

# ---------------------------
# Pinecone Configuration
# ---------------------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "statement-copilot-transactions-hybrid")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# Hybrid requires dotproduct metric in Pinecone (dense + sparse in same index)
PINECONE_METRIC = "dotproduct"

# ---------------------------
# OpenRouter Embedding Config
# ---------------------------
OPENROUTER_EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-8b")
OPENROUTER_EMBED_DIMENSIONS = int(os.getenv("OPENROUTER_EMBED_DIMENSIONS", "4096"))
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "60"))
OPENROUTER_BATCH = int(os.getenv("OPENROUTER_BATCH", "64"))

# The Pinecone index dimension must match your embedding dimension
EMBEDDING_DIMENSION = OPENROUTER_EMBED_DIMENSIONS

# ---------------------------
# Hybrid / BM25 Config
# ---------------------------
HYBRID_ALPHA_DEFAULT = float(os.getenv("HYBRID_ALPHA", "0.70"))
BM25_FIT_MAX_DOCS = int(os.getenv("BM25_FIT_MAX_DOCS", "200000"))
BM25_REFIT = os.getenv("BM25_REFIT", "0") == "1"


def _load_cached_host(index_name: str) -> Optional[str]:
    if not HOST_CACHE_PATH.exists():
        return None
    try:
        data = json.loads(HOST_CACHE_PATH.read_text(encoding="utf-8"))
        return data.get(index_name)
    except Exception:
        return None


def _save_cached_host(index_name: str, host: str) -> None:
    HOST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, str] = {}
    if HOST_CACHE_PATH.exists():
        try:
            data = json.loads(HOST_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[index_name] = host
    HOST_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_filter(filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    {k: v}  -> {k: {"$eq": v}}
    """
    if not filter_dict:
        return None
    out: Dict[str, Any] = {}
    for k, v in filter_dict.items():
        out[k] = v if isinstance(v, dict) else {"$eq": v}
    return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _parse_datetime(dt_val: Any) -> Optional[datetime]:
    if not dt_val:
        return None
    if isinstance(dt_val, datetime):
        return dt_val
    if isinstance(dt_val, str):
        s = dt_val.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def _l2_normalize(vec: List[float], eps: float = 1e-12) -> List[float]:
    s = 0.0
    for v in vec:
        s += v * v
    n = math.sqrt(s)
    if n < eps:
        return vec
    return [v / n for v in vec]


class OpenRouterEmbedder:
    """
    OpenRouter OpenAI-compatible embeddings endpoint:
      POST {OPENROUTER_BASE_URL}/embeddings
      headers: Authorization: Bearer <key>
    """

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable gerekli")

        self.api_key = api_key
        self.model = OPENROUTER_EMBED_MODEL
        self.dimensions = OPENROUTER_EMBED_DIMENSIONS
        self.base_url = OPENROUTER_BASE_URL
        self.timeout = OPENROUTER_TIMEOUT

        self.site_url = os.getenv("OPENROUTER_SITE_URL")
        self.app_name = os.getenv("OPENROUTER_APP_NAME")

        logger.info(f"OpenRouter embedding: model={self.model}, dims={self.dimensions}, base={self.base_url}")

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # optional but recommended by OpenRouter
        if self.site_url:
            h["HTTP-Referer"] = self.site_url
        if self.app_name:
            h["X-Title"] = self.app_name
        return h

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Returns list of embeddings (float lists) in the same order.
        Retries on transient failures. If 'dimensions' is rejected, retries without it.
        """
        if not texts:
            return []

        out: List[List[float]] = []
        for i in range(0, len(texts), OPENROUTER_BATCH):
            batch = texts[i : i + OPENROUTER_BATCH]
            emb = self._embed_batch(batch)
            out.extend(emb)
        return out

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"

        payload_with_dims = {
            "model": self.model,
            "input": batch,
            "dimensions": self.dimensions,
        }
        payload_no_dims = {
            "model": self.model,
            "input": batch,
        }

        def do_call(payload: Dict[str, Any]) -> Dict[str, Any]:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"OpenRouter embeddings HTTP {r.status_code}: {r.text[:500]}")
            return r.json()

        # retry wrapper
        last_err = None
        for attempt in range(1, 6):
            try:
                try:
                    data = do_call(payload_with_dims)
                except Exception as e_dims:
                    # some providers/models may not accept 'dimensions' -> retry without
                    msg = str(e_dims).lower()
                    if "dimensions" in msg or "dimension" in msg:
                        data = do_call(payload_no_dims)
                    else:
                        raise

                rows = data.get("data") or []
                if len(rows) != len(batch):
                    raise RuntimeError(f"Embedding response size mismatch: got {len(rows)} expected {len(batch)}")

                embeddings = []
                for row in rows:
                    embeddings.append(row.get("embedding"))
                if not all(isinstance(x, list) for x in embeddings):
                    raise RuntimeError("Invalid embeddings format from OpenRouter.")
                return embeddings

            except Exception as e:
                last_err = e
                if attempt == 5:
                    break
                sleep = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"OpenRouter embed failed (attempt {attempt}/5): {e} | retry in {sleep:.1f}s")
                time.sleep(sleep)

        raise RuntimeError(f"OpenRouter embedding failed after retries: {last_err}")


class PineconeManager:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable gerekli")

        # -----------------------------------------------------------------------------
        self.pc = Pinecone(api_key=api_key)

        # -----------------------------------------------------------------------------
        self.embedder = OpenRouterEmbedder()

        # -----------------------------------------------------------------------------
        self.bm25: Optional[BM25Encoder] = None

        self.index = None
        self.index_host: Optional[str] = None

    def _get_index_names(self) -> List[str]:
        idx_list = self.pc.list_indexes()

        names = []
        try:
            for item in idx_list:
                if isinstance(item, dict) and "name" in item:
                    names.append(item["name"])
                elif isinstance(item, str):
                    names.append(item)
        except TypeError:
            pass

        names_attr = getattr(idx_list, "names", None)
        if callable(names_attr):
            try:
                names.extend(list(names_attr()))
            except Exception:
                pass

        return sorted(list(set([n for n in names if n])))

    def create_index_if_not_exists(self):
        existing = set(self._get_index_names())

        if PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating index: {PINECONE_INDEX_NAME} (dim={EMBEDDING_DIMENSION}, metric={PINECONE_METRIC})")

            cloud = os.getenv("PINECONE_CLOUD")
            region = os.getenv("PINECONE_REGION")
            environment = os.getenv("PINECONE_ENVIRONMENT")

            if cloud and region:
                spec = ServerlessSpec(cloud=cloud, region=region)
            elif environment:
                pod_type = os.getenv("PINECONE_POD_TYPE", "p1.x1")
                pods = int(os.getenv("PINECONE_PODS", "1"))
                spec = PodSpec(environment=environment, pod_type=pod_type, pods=pods)
            else:
                # Default to a common serverless region if nothing provided.
                cloud = cloud or "aws"
                region = region or "us-east-1"
                logger.warning(
                    "PINECONE_CLOUD/PINECONE_REGION not set. Defaulting to %s/%s. "
                    "You can override in .env.",
                    cloud,
                    region,
                )
                spec = ServerlessSpec(cloud=cloud, region=region)

            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric=PINECONE_METRIC,
                spec=spec,
                deletion_protection="disabled",
            )
        else:
            logger.info(f"Index zaten mevcut: {PINECONE_INDEX_NAME}")

        # Validate index config (dimension/metric)
        desc = self.pc.describe_index(PINECONE_INDEX_NAME)
        desc_dim = desc.get("dimension")
        desc_metric = desc.get("metric")
        if desc_dim is not None and int(desc_dim) != int(EMBEDDING_DIMENSION):
            raise RuntimeError(
                f"Index dimension mismatch! index={desc_dim}, code={EMBEDDING_DIMENSION}. "
                f"Fix: Use a new PINECONE_INDEX_NAME (or delete and recreate the existing index)."
            )
        if desc_metric and str(desc_metric).lower() != str(PINECONE_METRIC).lower():
            raise RuntimeError(
                f"Index metric mismatch! index={desc_metric}, code={PINECONE_METRIC}. "
                f"For hybrid (sparse+dense), use dotproduct. Create with a new index name."
            )

        # Host resolve
        host_env = os.getenv("PINECONE_INDEX_HOST")
        if host_env:
            self.index_host = host_env
            logger.info("Index host loaded from env (PINECONE_INDEX_HOST).")
        else:
            cached = _load_cached_host(PINECONE_INDEX_NAME)
            if cached:
                self.index_host = cached
                logger.info("Index host loaded from cache (db/pinecone_index_host.json).")
            else:
                logger.info("Index host fetched via describe_index (first setup).")
                while True:
                    desc2 = self.pc.describe_index(PINECONE_INDEX_NAME)
                    ready = bool(desc2.get("status", {}).get("ready"))
                    if ready:
                        self.index_host = desc2.get("host")
                        break
                    time.sleep(1)

                if not self.index_host:
                    raise RuntimeError("Index host not found (describe_index host empty).")

                _save_cached_host(PINECONE_INDEX_NAME, self.index_host)
                logger.info("Index host saved to cache (db/pinecone_index_host.json).")

        # -----------------------------------------------------------------------------
        self.index = self.pc.Index(host=self.index_host)

        stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")

    # ---------------------------
    # Hybrid: BM25 setup
    # ---------------------------
    def ensure_bm25_ready(self, duckdb_path: Path):
        """
        BM25 requires a fit over a corpus (static DF). We cache params to BM25_PARAMS_PATH.
        """
        if self.bm25 is not None and not BM25_REFIT:
            return

        bm25 = BM25Encoder()

        if BM25_PARAMS_PATH.exists() and not BM25_REFIT:
            logger.info(f"Loading BM25 params: {BM25_PARAMS_PATH}")
            bm25.load(str(BM25_PARAMS_PATH))
            self.bm25 = bm25
            return

        logger.info("Fitting BM25 (initial setup / refit). This may take 1-2 minutes...")
        corpus = self._build_bm25_corpus_from_duckdb(duckdb_path, max_docs=BM25_FIT_MAX_DOCS)

        if not corpus:
            raise RuntimeError("BM25 corpus is empty. Is the DuckDB 'transactions' table populated")

        bm25.fit(corpus)
        BM25_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        bm25.dump(str(BM25_PARAMS_PATH))
        logger.info(f"BM25 params kaydedildi: {BM25_PARAMS_PATH}")

        self.bm25 = bm25

    def _build_bm25_corpus_from_duckdb(self, duckdb_path: Path, max_docs: int) -> List[str]:
        con = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            total = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            take = min(total, max_docs)
            logger.info(f"BM25 fit corpus: using {take:,}/{total:,} documents")

            corpus: List[str] = []
            offset = 0
            batch = 2000

            cols = ["merchant_norm", "description", "category", "subcategory", "amount", "direction", "channel"]
            while offset < take:
                lim = min(batch, take - offset)
                q = f"""
                    SELECT {", ".join(cols)}
                    FROM transactions
                    ORDER BY tx_id
                    LIMIT {lim} OFFSET {offset}
                """
                rows = con.execute(q).fetchall()
                for r in rows:
                    row = dict(zip(cols, r))
                    corpus.append(self._create_embed_text(row))
                offset += lim

            return corpus
        finally:
            con.close()

    # ---------------------------
    # Embedding + text building
    # ---------------------------
    def _embed(self, texts: List[str]) -> List[List[float]]:
        embs = self.embedder.embed_texts(texts)
        # For dotproduct metric, normalize dense vectors
        return [_l2_normalize(e) for e in embs]

    def _create_embed_text(self, row: dict) -> str:
        parts = []
        if row.get("merchant_norm"):
            parts.append(f"Merchant: {row['merchant_norm']}")
        if row.get("description"):
            parts.append(f"Description: {row['description']}")
        if row.get("category"):
            parts.append(f"Kategori: {row.get('category')}")
        if row.get("subcategory"):
            parts.append(f"Alt kategori: {row.get('subcategory')}")

        amount = _safe_float(row.get("amount", 0))
        direction = (row.get("direction") or "").strip()
        if direction == "expense":
            parts.append(f"Gider: {abs(amount):.2f} TL")
        elif direction == "income":
            parts.append(f"Gelir: {amount:.2f} TL")

        if row.get("channel"):
            parts.append(f"Kanal: {row.get('channel')}")

        return " | ".join([p for p in parts if p])

    def _create_metadata(self, row: dict) -> Dict[str, Any]:
        tags_arr = row.get("tags_arr")
        if tags_arr is None:
            tags_arr = []
        elif isinstance(tags_arr, str):
            tags_arr = [t.strip() for t in tags_arr.split(",") if t.strip()]
        elif not isinstance(tags_arr, list):
            tags_arr = [str(tags_arr)]

        tags_final = row.get("tags_final")
        if tags_final is None:
            tags_final = []
        elif isinstance(tags_final, str):
            tags_final = [t.strip() for t in tags_final.split(",") if t.strip()]
        elif not isinstance(tags_final, list):
            tags_final = [str(tags_final)]

        metadata: Dict[str, Any] = {
            "tx_id": str(row.get("tx_id", ""))[:100],
            "description": str(row.get("description", ""))[:500],  # CRITICAL: Enable vector search on descriptions
            "file_id": str(row.get("file_id", ""))[:50],
            "tenant_id": str(row.get("tenant_id", ""))[:100],
            "user_id": str(row.get("user_id", ""))[:100],
            "currency": str(row.get("currency", ""))[:10],
            "amount": _safe_float(row.get("amount", 0)),
            "direction": str(row.get("direction", ""))[:20],
            "category": str(row.get("category", ""))[:50],
            "subcategory": str(row.get("subcategory", ""))[:50],
            "category_final": str(row.get("category_final", ""))[:50],
            "subcategory_final": str(row.get("subcategory_final", ""))[:50],
            "confidence": _safe_float(row.get("confidence", 0)),
            "merchant_norm": str(row.get("merchant_norm", ""))[:100],
            "merchant_key": str(row.get("merchant_key", ""))[:100],
            "channel": str(row.get("channel", ""))[:30],
            "transaction_type": str(row.get("transaction_type", ""))[:30],
            "month_year": str(row.get("month_year", ""))[:10],
            "tags_arr": [str(x)[:64] for x in tags_arr[:64]],
            "tags_final": [str(x)[:64] for x in tags_final[:64]],
        }

        dt = _parse_datetime(row.get("date_time"))
        if dt:
            metadata["date_epoch"] = int(dt.timestamp())
            metadata["date_str"] = dt.strftime("%Y-%m-%d")

        return {k: v for k, v in metadata.items() if v is not None and v != ""}

    # ---------------------------
    # Upsert (Dense + Sparse BM25)
    # ---------------------------
    def upsert_transactions(self, transactions: List[dict], namespace: str = "__default__"):
        if not self.index:
            raise ValueError("Index is not ready yet")
        if not self.bm25:
            raise ValueError("BM25 not ready. Call ensure_bm25_ready first.")

        texts = [self._create_embed_text(tx) for tx in transactions]

        dense_vectors = self._embed(texts)
        sparse_vectors = self.bm25.encode_documents(texts)
        if isinstance(sparse_vectors, dict):
            sparse_vectors = [sparse_vectors for _ in texts]

        vectors = []
        for tx, dense, sparse in zip(transactions, dense_vectors, sparse_vectors):
            vectors.append(
                {
                    "id": str(tx["tx_id"]),
                    "values": dense,
                    "sparse_values": sparse,   # <-- for upsert
                    "metadata": self._create_metadata(tx),
                }
            )

        for i in range(0, len(vectors), BATCH_SIZE):
            chunk = vectors[i : i + BATCH_SIZE]
            self._retry(lambda: self.index.upsert(vectors=chunk, namespace=namespace))

    # ---------------------------
    # Query (Hybrid BM25 + Dense)
    # ---------------------------
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: str = "__default__",
        alpha: float = HYBRID_ALPHA_DEFAULT,
    ) -> List[Dict[str, Any]]:
        if not self.index:
            raise ValueError("Index is not ready yet")
        if not self.bm25:
            raise ValueError("BM25 not ready. Call ensure_bm25_ready first.")

        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in range 0..1")

        dense_q = self._embed([query_text])[0]                 # normalized
        sparse_q = self.bm25.encode_queries(query_text)

        # scale dense/sparse contributions (convex combo)
        hybrid_dense, hybrid_sparse = hybrid_convex_scale(dense_q, sparse_q, alpha=alpha)

        pinecone_filter = _normalize_filter(filter_dict)

        res = self.index.query(
            vector=hybrid_dense,
            sparse_vector=hybrid_sparse,     # <-- for query
            top_k=top_k,
            namespace=namespace,
            filter=pinecone_filter,
            include_metadata=True,
        )

        matches = getattr(res, "matches", None)
        if matches is None and isinstance(res, dict):
            matches = res.get("matches", [])
        matches = matches or []

        out = []
        for m in matches:
            if isinstance(m, dict):
                out.append({"id": m.get("id"), "score": m.get("score"), "metadata": m.get("metadata")})
            else:
                out.append(
                    {
                        "id": getattr(m, "id", None),
                        "score": getattr(m, "score", None),
                        "metadata": getattr(m, "metadata", None),
                    }
                )
        return out

    def delete_by_file(self, file_id: str, namespace: str = "__default__"):
        if not self.index:
            raise ValueError("Index is not ready yet")
        self._retry(lambda: self.index.delete(namespace=namespace, filter={"file_id": {"$eq": file_id}}))
        logger.info(f"file_id={file_id} vectors deleted (namespace={namespace})")

    @staticmethod
    def _retry(fn, retries: int = 5, base_sleep: float = 0.5):
        for attempt in range(1, retries + 1):
            try:
                return fn()
            except Exception as e:
                if attempt == retries:
                    raise
                sleep = base_sleep * (2 ** (attempt - 1))
                logger.warning(f"Pinecone call failed (attempt {attempt}/{retries}): {e} | retry in {sleep:.1f}s")
                time.sleep(sleep)


def load_transactions_to_pinecone(duckdb_path: Path, pinecone_manager: PineconeManager, namespace: str = "__default__"):
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        total_count = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        logger.info(f"Total {total_count:,} transactions will be loaded (namespace={namespace})")

        offset = 0
        processed = 0

        columns = [
            "tx_id", "file_id", "tenant_id", "user_id", "currency",
            "date_time", "value_date", "amount", "balance", "direction",
            "description", "merchant_norm", "merchant_key",
            "category", "subcategory", "category_final", "subcategory_final",
            "tags", "tags_arr", "tags_final", "confidence",
            "channel", "transaction_type", "transaction_code", "month_year",
        ]

        while offset < total_count:
            query = f"""
                SELECT {', '.join(columns)}
                FROM transactions
                ORDER BY tx_id
                LIMIT {BATCH_SIZE} OFFSET {offset}
            """
            result = con.execute(query)
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            if not rows:
                break

            pinecone_manager.upsert_transactions(rows, namespace=namespace)

            processed += len(rows)
            offset += BATCH_SIZE

            progress = (processed / total_count) * 100
            logger.info(f"Progress: {processed:,}/{total_count:,} ({progress:.1f}%)")
            time.sleep(0.05)

        logger.info(f" Total {processed:,} transactions uploaded to Pinecone (namespace={namespace})")
    finally:
        con.close()


def test_pinecone_search(pinecone_manager: PineconeManager, namespace: str = "__default__"):
    print("\n" + "=" * 60)
    print(" PINECONE HYBRID SEARCH TEST (BM25 + Dense)")
    print("=" * 60)

    test_queries = [
        ("grocery shopping", None),
        ("bill payment", {"direction": "expense"}),
        ("Netflix benzeri", {"category": "utilities", "subcategory": "tv_streaming"}),
        ("large transfer", {"direction": "transfer"}),
    ]

    for query, filters in test_queries:
        print(f"\n Query: '{query}' (filters: {filters})")
        results = pinecone_manager.query(query, top_k=3, filter_dict=filters, namespace=namespace, alpha=HYBRID_ALPHA_DEFAULT)

        for i, r in enumerate(results, 1):
            meta = r.get("metadata") or {}
            print(f"   {i}. [{(r.get('score') or 0):.3f}] {meta.get('merchant_norm', 'N/A')}")
            print(f"      {meta.get('amount', 0):.2f} TL | {meta.get('category', '')} | {meta.get('date_str', '')}")


def main():
    print(" Statement Copilot Pinecone Setup (OpenRouter Embeddings + Hybrid BM25)")
    print("=" * 60)

    if not os.getenv("PINECONE_API_KEY"):
        print(" PINECONE_API_KEY environment variable not found!")
        print("   export PINECONE_API_KEY='your-api-key'")
        return

    if not os.getenv("OPENROUTER_API_KEY"):
        print(" OPENROUTER_API_KEY environment variable not found!")
        print("   export OPENROUTER_API_KEY='your-openrouter-key'")
        return

    if not DB_PATH.exists():
        logger.error(f"DuckDB not found: {DB_PATH}")
        logger.error("Run scripts/setup/duckdb_setup.py first")
        return

    namespace = os.getenv("PINECONE_NAMESPACE", "__default__")

    try:
        print("\n Starting Pinecone Manager...")
        pm = PineconeManager()

        print("\n Checking index / preparing host...")
        pm.create_index_if_not_exists()

        print("\n Preparing BM25 (fit/load)...")
        pm.ensure_bm25_ready(DB_PATH)

        print("\n Loading transactions (dense + sparse)...")
        load_transactions_to_pinecone(DB_PATH, pm, namespace=namespace)

        test_pinecone_search(pm, namespace=namespace)

        print("\n" + "=" * 60)
        print(" Pinecone setup completed!")
        print("=" * 60)
    except Exception as e:
        logger.error(f"Hata: {e}")
        raise


if __name__ == "__main__":
    main()