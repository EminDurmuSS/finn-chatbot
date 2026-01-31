"""
Statement Copilot - Vector Store
================================
Pinecone hybrid search (BM25 + Dense vectors).

bunq Alignment: Hybrid search catches both exact terms and semantics.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ..config import settings
from .embeddings import get_embeddings_client, get_bm25_encoder

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID SEARCH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def hybrid_convex_scale(
    dense: List[float],
    sparse: Dict[str, List],
    alpha: float = 0.7
) -> Tuple[List[float], Dict[str, List]]:
    """
    Apply convex scaling to hybrid search vectors.
    
    Args:
        dense: Dense embedding vector
        sparse: Sparse vector dict with 'indices' and 'values'
        alpha: Weight for dense (0=sparse only, 1=dense only)
        
    Returns:
        Scaled (dense, sparse) vectors
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    
    # Scale dense by alpha
    scaled_dense = [v * alpha for v in dense]
    
    # Scale sparse by (1 - alpha)
    scaled_sparse = {
        "indices": sparse.get("indices", []),
        "values": [v * (1 - alpha) for v in sparse.get("values", [])]
    }
    
    return scaled_dense, scaled_sparse


# ═══════════════════════════════════════════════════════════════════════════════
# PINECONE VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Pinecone vector store with hybrid search.
    
    Features:
    - Hybrid search (BM25 + dense embeddings)
    - Metadata filtering
    - Batch upsert
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        index_host: Optional[str] = None,
        namespace: Optional[str] = None
    ):
        self.api_key = api_key or settings.pinecone_api_key
        self.index_name = index_name or settings.pinecone_index_name
        self.index_host = index_host or settings.pinecone_index_host
        self.namespace = namespace or settings.pinecone_namespace
        
        self.embeddings = get_embeddings_client()
        self.bm25 = get_bm25_encoder()
        
        self._index = None
        
        if not self.api_key:
            logger.warning("PINECONE_API_KEY not set. Vector search will use mock.")
    
    @property
    def index(self):
        """Lazy load Pinecone index"""
        if self._index is None and self.api_key:
            try:
                from pinecone import Pinecone
                
                pc = Pinecone(api_key=self.api_key)
                
                if self.index_host:
                    self._index = pc.Index(host=self.index_host)
                else:
                    self._index = pc.Index(self.index_name)
                    
                logger.info(f"Connected to Pinecone index: {self.index_name}")
                
            except Exception as e:
                logger.error(f"Pinecone connection error: {e}")
                raise
        
        return self._index
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search for transactions.
        
        Args:
            query: Search query
            top_k: Number of results
            alpha: Dense weight (0=sparse, 1=dense)
            filters: Pinecone metadata filters
            include_metadata: Include metadata in results
            
        Returns:
            List of matches with scores
        """
        if not self.api_key:
            logger.warning("Vector search unavailable - returning empty results")
            return []
        
        try:
            # Generate dense embedding
            dense = self.embeddings.embed(query)
            
            # Generate sparse BM25 vector
            sparse = self.bm25.encode(query)
            
            # Apply hybrid scaling
            scaled_dense, scaled_sparse = hybrid_convex_scale(dense, sparse, alpha)
            
            # Build query
            query_params = {
                "vector": scaled_dense,
                "sparse_vector": scaled_sparse,
                "top_k": top_k,
                "include_metadata": include_metadata,
                "namespace": self.namespace,
            }
            
            if filters:
                query_params["filter"] = filters
            
            # Execute search
            results = self.index.query(**query_params)

            # Support both dict and object response shapes (Pinecone SDK variants)
            raw_matches = []
            if isinstance(results, dict):
                raw_matches = results.get("matches", [])
            else:
                raw_matches = getattr(results, "matches", []) or []

            # Format results
            matches = []
            for match in raw_matches:
                if isinstance(match, dict):
                    match_id = match.get("id")
                    score = match.get("score")
                    metadata = match.get("metadata", {})
                else:
                    match_id = getattr(match, "id", None)
                    score = getattr(match, "score", None)
                    metadata = getattr(match, "metadata", {}) or {}

                if match_id is None:
                    continue

                matches.append({
                    "tx_id": match_id,
                    "score": score,
                    "metadata": metadata
                })
            
            logger.debug(f"Vector search: query='{query[:50]}...', matches={len(matches)}")
            
            return matches
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def search_dense_only(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Dense-only search (no BM25)"""
        return self.search(query, top_k, alpha=1.0, filters=filters)
    
    def search_sparse_only(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Sparse-only search (BM25 only)"""
        return self.search(query, top_k, alpha=0.0, filters=filters)
    
    def upsert(
        self,
        tx_id: str,
        text: str,
        metadata: Dict[str, Any]
    ):
        """
        Upsert a single transaction.
        
        Args:
            tx_id: Transaction ID
            text: Text to embed (description + merchant + category)
            metadata: Transaction metadata
        """
        if not self.api_key:
            return
        
        try:
            dense = self.embeddings.embed(text)
            sparse = self.bm25.encode(text)
            
            self.index.upsert(
                vectors=[{
                    "id": tx_id,
                    "values": dense,
                    "sparse_values": sparse,
                    "metadata": metadata
                }],
                namespace=self.namespace
            )
            
        except Exception as e:
            logger.error(f"Upsert error: {e}")
            raise
    
    def upsert_batch(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        """
        Batch upsert transactions.
        
        Args:
            items: List of {"tx_id", "text", "metadata"}
            batch_size: Batch size for upsert
        """
        if not self.api_key or not items:
            return
        
        # Generate embeddings in batches
        texts = [item["text"] for item in items]
        dense_embeddings = self.embeddings.embed_batch(texts)
        
        # Generate sparse vectors
        sparse_vectors = [self.bm25.encode(text) for text in texts]
        
        # Prepare vectors
        vectors = []
        for i, item in enumerate(items):
            vectors.append({
                "id": item["tx_id"],
                "values": dense_embeddings[i],
                "sparse_values": sparse_vectors[i],
                "metadata": item.get("metadata", {})
            })
        
        # Upsert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=self.namespace)
            except Exception as e:
                logger.error(f"Batch upsert error: {e}")
                raise
        
        logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
    
    def delete(self, tx_ids: List[str]):
        """Delete transactions by ID"""
        if not self.api_key or not tx_ids:
            return
        
        try:
            self.index.delete(ids=tx_ids, namespace=self.namespace)
            logger.debug(f"Deleted {len(tx_ids)} vectors from Pinecone")
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.api_key:
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "namespaces": stats.get("namespaces", {}),
                "dimension": stats.get("dimension", 0)
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK VECTOR STORE (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

class MockVectorStore:
    """
    Mock vector store for testing without Pinecone.
    Uses simple in-memory storage with cosine similarity.
    """
    
    def __init__(self):
        self.vectors: Dict[str, Dict] = {}
        self.embeddings = get_embeddings_client()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Simple cosine similarity search"""
        if not self.vectors:
            return []
        
        query_embedding = self.embeddings.embed(query)
        query_arr = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0:
            return []
        query_arr = query_arr / query_norm
        
        scores = []
        for tx_id, data in self.vectors.items():
            # Apply filters
            if filters:
                metadata = data.get("metadata", {})
                match = True
                for key, value in filters.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Cosine similarity
            vec_arr = np.array(data["values"], dtype=float)
            vec_norm = np.linalg.norm(vec_arr)
            if vec_norm == 0:
                continue
            vec_arr = vec_arr / vec_norm
            similarity = float(np.dot(query_arr, vec_arr))
            scores.append((tx_id, similarity, data.get("metadata", {})))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        results = []
        for tx_id, score, metadata in scores[:top_k]:
            results.append({
                "tx_id": tx_id,
                "score": float(score),
                "metadata": metadata if include_metadata else {}
            })
        
        return results
    
    def upsert(self, tx_id: str, text: str, metadata: Dict[str, Any]):
        """Store vector in memory"""
        embedding = self.embeddings.embed(text)
        self.vectors[tx_id] = {
            "values": embedding,
            "metadata": metadata
        }
    
    def delete(self, tx_ids: List[str]):
        """Delete vectors from memory"""
        for tx_id in tx_ids:
            self.vectors.pop(tx_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        return {"total_vectors": len(self.vectors)}


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create vector store singleton"""
    global _vector_store
    if _vector_store is None:
        if settings.pinecone_api_key:
            _vector_store = VectorStore()
        else:
            logger.warning("Using MockVectorStore (no Pinecone API key)")
            _vector_store = MockVectorStore()
    return _vector_store