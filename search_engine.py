import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import difflib

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class MedicalDevice:
    id: str
    name: str
    description: str
    use: str

    def full_text(self) -> str:
        """Combine all searchable text fields."""
        return f"{self.name}. {self.description}. {self.use}"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def get_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")

        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        if not self.client.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

    def get_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]

    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0]


class MedicalDeviceSearch:
    """Smart search for medical devices combining semantic and keyword approaches."""

    def __init__(
        self,
        devices_file: Path,
        embedding_provider: Optional[EmbeddingProvider] = None,
        cache_file: Optional[Path] = None
    ):
        self.devices_file = Path(devices_file)  # Ensure it's a Path object
        self.devices = self._load_devices()
        self.embedding_provider = embedding_provider
        self.cache_file = cache_file or self.devices_file.parent / "embeddings_cache.json"
        self.embeddings: Optional[List[List[float]]] = None

    def _load_devices(self) -> List[MedicalDevice]:
        """Load devices from JSON file."""
        with open(self.devices_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_data = []
        for item in data:
            if isinstance(item, dict):
                item = {k: v for k, v in item.items() if k != "image"}
            cleaned_data.append(item)

        return [MedicalDevice(**item) for item in cleaned_data]

    def _ensure_embeddings(self) -> None:
        """Load or create embeddings cache."""
        if self.embeddings is not None:
            return

        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if len(cached) == len(self.devices):
                    self.embeddings = cached
                    return
            except (json.JSONDecodeError, KeyError):
                pass  # Rebuild cache if invalid

        if self.embedding_provider is None:
            raise RuntimeError("Embedding provider required for semantic search")

        # Create new embeddings
        texts = [device.full_text() for device in self.devices]
        self.embeddings = self.embedding_provider.get_embeddings(texts)

        # Cache embeddings
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.embeddings, f)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if np is None:
            raise ImportError("NumPy required for similarity calculation")

        a_vec = np.array(a, dtype=float)
        b_vec = np.array(b, dtype=float)

        norm_a = np.linalg.norm(a_vec)
        norm_b = np.linalg.norm(b_vec)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a_vec, b_vec) / (norm_a * norm_b))

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        self._ensure_embeddings()

        if self.embedding_provider is None:
            raise RuntimeError("Embedding provider required for semantic search")

        query_embedding = self.embedding_provider.get_embedding(query)

        results = []
        for device, embedding in zip(self.devices, self.embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append({
                "device": device,
                "similarity_score": similarity,
                "search_type": "semantic"
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform keyword search with fuzzy matching on names."""
        query_lower = query.lower()
        results = []

        for device in self.devices:
            # Exact name match gets highest score
            name_lower = device.name.lower()
            if query_lower == name_lower:
                score = 1.0
                match_type = "exact_name"
            elif query_lower in name_lower:
                score = 0.8
                match_type = "partial_name"
            else:
                # Fuzzy matching on name
                matcher = difflib.SequenceMatcher(None, query_lower, name_lower)
                name_score = matcher.ratio()

                # Also check if query appears in description or use
                desc_score = 1.0 if query_lower in device.description.lower() else 0.0
                use_score = 1.0 if query_lower in device.use.lower() else 0.0

                # Combine scores (weighted towards name matching)
                score = (name_score * 0.7) + (desc_score * 0.2) + (use_score * 0.1)
                match_type = "fuzzy_name" if name_score > 0.3 else "keyword_match"

            if score > 0:
                results.append({
                    "device": device,
                    "similarity_score": score,
                    "search_type": match_type
                })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def combined_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results."""
        results = []

        # Try semantic search first
        try:
            semantic_results = self.semantic_search(query, top_k=top_k)
            results.extend(semantic_results)
        except Exception:
            # Fall back to keyword search if semantic fails
            pass

        # Always include keyword results
        keyword_results = self.keyword_search(query, top_k=top_k)
        results.extend(keyword_results)

        # Remove duplicates and sort by score
        seen_ids = set()
        unique_results = []
        for result in results:
            device_id = result["device"].id
            if device_id not in seen_ids:
                seen_ids.add(device_id)
                unique_results.append(result)

        unique_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return unique_results[:top_k]

    def search(self, query: str, method: str = "combined", top_k: int = 3) -> List[Dict[str, Any]]:
        """Main search method."""
        if method == "semantic":
            return self.semantic_search(query, top_k)
        elif method == "keyword":
            return self.keyword_search(query, top_k)
        else:  # combined
            return self.combined_search(query, top_k)


def format_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for display."""
    if not results:
        return "No matching devices found."

    output = []
    for i, result in enumerate(results, 1):
        device = result["device"]
        score = result["similarity_score"]
        search_type = result["search_type"]

        output.append(f"{i}. **{device.name}** ({device.id})")
        output.append(f"   *Match type:* {search_type.replace('_', ' ').title()}")
        output.append(f"   *Confidence:* {score:.3f}")
        output.append(f"   *Description:* {device.description}")
        output.append(f"   *Uses:* {device.use}")
        output.append("")

    return "\n".join(output)