#!/usr/bin/env python3
"""
Quick test script for the medical device search engine.
Run this to test functionality without the full Streamlit app.
"""

import os
from pathlib import Path
from search_engine import MedicalDeviceSearch, OpenAIEmbeddingProvider

def test_keyword_search():
    """Test keyword search functionality."""
    print("🔍 Testing Keyword Search")
    print("=" * 50)

    searcher = MedicalDeviceSearch(Path("devices.json"), embedding_provider=None)

    test_queries = [
        "stethoscope",  # exact match
        "stethascop",   # typo
        "heart",        # partial match in description
        "surgery",      # match in use field
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = searcher.search(query, method="keyword", top_k=2)
        for result in results:
            device = result["device"]
            print(f"  → {device.name} ({result['similarity_score']:.3f}) [{result['search_type']}]")

def test_semantic_search():
    """Test semantic search if API key is available."""
    print("\n🧠 Testing Semantic Search")
    print("=" * 50)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OpenAI API key not found. Skipping semantic search test.")
        return

    try:
        embedding_provider = OpenAIEmbeddingProvider(api_key=api_key)
        searcher = MedicalDeviceSearch(Path("devices.json"), embedding_provider)

        test_queries = [
            "device for listening to heart sounds",
            "machine that measures oxygen in blood",
            "tool for cutting during surgery",
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = searcher.search(query, method="semantic", top_k=2)
            for result in results:
                device = result["device"]
                print(f"  → {device.name} ({result['similarity_score']:.3f}) [semantic]")

    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")

def test_combined_search():
    """Test combined search approach."""
    print("\n🔄 Testing Combined Search")
    print("=" * 50)

    # Use keyword-only for this test (no API key needed)
    searcher = MedicalDeviceSearch(Path("devices.json"), embedding_provider=None)

    test_queries = [
        "oxygen monitor",
        "cardiac device",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = searcher.search(query, method="combined", top_k=3)
        for result in results:
            device = result["device"]
            print(f"  → {device.name} ({result['similarity_score']:.3f}) [{result['search_type']}]")

if __name__ == "__main__":
    print("🏥 Medical Device Search Engine - Test Suite")
    print("=" * 60)

    test_keyword_search()
    test_semantic_search()
    test_combined_search()

    print("\n✅ Test suite completed!")
    print("\nTo run the full demo app:")
    print("  streamlit run app.py")