#!/usr/bin/env python3
"""
Test script to verify OpenAI API key is working.
Run this after setting your OPENAI_API_KEY.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def test_api_key():
    """Test that the API key works with OpenAI."""
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("❌ No OPENAI_API_KEY found!")
        print("\nTo set your API key:")
        print("1. Get your key from: https://platform.openai.com/api-keys")
        print("2. Run: export OPENAI_API_KEY='your_key_here'")
        print("3. Or create .env file with: OPENAI_API_KEY=your_key_here")
        return False

    print(f"✅ API key found (starts with: {api_key[:10]}...)")

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        # Test with a simple embedding request
        print("🔄 Testing API connection...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=["test"]
        )

        if response.data and len(response.data) > 0:
            print("✅ API key is working! Semantic search is now available.")
            return True
        else:
            print("❌ API returned empty response")
            return False

    except Exception as e:
        print(f"❌ API key test failed: {e}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- No internet connection")
        print("- OpenAI service issues")
        print("- Insufficient credits")
        return False

def test_search_engine():
    """Test that the search engine can use semantic search."""
    try:
        from search_engine import MedicalDeviceSearch, OpenAIEmbeddingProvider

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Cannot test search engine without API key")
            return

        print("\n🔄 Testing search engine with semantic search...")

        embedding_provider = OpenAIEmbeddingProvider(api_key=api_key)
        searcher = MedicalDeviceSearch(Path("devices.json"), embedding_provider)

        # Test semantic search
        results = searcher.search("device for measuring oxygen", method="semantic", top_k=1)

        if results:
            device = results[0]["device"]
            score = results[0]["similarity_score"]
            print(f"✅ Semantic search working! Found: {device.name} (score: {score:.3f})")
        else:
            print("❌ Semantic search returned no results")

    except Exception as e:
        print(f"❌ Search engine test failed: {e}")

if __name__ == "__main__":
    print("🧪 OpenAI API Key Test")
    print("=" * 40)

    if test_api_key():
        test_search_engine()

    print("\n" + "=" * 40)
    print("If tests pass, you can now run: streamlit run app.py")