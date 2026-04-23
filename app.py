import streamlit as st
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv not installed, skip

import streamlit_authenticator as stauth
from search_engine import MedicalDeviceSearch, OpenAIEmbeddingProvider, format_results

# Authentication configuration
# Load from secrets if available (for deployment), otherwise use defaults
try:
    auth_config = {
        "credentials": {
            "usernames": {
                "robbie": {
                    "email": st.secrets.auth.get("robbie_email", "brennan.robert.m@gmail.com"),
                    "name": st.secrets.auth.get("robbie_name", "Robbie Brennan"),
                    "password": st.secrets.auth["robbie_password"]
                },
                "flo": {
                    "email": st.secrets.auth.get("flo_email", "fsctsai@gmail.com"),
                    "name": st.secrets.auth.get("flo_name", "Florence Tsai"),
                    "password": st.secrets.auth["flo_password"]
                }
                # Add more users as needed
            }
        },
        "cookie": {
            "expiry_days": st.secrets.auth.get("cookie_expiry_days", 30),
            "key": st.secrets.auth.get("cookie_key", "medical_search_auth"),
            "name": st.secrets.auth.get("cookie_name", "medical_search_cookie")
        },
        "preauthorized": {
            "emails": [st.secrets.auth.get("robbie_email", "brennan.robert.m@gmail.com"), st.secrets.auth.get("flo_email", "fsctsai@gmail.com")]
        }
    }
except (KeyError, AttributeError):
    # Fallback for local development
    st.warning("Secrets not found. Using demo credentials. For production, set up secrets.")
    auth_config = {
        "credentials": {
            "usernames": {
                "demo": {
                    "email": "demo@example.com",
                    "name": "Demo User",
                    "password": stauth.Hasher(['demo']).generate()[0]  # Hash for 'demo'
                }
            }
        },
        "cookie": {
            "expiry_days": 30,
            "key": "medical_search_auth",
            "name": "medical_search_cookie"
        },
        "preauthorized": {
            "emails": ["demo@example.com"]
        }
    }

authenticator = stauth.Authenticate(
    auth_config['credentials'],
    auth_config['cookie']['name'],
    auth_config['cookie']['key'],
    auth_config['cookie']['expiry_days']
)

# Page configuration
st.set_page_config(
    page_title="Medical Device Smart Search",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "search_engine" not in st.session_state:
    st.session_state.search_engine = None

def initialize_search_engine() -> Optional[MedicalDeviceSearch]:
    """Initialize the search engine with error handling."""
    try:
        devices_file = Path(__file__).parent / "devices.json"

        # Try to initialize with OpenAI embeddings
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")
        if api_key:
            embedding_provider = OpenAIEmbeddingProvider(api_key=api_key)
            return MedicalDeviceSearch(devices_file, embedding_provider)
        else:
            # Fallback to keyword-only search
            st.warning("⚠️ OpenAI API key not found. Using keyword search only.")
            return MedicalDeviceSearch(devices_file, embedding_provider=None)

    except Exception as e:
        st.error(f"Error initializing search engine: {e}")
        return None

def main():
    # Authentication
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    if authentication_status == False:
        st.error('Username/password is incorrect')
        return
    
    if authentication_status == None:
        st.warning('Please enter your username and password')
        return
    
    if authentication_status:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{name}*')
        
        # Rest of the app
        st.title("🏥 Medical Device Smart Search")
        st.markdown("Search through medical device inventory using intelligent matching")

    # Initialize search engine
    if st.session_state.search_engine is None:
        with st.spinner("Loading search engine..."):
            st.session_state.search_engine = initialize_search_engine()

    search_engine = st.session_state.search_engine
    if search_engine is None:
        st.error("Failed to initialize search engine. Please check your setup.")
        return

    # Search interface
    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input(
            "Search for medical devices:",
            placeholder="e.g., 'heart monitor', 'oxygen measurement', 'surgical knife'",
            help="Enter device names, descriptions, or procedures"
        )

    with col2:
        search_method = st.selectbox(
            "Search method:",
            ["combined", "semantic", "keyword"],
            help="Combined: best of both approaches, Semantic: AI-powered similarity, Keyword: exact/fuzzy name matching"
        )
        top_k = st.number_input(
            "Number of results:",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Select how many search results to display"
        )

    # Search button
    if st.button("🔍 Search", type="primary") and query.strip():
        with st.spinner("Searching..."):
            try:
                results = search_engine.search(
                    query.strip(),
                    method=search_method,
                    top_k=top_k
                )

                if results:
                    st.success(f"Found {len(results)} matching device(s)")

                    # Display results
                    for i, result in enumerate(results, 1):
                        device = result["device"]
                        score = result["similarity_score"]
                        search_type = result["search_type"]

                        with st.container():
                            # Header with score and match type
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            with col_a:
                                st.subheader(f"{i}. {device.name}")
                            with col_b:
                                st.metric("Confidence", f"{score:.3f}")
                            with col_c:
                                match_type_display = {
                                    "exact_name": "Exact Name",
                                    "partial_name": "Partial Name",
                                    "fuzzy_name": "Fuzzy Name",
                                    "keyword_match": "Keyword Match",
                                    "semantic": "Semantic Match"
                                }.get(search_type, search_type.title())
                                st.caption(f"📍 {match_type_display}")

                            # Device details
                            st.write(f"**ID:** {device.id}")
                            st.write(f"**Description:** {device.description}")
                            st.write(f"**Typical Uses:** {device.use}")

                            st.divider()

                else:
                    st.info("No matching devices found. Try different keywords or check spelling.")

            except Exception as e:
                st.error(f"Search failed: {e}")

    # Sample queries
    with st.expander("💡 Sample Search Queries"):
        st.markdown("""
        Try these example searches:
        - **"heart monitor"** - Direct name search
        - **"oxygen levels"** - Semantic description search
        - **"surgery knife"** - Fuzzy name matching (typo)
        - **"breathing support"** - Semantic use case search
        - **"cardiac rhythm"** - Technical description search
        - **"adson forceps"** - Specific surgical instrument
        - **"tissue grasping"** - Semantic search for forceps
        - **"vessel clamping"** - Use case search for hemostats
        - **"4.5\" scissors"** - Size-specific instrument search
        - **"large retractor"** - Size and type combination
        - **"curved clamp"** - Shape and function search
        """)

    # Device count
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.write(f"Total devices: {len(search_engine.devices)}")

    # API Key status
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    status = "✅ Available" if api_key_set else "⚠️ Not set (keyword search only)"
    st.sidebar.markdown("### 🔑 OpenAI API Key")
    st.sidebar.write(f"Status: {status}")

    if not api_key_set:
        st.sidebar.markdown("""
        **To enable semantic search:**
        1. Get API key from [OpenAI](https://platform.openai.com/api-keys)
        2. Set environment variable: `export OPENAI_API_KEY=your_key`
        3. Restart the app
        """)

if __name__ == "__main__":
    main()