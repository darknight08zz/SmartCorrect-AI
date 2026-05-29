import streamlit as st
import pandas as pd
import string
import logging
import warnings

# Suppress standard Python warnings
warnings.filterwarnings("ignore")

# Suppress HuggingFace transformers load reports and path warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from corrector import SpellCorrector
from bert_predictor import BERTContextCorrector
import utils

# Configure log systems
logging.basicConfig(
    filename="smartcorrect.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    filemode="a"
)
logger = logging.getLogger("app")

# Set Page Config
st.set_page_config(
    page_title="SmartCorrect AI ✨",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide deploy button, main menu, header, and footer to streamline UI
st.markdown("""
    <style>
        .stDeployButton { display: none !important; }
        #MainMenu { visibility: hidden !important; }
        header[data-testid="stHeader"] { display: none !important; }
        footer { visibility: hidden !important; }
        .block-container { padding-top: 1.5rem !important; }
    </style>
""", unsafe_allow_html=True)

# Silently bootstrap required NLTK resources on load
try:
    utils.download_nltk_resources()
except Exception as e:
    logger.error(f"Error during NLTK download bootstrap: {e}", exc_info=True)

# Custom CSS Injection for Modern Premium Aesthetics (Glassmorphic Carbon & Neon accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0d0f12 0%, #15181f 50%, #1c202a 100%);
        color: #e2e8f0;
    }
    
    /* Header Gradient Text styling */
    .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        font-size: 2.8rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Glassmorphic Container Cards & Bordered Containers */
    .glass-card, [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding: 24px !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 24px !important;
    }
    
    /* Hide empty element containers from style injects */
    div[data-testid="stElementContainer"]:has(style) {
        display: none !important;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.08);
        border-radius: 12px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 20px;
        color: #34d399;
        font-size: 1.15rem;
        line-height: 1.6;
        margin-bottom: 24px;
    }
    
    /* Before and After Panels */
    .before-panel {
        background: rgba(239, 68, 68, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(239, 68, 68, 0.15);
        padding: 16px;
        color: #fca5a5;
        min-height: 120px;
        font-size: 1.05rem;
    }
    
    .after-panel {
        background: rgba(16, 185, 129, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(16, 185, 129, 0.15);
        padding: 16px;
        color: #6ee7b7;
        min-height: 120px;
        font-size: 1.05rem;
    }
    
    /* Custom buttons styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #0b0c10 !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        font-size: 1rem !important;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.45) !important;
    }
    
    /* Input formatting override */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.03) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 8px rgba(0, 242, 254, 0.2) !important;
    }
    
    /* Sidebar aesthetic adjustments */
    [data-testid="stSidebar"] {
        background-color: #090b0e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Resource Cache Loaders -----------------
@st.cache_resource
def get_spell_corrector() -> SpellCorrector:
    """Instantiates and caches the core lexical spelling engine."""
    try:
        logger.info("Instantiating cached SpellCorrector resource.")
        return SpellCorrector()
    except Exception as e:
        logger.error(f"Failed to load cached SpellCorrector: {e}", exc_info=True)
        return None

@st.cache_resource
def get_bert_corrector() -> BERTContextCorrector:
    """Instantiates and caches the deep context BERT MLM engine."""
    try:
        logger.info("Instantiating cached BERTContextCorrector resource.")
        return BERTContextCorrector()
    except Exception as e:
        logger.error(f"Failed to load cached BERTContextCorrector: {e}", exc_info=True)
        return None

# Initialize models
spell_engine = get_spell_corrector()
bert_engine = get_bert_corrector()

# ----------------- Sidebar Configuration Panel -----------------
with st.sidebar:
    st.markdown('<h2 style="margin-top:-10px; font-family:\'Outfit\', sans-serif;">⚙️ Settings Panel</h2>', unsafe_allow_html=True)
    st.markdown("Customize spellchecking and context predictions:")
    
    # 1. Enable BERT Toggle
    enable_bert = st.toggle(
        "Enable BERT Context Correction",
        value=True,
        help="Checks for homophones and semantic grammatical slips on top of basic spellchecks."
    )
    
    # 2. Skip Proper Nouns Toggle
    skip_proper_nouns = st.toggle(
        "Skip Proper Nouns",
        value=True,
        help="Avoids correcting proper nouns (e.g., names of people, places, or brands like Rahul, Mumbai, Google)."
    )
    
    # 3. Custom Tech Terms Area
    custom_terms_str = st.text_area(
        "Add Custom Dictionary Terms",
        value="",
        placeholder="e.g. pytorch, google, streamlit (comma-separated)",
        help="Enter words that should NEVER be corrected by the spelling/grammar system."
    )
    
    # 4. BERT Confidence Slider
    bert_threshold = st.slider(
        "BERT Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        disabled=not enable_bert,
        help="High values only execute context corrections BERT is extremely confident about."
    )
    
    st.markdown("---")
    # Model Status Section (collapsed by default)
    with st.sidebar.expander("📊 Model Status", expanded=False):
        st.markdown("🟢 **TextBlob Lexical Dictionary**: Active")
        if enable_bert:
            if bert_engine and bert_engine.nlp is not None:
                st.markdown("🟢 **BERT MLM Context pipeline**: Loaded")
            else:
                st.markdown("🟡 **BERT MLM Context pipeline**: Offline (Fallback Mode)")
        else:
            st.markdown("⚪ **BERT MLM Context pipeline**: Disabled")

# Parse custom terms
custom_terms = [t.strip() for t in custom_terms_str.split(",") if t.strip()]

# ----------------- Main Section Header -----------------
st.markdown('<h1 class="title-text">SmartCorrect AI ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Intelligent Autocorrect powered by TextBlob + BERT</p>', unsafe_allow_html=True)

# Session State for retaining typed text
if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""

# ----------------- Main Input Area -----------------
# Main interactive input area styled with glassmorphism container
with st.container(border=True):
    st.markdown("### 📝 Enter Text to Correct")
    
    # Simple Load Examples helper
    col_ex1, col_ex2, col_ex3, _ = st.columns([1, 1.2, 1.2, 3.6])
    if col_ex1.button("💡 Homophone Example"):
        st.session_state["user_text"] = "I went to the beech yesterday."
    elif col_ex2.button("💡 Context Grammar Example"):
        st.session_state["user_text"] = "She is a grate cook."
    elif col_ex3.button("💡 Heavy Typing Typos"):
        st.session_state["user_text"] = "The weather is very beautifull today."
    
    # The main text input area
    user_input = st.text_area(
        "Type or paste your text below:",
        value=st.session_state["user_text"],
        placeholder="Type your text here and let SmartCorrect fix it...",
        height=150
    )
    st.session_state["user_text"] = user_input
    
    # Controls row
    col_btn_correct, col_btn_clear, _ = st.columns([2, 1, 4])
    btn_correct = col_btn_correct.button("Correct My Text ✅")
    btn_clear = col_btn_clear.button("Clear")
    
    if btn_clear:
        st.session_state["user_text"] = ""
        st.rerun()

# ----------------- Execution & Results rendering -----------------
if btn_correct:
    if not user_input.strip():
        st.warning("⚠️ Please type or paste some text first!")
    else:
        try:
            logger.info(f"User requested spelling correction for input text length: {len(user_input)}")
            with st.spinner("Analyzing spelling and context semantics..."):
                if enable_bert and bert_engine and bert_engine.nlp is not None:
                    # Run Full TextBlob + BERT comparative pipeline
                    # Note: We scale the BERT confidence threshold based on UI slider selection.
                    # Since our optimized algorithm uses low probability thresholds (0.005), 
                    # we map the UI slider value proportionally to keep it highly intuitive for the user.
                    scaled_threshold = max(0.001, bert_threshold * 0.01)
                    
                    results = bert_engine.full_context_correct(
                        user_input,
                        skip_proper_nouns=skip_proper_nouns,
                        custom_terms=custom_terms,
                        threshold=scaled_threshold
                    )
                    corrected_text = results["corrected_text"]
                    spelling_changes = results["spelling_changes"]
                    context_changes = results["context_changes"]
                    all_changes = results["all_changes"]
                    mode_used = "TextBlob + BERT Context Engine"
                else:
                    # Run TextBlob Lexical only
                    if spell_engine:
                        corrected_text, spelling_changes = spell_engine.basic_correct(
                            user_input,
                            skip_proper_nouns=skip_proper_nouns,
                            custom_terms=custom_terms
                        )
                    else:
                        corrected_text, spelling_changes = user_input, []
                    context_changes = []
                    all_changes = spelling_changes
                    mode_used = "TextBlob (Lexical Mode Only)"
            
            # --- 1. RESULTS CARD ---
            with st.container(border=True):
                st.markdown("### ✨ Correction Results")
                st.markdown(f'<div class="success-box"><strong>Corrected Output:</strong><br>{corrected_text}</div>', unsafe_allow_html=True)
                
                # Copy to clipboard container
                st.write("#### 📋 Copy Corrected Text:")
                st.code(corrected_text, language="text")
            
            # --- 2. METRICS ROW ---
            st.markdown("### 📊 Diagnostics Dashboard")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            total_words = len(user_input.split())
            corrections_count = len(all_changes)
            
            with col_m1:
                st.metric("Total Words Checked", total_words)
            with col_m2:
                st.metric("Corrections Made", corrections_count)
            with col_m3:
                st.metric("Correction Mode Used", mode_used)
                
            # --- 3. COMPARISON TABLE ---
            with st.container(border=True):
                st.markdown("### 📋 Detailed Modifications Log")
                
                if all_changes:
                    table_data = []
                    for change in all_changes:
                        # Determine correction type
                        is_context = "reason" in change and change["reason"] == "Contextual Error"
                        corr_type = "Contextual (BERT)" if is_context else "Lexical (TextBlob)"
                        
                        table_data.append({
                            "Original Word": change["original"],
                            "Corrected Word": change["corrected"],
                            "Correction Type (Spelling/Context)": corr_type
                        })
                    
                    df_changes = pd.DataFrame(table_data)
                    st.dataframe(df_changes, use_container_width=True)
                else:
                    st.info("🎉 Excellent! No spelling or contextual corrections were needed.")
            
            # --- 4. BEFORE/AFTER COMPARE SECTION ---
            st.markdown("### 👁️ Visual Editing Comparison")
            col_before, col_after = st.columns(2)
            
            with col_before:
                with st.container(border=True):
                    st.markdown("#### 🟥 Before")
                    st.markdown(f'<div class="before-panel">{user_input}</div>', unsafe_allow_html=True)
                
            with col_after:
                with st.container(border=True):
                    st.markdown("#### 🟩 After")
                    st.markdown(f'<div class="after-panel">{corrected_text}</div>', unsafe_allow_html=True)
                
            # --- 5. EDIT HIGHLIGHTS HTML DIFF ---
            with st.container(border=True):
                st.markdown("#### 🔍 Word-Level Edit Highlights")
                diff_html = utils.get_diff_html(user_input, corrected_text)
                st.markdown(diff_html, unsafe_allow_html=True)
                st.caption("Visual Legend: Pink strikethrough for deleted original spelling slips; green underline for clean corrections.")
            
            logger.info("Successfully corrected text. Rendering dashboard metrics.")
            
        except Exception as e:
            logger.error(f"Error during Streamlit frontend correction click: {e}", exc_info=True)
            st.error(f"❌ An error occurred during the spelling correction process: {e}")
