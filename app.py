import os
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agri-Tech Diagnostic",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent

DEF_MODEL_PATH = BASE_DIR / "deficiency_300px_autosave.keras"
GRO_MODEL_PATH = BASE_DIR / "best_growth_fusion_model_v2.keras"  # change if needed

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    :root {
        --bg: #f4f8f2;
        --card: rgba(255, 255, 255, 0.90);
        --text: #1f2d1f;
        --muted: #667085;
        --primary: #2f855a;
        --primary-dark: #256b48;
        --green: #16a34a;
        --red: #dc2626;
        --shadow: 0 10px 30px rgba(31, 45, 31, 0.10);
        --radius-lg: 20px;
        --radius-md: 16px;
    }

    .stApp {
        background: linear-gradient(180deg, #eef7ea 0%, #f7faf7 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 28px 30px;
        border-radius: var(--radius-lg);
        background: linear-gradient(135deg, #1f6f4a 0%, #38a169 100%);
        color: white;
        box-shadow: var(--shadow);
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .hero p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 1rem;
        color: rgba(255,255,255,0.90);
    }

    .pill-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .pill {
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.22);
        font-size: 0.9rem;
        font-weight: 500;
    }

    .card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.60);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        padding: 22px;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        color: var(--text);
    }

    .section-sub {
        font-size: 0.95rem;
        color: var(--muted);
        margin-bottom: 1rem;
    }

    .result-card {
        border-radius: 18px;
        padding: 20px;
        color: white;
        box-shadow: var(--shadow);
        min-height: 140px;
    }

    .result-green {
        background: linear-gradient(135deg, #15803d 0%, #22c55e 100%);
    }

    .result-red {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
    }

    .result-label {
        font-size: 0.95rem;
        opacity: 0.92;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .preview-box {
        border: 2px dashed #b7cbb2;
        border-radius: 18px;
        padding: 14px;
        background: #fbfdfb;
    }

    .info-box {
        background: #f7fbf5;
        border-left: 5px solid #38a169;
        padding: 16px 18px;
        border-radius: 14px;
        color: #2b4733;
        margin-top: 10px;
    }

    .footer {
        text-align: center;
        color: #667085;
        padding-top: 10px;
        font-size: 0.92rem;
    }

    div[data-testid="stFileUploader"] {
        background: #f9fcf8;
        border-radius: 14px;
        padding: 10px;
        border: 1px solid #dbe8d7;
    }

    .stButton button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        padding: 0.85rem 1rem;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(47, 133, 90, 0.28);
    }

    .stButton button:hover {
        filter: brightness(1.03);
        transform: translateY(-1px);
        transition: 0.2s ease;
    }

    @media (max-width: 768px) {
        .hero h1 {
            font-size: 1.7rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LABELS ----------------
CROP_MAP = {"Rice": 0, "Maize": 1, "Coffee": 2}

DEF_LABELS = {
    0: "Nitrogen (N)",
    1: "Phosphorus (P)",
    2: "Potassium (K)",
    3: "Healthy"
}

GROWTH_LABELS = {
    0: "Stage 1 (Seedling)",
    1: "Stage 2 (Vegetative)",
    2: "Stage 3 (Flowering)"
}

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_models():
    if not DEF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Deficiency model not found: {DEF_MODEL_PATH}")
    if not GRO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Growth model not found: {GRO_MODEL_PATH}")

    def_model = tf.keras.models.load_model(str(DEF_MODEL_PATH))
    gro_model = tf.keras.models.load_model(str(GRO_MODEL_PATH))
    return def_model, gro_model

try:
    def_model, gro_model = load_models()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ---------------- SHARED FUNCTIONS ----------------
def render_header():
    st.markdown("""
    <div class="hero">
        <h1>🌱 Smart Crop Health & Growth Analyzer</h1>
        <p>AI-powered crop diagnostics with separate workflows for growth stage and nutrient deficiency detection.</p>
        <div class="pill-row">
            <div class="pill">Simpler to use</div>
            <div class="pill">Happy Crop Life</div>
            <div class="pill">Easy to Understand</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def prepare_meta_input(selected_crop):
    crop_idx = CROP_MAP[selected_crop]
    meta_input = np.zeros((1, 3), dtype=np.float32)
    meta_input[0, crop_idx] = 1.0
    return meta_input

def preprocess_image(img, size):
    arr = np.array(img.resize(size), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- PAGE: HOME ----------------
def home_page():
    render_header()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Welcome</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Choose a module from the sidebar and run only the prediction you need.</div>', unsafe_allow_html=True)

    st.write("### Available Modules")
    st.write("- **Growth Prediction**: Predicts the crop growth stage from the uploaded leaf image.")
    st.write("- **Deficiency Prediction**: Detects whether the crop is healthy or affected by nutrient deficiency.")
    st.write("- Each workflow is shown separately, so the interface stays simple and focused.")

    st.markdown("""
    <div class="info-box">
        Upload a clear leaf image, select the crop, and use the sidebar to switch between prediction modules.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PAGE: GROWTH ----------------
def growth_page():
    render_header()
    st.markdown("## 🌿 Growth Prediction")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload Leaf Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Choose crop type and upload a leaf image for growth-stage analysis.</div>', unsafe_allow_html=True)

        selected_crop = st.selectbox("Select Crop Type", list(CROP_MAP.keys()), key="growth_crop")
        uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], key="growth_upload")
        analyze = st.button("Analyze Growth Stage", key="growth_btn")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Preview your uploaded image before running the model.</div>', unsafe_allow_html=True)

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Leaf Sample", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Upload an image to preview it here.")

        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and analyze:
        with st.spinner("Analyzing growth stage..."):
            img = Image.open(uploaded_file).convert("RGB")
            meta_input = prepare_meta_input(selected_crop)
            img_224 = preprocess_image(img, (224, 224))

            gro_pred = gro_model.predict(
                {"image_input": img_224, "crop_input": meta_input},
                verbose=0
            )
            gro_idx = int(np.argmax(gro_pred))
            gro_res = GROWTH_LABELS[gro_idx]
            gro_conf = float(np.max(gro_pred)) * 100

        st.markdown("### Growth Result")
        st.markdown(f"""
        <div class="result-card result-green">
            <div class="result-label">Predicted Growth Stage</div>
            <div class="result-value">{gro_res}</div>
            <div style="margin-top:10px; font-size:0.95rem;">Confidence: {gro_conf:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    elif analyze and not uploaded_file:
        st.warning("Please upload a leaf image before clicking Analyze Growth Stage.")

# ---------------- PAGE: DEFICIENCY ----------------
def deficiency_page():
    render_header()
    st.markdown("## 🧪 Deficiency Prediction")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload Leaf Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Choose crop type and upload a leaf image for deficiency analysis.</div>', unsafe_allow_html=True)

        selected_crop = st.selectbox("Select Crop Type", list(CROP_MAP.keys()), key="def_crop")
        uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], key="def_upload")
        analyze = st.button("Analyze Deficiency", key="def_btn")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Preview your uploaded image before running the model.</div>', unsafe_allow_html=True)

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Leaf Sample", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Upload an image to preview it here.")

        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and analyze:
        with st.spinner("Analyzing deficiency status..."):
            img = Image.open(uploaded_file).convert("RGB")
            meta_input = prepare_meta_input(selected_crop)
            img_300 = preprocess_image(img, (300, 300))

            def_pred = def_model.predict(
                {"image_input": img_300, "crop_input": meta_input},
                verbose=0
            )
            def_idx = int(np.argmax(def_pred))
            def_res = DEF_LABELS[def_idx]
            def_conf = float(np.max(def_pred)) * 100

        st.markdown("### Deficiency Result")

        if def_res == "Healthy":
            st.markdown(f"""
            <div class="result-card result-green">
                <div class="result-label">Health Status</div>
                <div class="result-value">{def_res}</div>
                <div style="margin-top:10px; font-size:0.95rem;">Confidence: {def_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-red">
                <div class="result-label">Detected Deficiency</div>
                <div class="result-value">{def_res}</div>
                <div style="margin-top:10px; font-size:0.95rem;">Confidence: {def_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    elif analyze and not uploaded_file:
        st.warning("Please upload a leaf image before clicking Analyze Deficiency.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Growth Prediction", "Deficiency Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("Use the sidebar to switch between the two prediction pages.")

# ---------------- ROUTING ----------------
if page == "Home":
    home_page()
elif page == "Growth Prediction":
    growth_page()
elif page == "Deficiency Prediction":
    deficiency_page()

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Developed with Streamlit, TensorFlow, and AI-based crop diagnostics.
</div>
""", unsafe_allow_html=True)
