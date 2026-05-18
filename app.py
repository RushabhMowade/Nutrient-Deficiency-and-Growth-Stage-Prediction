import os
from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Crop Health Analyzer", layout="wide")


# ---------------- DATA & MODELS ----------------
CROP_MAP = {"Rice": 0, "Maize": 1, "Coffee": 2}

# Updated Growth Stage Labels to 1-based indexing
GROWTH_LABELS = {
    1: "Seedling Stage",
    2: "Vegetative Stage",
    3: "Flowering Stage"
}

DEF_LABELS = {
    0: "Nitrogen (N)",
    1: "Phosphorus (P)",
    2: "Potassium (K)",
    3: "Healthy"
}

AGRI_ADVICE = {
    "Nitrogen (N)": {
        "Reasoning": "Yellowing starts at leaf tips and moves along the midrib in a V-shape.",
        "Action": "Apply 50-100kg/ha of Urea. Maintain consistent irrigation.",
        "Fertilizer": "Urea (46-0-0) or Ammonium Nitrate"
    },
    "Phosphorus (P)": {
        "Reasoning": "Stunted growth; leaves dark green with purplish tints on edges.",
        "Action": "Apply phosphate fertilizers near the root zone.",
        "Fertilizer": "DAP or Single Super Phosphate (SSP)"
    },
    "Potassium (K)": {
        "Reasoning": "Yellowing and browning (scorching) along the leaf margins.",
        "Action": "Apply Muriate of Potash. Check soil drainage.",
        "Fertilizer": "Muriate of Potash (MOP)"
    },
    "Healthy": {
        "Reasoning": "No visual deficiencies detected. Optimal chlorophyll levels.",
        "Action": "Continue standard crop management.",
        "Fertilizer": "No corrective fertilizer needed."
    }
}


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
    color: #1b4332;
    margin-bottom: 0.5rem;
}

.sub-text {
    text-align: center;
    color: #5c6b73;
    margin-bottom: 1.5rem;
}

.preview-box {
    background: #f8fafc;
    border: 1px solid #d9e2ec;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}

.result-card {
    background: linear-gradient(135deg, #f4fff6, #e9f9ee);
    border: 1px solid #cce7d3;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}

.reason-box {
    background: #eef6ff;
    border-left: 6px solid #2f80ed;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.action-box {
    background: #fff8e8;
    border-left: 6px solid #f4b400;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.fert-box {
    background: #edfdf3;
    border-left: 6px solid #16a34a;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.info-box {
    background: #f6f9fc;
    border: 1px dashed #b8c4d6;
    border-radius: 14px;
    padding: 16px;
    margin-top: 10px;
}

.box-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 6px;
}

.box-text {
    font-size: 0.97rem;
    color: #374151;
    line-height: 1.6;
}

.diagnosis-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #166534;
    margin-bottom: 8px;
}

.conf-text {
    font-size: 1rem;
    color: #374151;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    d_model = tf.keras.models.load_model("deficiency_300px_autosave.keras")
    g_model = tf.keras.models.load_model("best_growth_fusion_model_v2.keras")
    return d_model, g_model


def_model, gro_model = load_models()


# ---------------- XAI LOGIC (Grad-CAM) ----------------
def get_gradcam(model, img_array, meta_input):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("top_activation").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, meta_input])
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = np.maximum(tf.squeeze(heatmap).numpy(), 0) / (np.max(heatmap) + 1e-10)
    return heatmap


# ---------------- APP HEADER ----------------
st.markdown('<div class="main-title">Smart Crop Health Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Predict crop growth stage and diagnose nutrient deficiencies with visual reasoning.</div>',
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Growth Prediction", "Deficiency Prediction"])


# ---------------- GROWTH PAGE ----------------
if page == "Growth Prediction":
    st.header("Growth Stage Analysis")

    crop = st.selectbox("Select Crop", list(CROP_MAP.keys()))
    file = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"], key="gro")

    if file is not None:
        img = Image.open(file).convert("RGB")
        st.markdown('<div class="preview-box">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image Preview", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if file is not None and st.button("Predict Growth", use_container_width=True):
        file.seek(0)
        img = Image.open(file).convert("RGB")
        img_arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        meta = np.zeros((1, 3))
        meta[0, CROP_MAP[crop]] = 1.0

        preds = gro_model.predict([img_arr, meta], verbose=0)
        
        # Shift index by +1 to transition from 0-indexed outputs to your 1-indexed labels
        pred_idx = int(np.argmax(preds)) + 1
        
        # If your model natively outputs 4 classes (where index 0 is placeholder/unused), 
        # remove the "+ 1" line above and use: pred_idx = int(np.argmax(preds))
        
        growth_stage_label = GROWTH_LABELS.get(pred_idx, "Unknown Stage")
        confidence = float(np.max(preds) * 100)

        st.markdown(f"""
        <div class="result-card">
            <div class="diagnosis-title">Growth Stage: {growth_stage_label}</div>
            <div class="conf-text">Confidence: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- DEFICIENCY PAGE ----------------
elif page == "Deficiency Prediction":
    st.header("Deficiency & Visual Reasoning")

    col1, col2 = st.columns([1, 1.25])

    with col1:
        crop = st.selectbox("Select Crop", list(CROP_MAP.keys()))
        file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"], key="def")
        run_btn = st.button("Run Diagnosis", use_container_width=True)

        if file is not None:
            preview_img = Image.open(file).convert("RGB")
            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
            st.image(preview_img, caption="Uploaded Leaf Preview", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <div class="box-title">How this works</div>
            <div class="box-text">
                Upload a clear leaf image, select the crop, and click <b>Run Diagnosis</b>.
                The model predicts the deficiency class and the heatmap highlights the visual regions used during prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if file is not None and run_btn:
        file.seek(0)
        img = Image.open(file).convert("RGB")

        img_arr = np.array(img.resize((300, 300)), dtype=np.float32) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        meta = np.zeros((1, 3))
        meta[0, CROP_MAP[crop]] = 1.0

        preds = def_model.predict([img_arr, meta], verbose=0)
        pred_idx = int(np.argmax(preds))
        label = DEF_LABELS[pred_idx]
        confidence = float(np.max(preds) * 100)

        heatmap = get_gradcam(def_model, img_arr, meta)
        heatmap_res = cv2.resize(heatmap, (300, 300))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.uint8(img_arr[0] * 255), 0.6, heatmap_color, 0.4, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        adv = AGRI_ADVICE[label]

        res1, res2 = st.columns([1, 1.2])

        with res1:
            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
            st.image(overlay, caption="Visual Reasoning Heatmap", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with res2:
            st.markdown(f"""
            <div class="result-card">
                <div class="diagnosis-title">Diagnosis: {label}</div>
                <div class="conf-text">Confidence: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="reason-box">
                <div class="box-title">Visual Reasoning</div>
                <div class="box-text">{adv['Reasoning']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="action-box">
                <div class="box-title">Recommended Action</div>
                <div class="box-text">{adv['Action']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="fert-box">
                <div class="box-title">Suggested Fertilizer</div>
                <div class="box-text">{adv['Fertilizer']}</div>
            </div>
            """, unsafe_allow_html=True)
