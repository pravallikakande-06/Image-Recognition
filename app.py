"""
Image Recognition App
Classifies an image into 1,000 ImageNet categories using a pretrained
Vision Transformer / ResNet model from Hugging Face. Runs fully offline
after the first model download. No API key needed.
"""
import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Image Recognition", page_icon="🖼️", layout="centered")

# ---------------- Custom CSS ----------------
st.markdown(
    """
<style>
/* ========== Design tokens ========== */
:root {
    --ink: #1b1740;
    --ink-soft: #565d86;
    --indigo: #4f46e5;
    --violet: #7c3aed;
    --sky: #2f80ed;
    --mint: #10b981;
    --glass: rgba(255, 255, 255, 0.60);
    --glass-strong: rgba(255, 255, 255, 0.86);
    --glass-border: rgba(255, 255, 255, 0.95);
    --shadow: 0 12px 34px rgba(79, 70, 229, 0.13), 0 2px 6px rgba(79, 70, 229, 0.06);
    --shadow-hover: 0 18px 42px rgba(79, 70, 229, 0.24), 0 3px 8px rgba(79, 70, 229, 0.10);
    --radius: 20px;
    --grad: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #2f80ed 100%);
    --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            "Helvetica Neue", Arial, sans-serif;
}

/* ========== Base / animated background ========== */
.stApp {
    color-scheme: light;
    background:
        linear-gradient(135deg, #eef2ff 0%, #e4e8ff 40%, #f1e9ff 75%, #e8f1ff 100%);
    background-attachment: fixed;
    font-family: var(--font);
    color: var(--ink);
}
/* soft floating color orbs behind everything */
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    z-index: 0;
    width: 46vmax;
    height: 46vmax;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.45;
    pointer-events: none;
}
.stApp::before {
    top: -18vmax; left: -14vmax;
    background: radial-gradient(circle, #a78bfa, transparent 65%);
    animation: floatA 18s ease-in-out infinite alternate;
}
.stApp::after {
    bottom: -20vmax; right: -14vmax;
    background: radial-gradient(circle, #60a5fa, transparent 65%);
    animation: floatB 22s ease-in-out infinite alternate;
}
@keyframes floatA { to { transform: translate(8vmax, 6vmax) scale(1.1); } }
@keyframes floatB { to { transform: translate(-7vmax, -5vmax) scale(1.12); } }

html, body, [class*="st-"], button, input, textarea { font-family: var(--font); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"], [data-testid="stSidebar"] { position: relative; z-index: 1; }

.stApp p, .stApp label, .stApp li, .stApp span,
[data-testid="stMarkdownContainer"] { color: var(--ink); }

/* Custom scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #a5b4fc, #c4b5fd);
    border-radius: 10px;
}
::-webkit-scrollbar-track { background: transparent; }

/* Centered content */
.block-container {
    max-width: 820px;
    padding-top: 2.6rem;
    padding-bottom: 3rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

/* ========== Title, subtitle, chips ========== */
h1 {
    text-align: center;
    font-weight: 800 !important;
    font-size: clamp(2.2rem, 7vw, 3.6rem) !important;
    letter-spacing: -0.04em;
    line-height: 1.12 !important;
    padding: 0.2rem 0 0.4rem !important;
    background: linear-gradient(100deg, #4338ca 0%, #7c3aed 45%, #2f80ed 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 6px 14px rgba(79, 70, 229, 0.18));
    animation: titleIn 0.8s cubic-bezier(.2, .8, .2, 1) both;
}
h1 a, [data-testid="stHeaderActionElements"] { display: none !important; }

.app-subtitle {
    text-align: center;
    max-width: 540px;
    margin: 0 auto 1rem auto;
    font-size: 1.1rem;
    line-height: 1.65;
    color: var(--ink-soft) !important;
}
.chip-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin: 0 auto 2rem auto;
}
.chip {
    padding: 0.38rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--indigo) !important;
    background: var(--glass-strong);
    border: 1px solid rgba(79, 70, 229, 0.2);
    border-radius: 999px;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.08);
}
@keyframes titleIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ========== Sidebar ========== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(238,242,255,0.88));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid var(--glass-border);
    box-shadow: 10px 0 40px rgba(79, 70, 229, 0.10);
}
[data-testid="stSidebar"] h2 {
    font-weight: 800;
    font-size: 1.4rem;
    letter-spacing: -0.02em;
    padding-bottom: 0.6rem;
    margin-bottom: 0.8rem;
    background: var(--grad);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
}
[data-testid="stSidebar"] h2::after {
    content: "";
    position: absolute;
    left: 0; bottom: 0;
    width: 56px; height: 4px;
    border-radius: 4px;
    background: var(--grad);
}
[data-testid="stSidebar"] label p { font-weight: 600; color: var(--ink); }
[data-testid="stSidebar"] hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79, 70, 229, 0.35), transparent);
    margin: 1.3rem 0;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background: #ffffff;
    border: 1.5px solid rgba(79, 70, 229, 0.22);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.06);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--indigo);
    box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.13);
}
div[data-baseweb="select"] * { color: var(--ink); }

/* Slider */
[data-testid="stSlider"] [role="slider"] {
    background: var(--grad);
    border: 3px solid #ffffff;
    box-shadow: 0 3px 10px rgba(79, 70, 229, 0.5);
    transition: transform 0.15s ease;
}
[data-testid="stSlider"] [role="slider"]:hover { transform: scale(1.15); }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--indigo); font-weight: 800; }

/* ImageNet info card (sidebar caption) */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    position: relative;
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.10), rgba(124, 58, 237, 0.10));
    border: 1px solid rgba(79, 70, 229, 0.20);
    border-radius: 16px;
    padding: 1rem 1rem 1rem 3rem;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.08);
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]::before {
    content: "ℹ️";
    position: absolute;
    left: 0.95rem; top: 0.95rem;
    font-size: 1.2rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--ink-soft);
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0;
}

/* ========== Tabs ========== */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.5rem;
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 0.4rem;
    box-shadow: var(--shadow);
    border-bottom: none;
}
[data-testid="stTabs"] button[role="tab"] {
    flex: 1;
    justify-content: center;
    height: auto;
    padding: 0.75rem 1rem;
    border-radius: 13px;
    background: transparent;
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.25s ease;
}
[data-testid="stTabs"] button[role="tab"] p {
    font-weight: 600;
    font-size: 1.02rem;
    color: var(--ink-soft);
    transition: color 0.2s ease;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    background: rgba(79, 70, 229, 0.10);
    transform: translateY(-1px);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: var(--grad);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.38);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #ffffff; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: 1.3rem; }

/* ========== File uploader ========== */
[data-testid="stFileUploader"] label p { font-weight: 700; font-size: 1.02rem; }
[data-testid="stFileUploaderDropzone"] {
    background:
        linear-gradient(135deg, rgba(255,255,255,0.92), rgba(238,242,255,0.85));
    border: 2px dashed rgba(79, 70, 229, 0.45);
    border-radius: var(--radius);
    padding: 2.2rem 1.2rem;
    box-shadow: var(--shadow);
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--violet);
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}
[data-testid="stFileUploaderDropzone"] svg { color: var(--indigo); }
[data-testid="stFileUploaderDropzone"] button {
    border-radius: 11px;
    border: 1.5px solid var(--indigo);
    background: #ffffff;
    transition: background 0.2s ease, transform 0.2s ease;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * { color: var(--indigo); font-weight: 700; }
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(79, 70, 229, 0.08);
    transform: translateY(-1px);
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: var(--ink-soft); }
[data-testid="stFileUploaderFile"] {
    background: var(--glass-strong);
    border-radius: 12px;
    padding: 0.4rem 0.7rem;
    margin-top: 0.6rem;
}

/* ========== Camera ========== */
[data-testid="stCameraInput"] {
    background: var(--glass-strong);
    border: 1px solid var(--glass-border);
    border-top: 5px solid transparent;
    border-image: var(--grad) 1;
    border-image-slice: 1;
    border-radius: var(--radius);
    padding: 1.1rem;
    box-shadow: var(--shadow);
}
[data-testid="stCameraInput"] label p { font-weight: 700; font-size: 1.02rem; }
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img { border-radius: 14px; max-width: 100%; }
[data-testid="stCameraInput"] button { border-radius: 11px; }

/* ========== Image preview ========== */
[data-testid="stImage"] {
    background: var(--glass-strong);
    border: 1px solid var(--glass-border);
    border-radius: 22px;
    padding: 1rem 1rem 0.8rem;
    box-shadow: var(--shadow);
    text-align: center;
    margin-top: 0.5rem;
}
[data-testid="stImage"] img {
    max-width: 100%;
    height: auto;
    border-radius: 14px;
    box-shadow: 0 10px 28px rgba(27, 23, 64, 0.22);
    margin: 0 auto;
}
[data-testid="stImage"] [data-testid="stImageCaption"],
[data-testid="stImage"] figcaption {
    color: var(--ink-soft);
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.6rem;
}

/* ========== Recognize button (primary) ========== */
.stButton { display: flex; justify-content: center; margin-top: 0.6rem; }
.stButton > button,
button[data-testid="stBaseButton-primary"] {
    position: relative;
    overflow: hidden;
    width: 100%;
    padding: 1rem 1.5rem;
    min-height: 3.6rem;
    border: none;
    border-radius: 18px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2f80ed 100%);
    background-size: 170% 170%;
    background-position: 0% 50%;
    box-shadow: 0 12px 28px rgba(79, 70, 229, 0.42), inset 0 1px 0 rgba(255,255,255,0.35);
    transition: transform 0.2s ease, box-shadow 0.25s ease, background-position 0.5s ease;
}
/* light sweep on hover */
.stButton > button::after,
button[data-testid="stBaseButton-primary"]::after {
    content: "";
    position: absolute;
    top: 0; left: -80%;
    width: 50%; height: 100%;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,0.35), transparent);
    transform: skewX(-20deg);
    transition: left 0.6s ease;
}
.stButton > button p,
button[data-testid="stBaseButton-primary"] p {
    color: #ffffff;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: 0.01em;
}
.stButton > button:hover,
button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-3px);
    background-position: 100% 50%;
    box-shadow: 0 18px 38px rgba(124, 58, 237, 0.52), inset 0 1px 0 rgba(255,255,255,0.4);
    border: none;
}
.stButton > button:hover::after,
button[data-testid="stBaseButton-primary"]:hover::after { left: 130%; }
.stButton > button:active,
button[data-testid="stBaseButton-primary"]:active {
    transform: translateY(0);
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.45);
}
.stButton > button:focus-visible {
    outline: 3px solid rgba(124, 58, 237, 0.5);
    outline-offset: 3px;
}

/* ========== Alerts: result card + info card ========== */
[data-testid="stAlert"] {
    border-radius: var(--radius);
    padding: 0.4rem 0.6rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--glass-border);
}
[data-testid="stAlert"] p { color: var(--ink); }

/* Top prediction (success) */
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
    background:
        radial-gradient(circle at 100% 0%, rgba(16, 185, 129, 0.22), transparent 55%),
        linear-gradient(135deg, rgba(255,255,255,0.95), rgba(236, 253, 245, 0.92));
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-left: 8px solid var(--mint);
    padding: 1.2rem 1.3rem;
    margin-top: 1rem;
    box-shadow: 0 16px 40px rgba(16, 185, 129, 0.20), 0 2px 8px rgba(16, 185, 129, 0.10);
}
[data-testid="stAlertContentSuccess"] p {
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.3;
    color: #064e3b;
}
[data-testid="stAlertContentSuccess"] strong {
    display: block;
    margin-bottom: 0.2rem;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #059669;
}
[data-testid="stAlertContentSuccess"] strong::before { content: "🏆 "; }

/* Info message */
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
    background:
        radial-gradient(circle at 0% 0%, rgba(124, 58, 237, 0.14), transparent 55%),
        linear-gradient(135deg, rgba(255,255,255,0.9), rgba(224, 231, 255, 0.85));
    border: 1px solid rgba(79, 70, 229, 0.22);
    padding: 1.3rem 1.2rem;
    text-align: center;
}
[data-testid="stAlertContentInfo"] p {
    font-size: 1.08rem;
    font-weight: 600;
    color: var(--ink);
}

/* ========== All predictions panel ========== */
h3 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin-top: 1.6rem;
    padding: 0.8rem 1.1rem !important;
    background: var(--glass);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-left: 6px solid var(--violet);
    border-radius: 16px;
    box-shadow: var(--shadow);
}

/* Each prediction row (containers created with key="pred_N") */
[class*="st-key-pred_"] {
    position: relative;
    background: var(--glass-strong);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 0.9rem 1.1rem 1rem 3.6rem;
    margin-bottom: 0.6rem;
    gap: 0.55rem;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.09);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[class*="st-key-pred_"]:hover {
    transform: translateY(-2px) translateX(2px);
    box-shadow: var(--shadow-hover);
}
[class*="st-key-pred_"] p { font-size: 1rem; font-weight: 600; margin: 0; }

/* rank badge */
[class*="st-key-pred_"]::before {
    position: absolute;
    left: 0.9rem; top: 0.85rem;
    width: 1.9rem; height: 1.9rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%;
    font-size: 0.85rem;
    font-weight: 800;
    color: var(--indigo);
    background: rgba(79, 70, 229, 0.12);
}
.st-key-pred_0::before { content: "1"; }
.st-key-pred_1::before { content: "2"; }
.st-key-pred_2::before { content: "3"; }
.st-key-pred_3::before { content: "4"; }
.st-key-pred_4::before { content: "5"; }
.st-key-pred_5::before { content: "6"; }
.st-key-pred_6::before { content: "7"; }
.st-key-pred_7::before { content: "8"; }
.st-key-pred_8::before { content: "9"; }
.st-key-pred_9::before { content: "10"; }

/* highlight the best match */
.st-key-pred_0 {
    border: 1.5px solid rgba(124, 58, 237, 0.45);
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(237, 233, 254, 0.9));
}
.st-key-pred_0::before { color: #ffffff; background: var(--grad); }
.st-key-pred_0 p { font-weight: 800; }

/* Progress bars */
[data-testid="stProgress"] div[role="progressbar"] {
    background: rgba(79, 70, 229, 0.13);
    border-radius: 999px;
    height: 0.7rem;
    overflow: hidden;
}
[data-testid="stProgress"] div[role="progressbar"] > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed, #2f80ed);
    border-radius: 999px;
    box-shadow: 0 0 12px rgba(124, 58, 237, 0.5);
}

/* Spinner */
[data-testid="stSpinner"] p { color: var(--indigo); font-weight: 700; }

/* Footer */
.app-footer {
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    font-size: 0.85rem;
    color: var(--ink-soft) !important;
    border-top: 1px solid rgba(79, 70, 229, 0.15);
}

/* ========== Responsive ========== */
@media (max-width: 640px) {
    .block-container { padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1.6rem; }
    .app-subtitle { font-size: 0.98rem; }
    [data-testid="stTabs"] button[role="tab"] { padding: 0.65rem 0.5rem; }
    [data-testid="stAlertContentSuccess"] p { font-size: 1.3rem; }
    [data-testid="stFileUploaderDropzone"] { padding: 1.3rem 0.8rem; }
    [class*="st-key-pred_"] { padding-left: 3.2rem; }
}

/* Respect reduced-motion preferences */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

MODELS = {
    "ViT Base (Google)": "google/vit-base-patch16-224",
    "ResNet-50 (Microsoft)": "microsoft/resnet-50",
}


@st.cache_resource(show_spinner="Loading model (first run downloads it, please wait)...")
def load_classifier(model_id: str):
    return pipeline("image-classification", model=model_id)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
top_k = st.sidebar.slider("Number of predictions", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Models are trained on ImageNet (1,000 everyday object classes).")

# ---------------- Main ----------------
st.title("🖼️ Image Recognition")
st.markdown(
    '<p class="app-subtitle">Upload an image or take a photo, and the model will tell you what it sees.</p>'
    '<div class="chip-row">'
    '<span class="chip">ViT &amp; ResNet-50</span>'
    '<span class="chip">1,000 ImageNet classes</span>'
    '<span class="chip">Runs offline</span>'
    '<span class="chip">No API key</span>'
    "</div>",
    unsafe_allow_html=True,
)

tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])
image = None

with tab_upload:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if file:
        image = Image.open(file).convert("RGB")

with tab_camera:
    photo = st.camera_input("Take a picture")
    if photo:
        image = Image.open(photo).convert("RGB")

if image is not None:
    st.image(image, caption="Input image")

    if st.button("🔍 Recognize", type="primary"):
        classifier = load_classifier(MODELS[model_name])
        with st.spinner("Analyzing image..."):
            results = classifier(image, top_k=top_k)

        best = results[0]
        st.success(f"**Top prediction:** {best['label']}  ({best['score'] * 100:.2f}%)")

        st.subheader("All predictions")
        for i, r in enumerate(results):
            # Container only groups each row so CSS can style it as a card
            with st.container(key=f"pred_{i}"):
                st.write(f"{r['label']} — {r['score'] * 100:.2f}%")
                st.progress(float(r["score"]))
else:
    st.info("👆 Upload an image or use the camera to get started.")

st.markdown(
    '<p class="app-footer">Built with Streamlit and Hugging Face Transformers</p>',
    unsafe_allow_html=True,
)