"""
Diabetic Retinopathy Stage Detection - Streamlit app (Streamlit Community Cloud version).

Uses exactly the same preprocessing as the training notebook (v2: retina crop, square padding,
common field-of-view mask, bilateral denoising, Ben Graham normalisation with normalised
convolution), the fine-tuned EfficientNetB0 model, and Grad-CAM.
Computer Vision coursework prototype (NIBM) - not a medical device, not for diagnosis.
"""
import time

import cv2
import keras
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title='DR Stage Detection', page_icon='\U0001F441\ufe0f', layout='wide')

WORK_SIZE, IMG_SIZE, BAND = 512, 224, 0.76
STAGES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
COLORS = ['#2e7d32', '#9e9d24', '#ef8f00', '#e65100', '#c62828']   # No DR -> Proliferative


# ---------------- preprocessing (identical to the training notebook) ----------------
def crop_to_retina(img, tol=10):
    """Remove the black border: keep rows/columns that are at least 1% retina."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows = np.where(mask.mean(axis=1) > 0.01)[0]
    cols = np.where(mask.mean(axis=0) > 0.01)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img):
    """Pad with black so width == height (keeps the retina round)."""
    h, w = img.shape[:2]
    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    return cv2.copyMakeBorder(img, top, size - h - top, left, size - w - left,
                              cv2.BORDER_CONSTANT, value=0)


def fov_mask(size, band=BAND, radius=0.95):
    """Common field of view: the same circle-with-flat-edges shape for every image."""
    yy, xx = np.mgrid[:size, :size]
    c, r = (size - 1) / 2, size / 2
    circle = (xx - c) ** 2 + (yy - c) ** 2 <= (radius * r) ** 2
    flat = np.abs(yy - c) <= band * r
    return (circle & flat).astype(np.uint8)


FOV = fov_mask(WORK_SIZE)


def geometry(img):
    """Crop, pad to square, resize to the working size; also return the retina mask inside the FOV."""
    img = pad_to_square(crop_to_retina(img))
    interp = cv2.INTER_AREA if img.shape[0] > WORK_SIZE else cv2.INTER_CUBIC
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE), interpolation=interp)
    eye = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10).astype(np.uint8)
    return img, eye & FOV


def finish(img):
    """Apply the common FOV mask and resize to the CNN input size."""
    return cv2.resize(img * FOV[:, :, None], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def denoise(img):
    """Edge-preserving bilateral filter."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=5)


def ben_graham(img, mask):
    """Ben Graham normalisation; local average computed over retina pixels only."""
    sigma = img.shape[0] / 60
    m = mask.astype(np.float32)
    f = img.astype(np.float32)
    local_sum = cv2.GaussianBlur(f * m[:, :, None], (0, 0), sigma)
    local_count = cv2.GaussianBlur(m, (0, 0), sigma)[:, :, None]
    local_mean = local_sum / np.maximum(local_count, 1e-3)
    return np.clip(4 * f - 4 * local_mean + 128, 0, 255).astype(np.uint8)


def preprocess(img):
    """Return (natural-colour display image, model input) for one RGB photo."""
    g, m = geometry(img)
    return finish(g), finish(ben_graham(denoise(g), m))


# ---------------- model and Grad-CAM (loaded once, then cached) ----------------
@st.cache_resource(show_spinner='Loading the model (first visit only)...')
def load_model():
    model = keras.models.load_model('final_model_fp32.keras')
    base = next(layer for layer in model.layers if isinstance(layer, keras.Model))
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    feat_model = keras.Model(inp, base(inp, training=False))   # last conv feature maps (7 x 7 x 1280)
    W = model.layers[-1].get_weights()[0]                      # Dense weights: 1280 features x 5 stages
    return model, feat_model, W


def cam_overlay(x, display, cls, feat_model, W):
    """Grad-CAM for a pooling -> Dense head: feature maps weighted by the class's Dense weights."""
    A = feat_model(x[None].astype('float32'), training=False).numpy()[0]
    cam = np.maximum(A @ W[:, cls], 0)
    cam = cv2.resize(cam / (cam.max() + 1e-8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)[:, :, ::-1]
    eye = display.max(axis=2, keepdims=True) > 0
    return ((0.6 * display + 0.4 * heat) * eye).astype(np.uint8)


# ---------------- look and feel ----------------
CSS = """
<style>
#header {text-align: center; padding: 6px 10px 14px;}
#header h1 {font-size: 2rem; margin-bottom: 2px;}
#header p {opacity: .8; margin: 0;}
.card {border: 1px solid rgba(127,127,127,.35); border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;}
.card .lbl {font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7;}
.card .stage {font-size: 2rem; font-weight: 700; margin: 2px 0 10px;}
.card .bar {height: 10px; border-radius: 6px; background: rgba(127,127,127,.2); overflow: hidden;}
.card .bar div {height: 100%; border-radius: 6px;}
.card .conf {font-size: .9rem; margin: 6px 0 12px; opacity: .85;}
.card .pill {display: inline-block; padding: 4px 12px; border-radius: 999px; margin: 0 6px 6px 0; font-size: .9rem;}
.card .yes {background: rgba(198,40,40,.15); color: #c62828;}
.card .no {background: rgba(46,125,50,.15); color: #2e7d32;}
.card .warn {margin-top: 10px; padding: 8px 12px; border-radius: 10px; background: rgba(239,143,0,.18); color: #b26a00;}
.card .time {margin-top: 10px; font-size: .8rem; opacity: .6;}
#footer {text-align: center; font-size: .85rem; opacity: .7; padding: 16px 8px 4px;}
</style>
"""

EMPTY_CARD = ("<div class='card'><div class='lbl'>Result</div>"
              "<div style='opacity:.7;margin-top:6px'>Upload a retinal fundus photograph to start.</div></div>")

ABOUT_MD = """
**Model:** EfficientNetB0 pretrained on ImageNet, fine-tuned on APTOS 2019 (2,436 training images, 5 ICDR stages).
**Preprocessing:** retina crop, common field-of-view mask, bilateral denoising and Ben Graham colour normalisation, the same as training.

| Test results (523 unseen images) | |
|---|---|
| Quadratic weighted kappa | 0.846 |
| 5-stage accuracy | 74.8% |
| Healthy vs DR accuracy | 97.3% |
| DR sensitivity / specificity | 98.4% / 96.3% |
| ROC-AUC (healthy vs DR) | 0.995 |
"""


def result_card(probs, cls, secs):
    """Colour-coded HTML summary of the prediction."""
    conf, p_dr, p_ref = probs[cls], 1 - probs[0], probs[2:].sum()
    warn = ("<div class='warn'>\u26a0\ufe0f Low confidence: recommend review by a specialist</div>"
            if conf < 0.6 else "")
    return f"""
<div class='card'>
  <div class='lbl'>Predicted stage</div>
  <div class='stage' style='color:{COLORS[cls]}'>{STAGES[cls]}</div>
  <div class='bar'><div style='width:{conf * 100:.0f}%;background:{COLORS[cls]}'></div></div>
  <div class='conf'>{conf:.0%} confidence</div>
  <span class='pill {"yes" if cls > 0 else "no"}'>DR present: {"Yes" if cls > 0 else "No"} ({p_dr:.0%})</span>
  <span class='pill {"yes" if cls >= 2 else "no"}'>Referable: {"Yes" if cls >= 2 else "No"} ({p_ref:.0%})</span>
  {warn}
  <div class='time'>Processed in {secs:.1f} s</div>
</div>"""


# ---------------- page ----------------
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("<div id='header'><h1>Diabetic Retinopathy Stage Detection</h1>"
            "<p>Upload a retinal fundus photograph to estimate its ICDR stage, with a Grad-CAM explanation.</p></div>",
            unsafe_allow_html=True)

model, feat_model, W = load_model()

left, right = st.columns([5, 7], gap='large')
with left:
    uploaded = st.file_uploader('Fundus photograph', type=['png', 'jpg', 'jpeg'])
    photo = Image.open(uploaded).convert('RGB') if uploaded is not None else None
    if photo is not None:
        st.image(photo, caption='Uploaded photograph', width=380)

with right:
    if photo is None:
        st.markdown(EMPTY_CARD, unsafe_allow_html=True)
    else:
        with st.spinner('Analysing...'):
            t0 = time.time()
            display, x = preprocess(np.asarray(photo))
            probs = model(x[None].astype('float32'), training=False).numpy()[0]
            cls = int(probs.argmax())
            cam = cam_overlay(x, display, cls, feat_model, W)
            secs = time.time() - t0
        st.markdown(result_card(probs, cls, secs), unsafe_allow_html=True)
        st.markdown('**Stage probabilities**')
        for stage, p in zip(STAGES, probs):
            st.progress(float(p), text=f'{stage}: {p:.0%}')
        c1, c2 = st.columns(2)
        c1.image(x, caption='Preprocessed input (what the model sees)', width=280)
        c2.image(cam, caption='Grad-CAM: regions behind the prediction', width=280)

with st.expander('About the model'):
    st.markdown(ABOUT_MD)
st.markdown("<div id='footer'>Computer Vision coursework prototype (NIBM) \u00b7 not a medical device \u00b7 "
            "not for diagnosis</div>", unsafe_allow_html=True)
