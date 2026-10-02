"""
Diabetic Retinopathy Stage Detection - Gradio app (Hugging Face Spaces version).

Uses exactly the same preprocessing as the training notebook (v2: retina crop, square padding,
common field-of-view mask, bilateral denoising, Ben Graham normalisation with normalised
convolution), the fine-tuned EfficientNetB0 model, and Grad-CAM.
Computer Vision coursework prototype (NIBM) - not a medical device, not for diagnosis.
"""
import glob
import inspect
import os
import time

import cv2
import gradio as gr
import keras
import numpy as np

WORK_SIZE, IMG_SIZE, BAND = 512, 224, 0.76
STAGES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']


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


# ---------------- model and Grad-CAM ----------------
model = keras.models.load_model('final_model_fp32.keras')
base = next(layer for layer in model.layers if isinstance(layer, keras.Model))
_inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
feat_model = keras.Model(_inp, base(_inp, training=False))   # last conv feature maps (7 x 7 x 1280)
W = model.layers[-1].get_weights()[0]                        # Dense weights: 1280 features x 5 stages


def cam_overlay(x, display, cls):
    """Grad-CAM for a pooling -> Dense head: feature maps weighted by the class's Dense weights."""
    A = feat_model.predict(x[None].astype('float32'), verbose=0)[0]
    cam = np.maximum(A @ W[:, cls], 0)
    cam = cv2.resize(cam / (cam.max() + 1e-8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)[:, :, ::-1]
    eye = display.max(axis=2, keepdims=True) > 0
    return ((0.6 * display + 0.4 * heat) * eye).astype(np.uint8)


# ---------------- look and feel ----------------
COLORS = ['#2e7d32', '#9e9d24', '#ef8f00', '#e65100', '#c62828']
CSS = """
.gradio-container {max-width: 1150px !important; margin: auto;}
#header {text-align: center; padding: 18px 10px 4px;}
#header h1 {font-size: 1.9rem; margin-bottom: 4px;}
#header p {opacity: .8; margin: 0;}
.card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px 20px;
       background: var(--block-background-fill);}
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
#footer {text-align: center; font-size: .85rem; opacity: .7; padding: 8px;}
"""
THEME = gr.themes.Soft(primary_hue='blue', neutral_hue='slate')

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


def analyse(image):
    """Full pipeline for one photo: preprocessing -> prediction -> Grad-CAM."""
    if image is None:
        return EMPTY_CARD, None, None, None
    t0 = time.time()
    display, x = preprocess(np.asarray(image.convert('RGB')))
    probs = model.predict(x[None].astype('float32'), verbose=0)[0]
    cls = int(probs.argmax())
    cam = cam_overlay(x, display, cls)
    return result_card(probs, cls, time.time() - t0), {s: float(p) for s, p in zip(STAGES, probs)}, x, cam


# ---------------- layout ----------------
style = dict(theme=THEME, css=CSS)
style_in_blocks = 'css' in inspect.signature(gr.Blocks.__init__).parameters   # differs between Gradio versions
examples = sorted(glob.glob('examples/*.png') + glob.glob('examples/*.jpg'))   # optional folder

with gr.Blocks(title='DR Stage Detection', **(style if style_in_blocks else {})) as demo:
    gr.HTML("<div id='header'><h1>Diabetic Retinopathy Stage Detection</h1>"
            "<p>Upload a retinal fundus photograph to estimate its ICDR stage, with a Grad-CAM explanation.</p></div>")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            img_in = gr.Image(type='pil', label='Fundus photograph', height=330)
            btn = gr.Button('Analyse image', variant='primary', size='lg')
            if examples:
                gr.Examples(examples=[[e] for e in examples], inputs=img_in, label='Example images (click one)')
        with gr.Column(scale=7):
            card = gr.HTML(EMPTY_CARD)
            probs_out = gr.Label(num_top_classes=5, label='Stage probabilities')
            with gr.Row():
                pre_out = gr.Image(label='Preprocessed input (what the model sees)', height=260)
                cam_out = gr.Image(label='Grad-CAM: regions behind the prediction', height=260)
    with gr.Accordion('About the model', open=False):
        gr.Markdown(ABOUT_MD)
    gr.HTML("<div id='footer'>Computer Vision coursework prototype (NIBM) \u00b7 not a medical device \u00b7 not for diagnosis</div>")

    outputs = [card, probs_out, pre_out, cam_out]
    img_in.change(analyse, inputs=img_in, outputs=outputs)
    btn.click(analyse, inputs=img_in, outputs=outputs)

demo.launch(**({} if style_in_blocks else style))
