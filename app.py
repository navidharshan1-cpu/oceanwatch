"""OceanWatch: underwater marine litter detection with YOLO26-l (CNN) and RT-DETR-l (transformer).

Streamlit app for the MAAI901 project. Run locally with:  streamlit run app.py
"""
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- settings
GITHUB_REPO = "navidharshan1-cpu/oceanwatch"   # <-- CHANGE THIS to your GitHub "username/repository"
RELEASE_TAG = "v1.0"                               # the GitHub Release that holds the two model files

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = ROOT / "samples" if (ROOT / "samples").is_dir() else ROOT
RESULT_DIR = ROOT / "results" if (ROOT / "results").is_dir() else ROOT
NAMES = ["trash", "bio", "rov"]
LABELS = {"trash": "Litter", "bio": "Marine life", "rov": "ROV parts"}
COLORS = {0: (230, 126, 34), 1: (46, 204, 113), 2: (52, 152, 219)}  # RGB: trash orange, bio green, rov blue
MODEL_FILES = {"YOLO26-l": "yolo26l_3cls.pt", "RT-DETR-l": "rtdetr_l_3cls.pt"}

st.set_page_config(page_title="OceanWatch", page_icon="🌊", layout="wide")


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner=False)
def load_models():
    """Load both trained models once. Downloads the weights from the GitHub Release if they are not in models/."""
    from ultralytics import RTDETR, YOLO

    MODEL_DIR.mkdir(exist_ok=True)
    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if not path.exists():
            url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{fname}"
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        models[name] = (YOLO if name.startswith("YOLO") else RTDETR)(str(path))
    return models


def check_models(models):
    """Each model must have exactly the 3 OceanWatch classes; otherwise the wrong file was uploaded."""
    problems = []
    for name, m in models.items():
        classes = [m.names[i] for i in sorted(m.names)]
        if classes != NAMES:
            shown = ", ".join(classes[:6]) + (", ..." if len(classes) > 6 else "")
            problems.append(f"**{MODEL_FILES[name]}** ({name}) has {len(classes)} classes ({shown}), "
                            f"not trash / bio / rov.")
    return problems


# ---------------------------------------------------------------- helpers
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def ground_truth(name, h, w):
    """Boxes from samples/<name>.txt (YOLO format) if this is one of the labelled test images."""
    f = SAMPLE_DIR / f"{Path(name).stem}.txt"
    if not f.exists():
        return None
    boxes = []
    for line in f.read_text().splitlines():
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            boxes.append((int(c), [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return boxes


def match(preds, gt, thr=0.5):
    """Greedy matching by confidence: correct = same class and IoU >= 0.5. Returns TP, FP, FN."""
    used, tp = set(), 0
    for c, box, _ in sorted(preds, key=lambda p: -p[2]):
        best, best_iou = None, thr
        for j, (gc, gbox) in enumerate(gt):
            v = iou(box, gbox)
            if j not in used and gc == c and v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(preds) - tp, len(gt) - tp


def draw(rgb, boxes, scores=None):
    img = rgb.copy()
    thick = max(2, round(min(img.shape[:2]) / 300))
    for k, (c, (x0, y0, x1, y1)) in enumerate(boxes):
        col = COLORS[c]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, thick)
        label = NAMES[c] + (f" {scores[k]:.2f}" if scores is not None else "")
        cv2.putText(img, label, (int(x0) + 2, max(14, int(y0) - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * thick, col, thick)
    return img


def run_models(models, bgr, conf):
    """Run both models on one BGR image. Returns {model: (preds, ms)} with preds = [(class, [x1,y1,x2,y2], conf)]."""
    out = {}
    for name, m in models.items():
        t = time.perf_counter()
        r = m.predict(bgr, imgsz=640, conf=conf, verbose=False)[0]
        ms = (time.perf_counter() - t) * 1000
        preds = [(int(c), b, float(s)) for c, b, s in
                 zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        out[name] = (preds, ms)
    return out


def summary_table(results, gt):
    rows = []
    for name, (preds, ms) in results.items():
        row = {"Model": name}
        for k, cname in enumerate(NAMES):
            row[LABELS[cname]] = sum(c == k for c, *_ in preds)
        row["Avg confidence"] = f"{np.mean([s for *_, s in preds]):.2f}" if preds else "-"
        row["Time (ms, this server)"] = f"{ms:.0f}"
        if gt is not None:
            tp, fp, fn = match(preds, gt)
            row.update({"Correct": tp, "False alarms": fp, "Missed": fn,
                        "Precision": f"{tp / (tp + fp):.0%}" if tp + fp else "-",
                        "Recall": f"{tp / (tp + fn):.0%}" if tp + fn else "-"})
        rows.append(row)
    return pd.DataFrame(rows)


def detections_table(results):
    rows = [{"Model": name, "Class": NAMES[c], "Confidence": round(s, 2),
             "Box (x1, y1, x2, y2)": ", ".join(f"{v:.0f}" for v in b)}
            for name, (preds, _) in results.items() for c, b, s in preds]
    return pd.DataFrame(rows, columns=["Model", "Class", "Confidence", "Box (x1, y1, x2, y2)"])


@st.cache_data
def load_csv(name, **kw):
    f = RESULT_DIR / name
    return pd.read_csv(f, **kw) if f.exists() else None


# ---------------------------------------------------------------- page
st.title("🌊 OceanWatch: underwater marine litter detection")
st.markdown(
    "Compares **YOLO26-l** (CNN) and **RT-DETR-l** (transformer), both trained on the "
    "[TrashCan 1.0](https://conservancy.umn.edu/handle/11299/214865) dataset of deep-sea ROV images. "
    "Classes: :orange[**trash**], :green[**bio**] (marine life), :blue[**rov**] (the robot's own parts).")

tab_detect, tab_perf, tab_about = st.tabs(["🔍 Detect litter", "📊 Model performance", "ℹ️ About"])

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05,
                     help="Only detections with at least this confidence are shown.")
    source = st.radio("Image source", ["Upload an image", "Example test image (with ground truth)"])
    st.caption("Example images come from the held-out test set, so the app can measure "
               "accuracy against the true labels.")

with tab_detect:
    name, data = None, None
    if source == "Upload an image":
        up = st.file_uploader("Underwater image", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            name, data = up.name, up.getvalue()
    else:
        samples = sorted(p.name for p in SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []
        if samples:
            name = st.selectbox("Example test image", samples)
            data = (SAMPLE_DIR / name).read_bytes()
        else:
            st.warning("No example images found in the samples/ folder.")

    if data is None:
        st.info("Upload an underwater image, or choose an example test image in the sidebar.")
    else:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read this file as an image.")
            st.stop()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gt = ground_truth(name, h, w)

        with st.spinner("Loading the models (first time only, about a minute)..."):
            models = load_models()
        problems = check_models(models)
        if problems:
            st.error("Wrong model file in the GitHub Release:\n\n" + "\n\n".join(f"- {p}" for p in problems) +
                     "\n\nUpload the 3-class model (from the yolo26l-3cls / rtdetr-l-3cls folder) to Release "
                     f"{RELEASE_TAG}, then Reboot the app.")
            st.stop()
        with st.spinner("Detecting..."):
            results = run_models(models, bgr, conf)

        cols = st.columns(3 if gt is not None else 2)
        for col, (mname, (preds, _)) in zip(cols, results.items()):
            col.image(draw(rgb, [(c, b) for c, b, _ in preds], [s for *_, s in preds]),
                      caption=f"{mname} prediction", use_container_width=True)
        if gt is not None:
            cols[2].image(draw(rgb, gt), caption="Ground truth (correct answer)", use_container_width=True)
            st.success(f"Test image with known labels ({len(gt)} objects): accuracy for this image is measured below.")
        else:
            st.info("This image has no ground-truth labels, so its accuracy cannot be measured. "
                    "The overall test-set accuracy is in the **Model performance** tab.")

        st.subheader("Results for this image")
        st.dataframe(summary_table(results, gt), hide_index=True, use_container_width=True)
        st.subheader("Every detection with its confidence score")
        st.dataframe(detections_table(results), hide_index=True, use_container_width=True)

with tab_perf:
    st.subheader("Accuracy and speed on the test set")
    st.caption("1,081 images from 57 ROV videos that were never used for training or model selection. "
               "Speed measured on a Tesla T4 GPU, one image at a time.")
    res = load_csv("final_test_results.csv", index_col=0)
    if res is not None:
        res = res.astype(float)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Litter AP50: RT-DETR-l", f"{res.loc['trash AP50', 'RT-DETR-l']:.1%}",
                  f"{(res.loc['trash AP50', 'RT-DETR-l'] - res.loc['trash AP50', 'YOLO26-l']) * 100:+.1f} pts vs YOLO")
        c2.metric("Litter AP50: YOLO26-l", f"{res.loc['trash AP50', 'YOLO26-l']:.1%}")
        c3.metric("Speed: YOLO26-l", f"{res.loc['FPS', 'YOLO26-l']:.1f} FPS")
        c4.metric("Speed: RT-DETR-l", f"{res.loc['FPS', 'RT-DETR-l']:.1f} FPS")
        st.dataframe(res.round(3), use_container_width=True)
    else:
        st.warning("results/final_test_results.csv not found.")

    exp = load_csv("experiment_conditions_enhancement.csv")
    if exp is not None:
        st.subheader("Image conditions and enhancement")
        st.caption("Test images split at the median contrast (clear vs challenging), and tested again after "
                   "CLAHE and white balance + CLAHE. The models were not retrained.")
        metric = st.selectbox("Metric", ["trash AP50", "mAP50", "mAP50-95"])
        t = exp.pivot_table(index=["images", "enhancement"], columns="model", values=metric)
        t = t.reindex(pd.MultiIndex.from_product(
            [["all", "clear", "challenging"], ["original", "CLAHE", "white balance + CLAHE"]]))
        st.dataframe(t[[c for c in ["YOLO26-l", "RT-DETR-l"] if c in t.columns]].round(3),
                     use_container_width=True)

with tab_about:
    st.markdown("""
**OceanWatch** compares a CNN detector and a transformer detector for finding marine litter in images from
underwater robots (ROVs).

- **Data:** TrashCan 1.0 (Hong, Fulton & Sattar, 2020), 7,212 images from JAMSTEC deep-sea ROV videos.
  The 22 original classes are merged into **trash**, **bio** and **rov**.
- **Fair split:** the official split puts frames from 113 of its 115 test videos into training too.
  We split by **whole video** (70/15/15), so the test set contains only unseen dives.
- **Models:** YOLO26-l (Ultralytics) and RT-DETR-l (Zhao et al., 2024), COCO-pretrained, trained 40 epochs
  with identical image size, batch, seed and data.
- **Main finding:** RT-DETR-l finds litter more accurately; YOLO26-l is about 3x faster.
- **Limits:** one training run per model; all images from one source (deep-sea, Sea of Japan) with on-screen
  text; accuracy drops on very different water, such as bright shallow reefs.

*MAAI901 Deep Learning for Visual Data, University of Wollongong in Dubai.*
""")
