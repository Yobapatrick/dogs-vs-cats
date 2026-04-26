"""Streamlit demo — Dogs vs Cats live comparison.

Loads both trained models (LogReg baseline and CNN) and runs them
side-by-side on user-uploaded images. Mirrors the narrative of the
README: how much does the CNN architecture actually buy you?

Run locally:
    streamlit run app/streamlit_app.py

Deploy:
    See app/README.md for Hugging Face Spaces instructions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Make sure the project root is on sys.path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import Predictor  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGREG_CKPT = CHECKPOINTS_DIR / "logreg-best.ckpt"
CNN_CKPT = CHECKPOINTS_DIR / "cnn-best.ckpt"

CLASS_NAMES = ("cat", "dog")
EMOJI = {"cat": "🐱", "dog": "🐶"}
COLOR = {"cat": "#f97316", "dog": "#3b82f6"}

st.set_page_config(
    page_title="Dogs vs Cats — LogReg vs CNN",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CACHED MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def load_predictors() -> tuple[Predictor | None, Predictor | None, str | None]:
    """Load both trained models once per session. Returns (logreg, cnn, error)."""
    if not LOGREG_CKPT.exists() or not CNN_CKPT.exists():
        return None, None, (
            f"Checkpoints not found at {CHECKPOINTS_DIR}/. "
            "Run `make train-all` first, or place pre-trained .ckpt files there."
        )

    try:
        logreg = Predictor.from_checkpoint(LOGREG_CKPT, model_type="logreg", class_names=CLASS_NAMES)
        cnn = Predictor.from_checkpoint(CNN_CKPT, model_type="cnn", class_names=CLASS_NAMES)
        return logreg, cnn, None
    except Exception as exc:  # noqa: BLE001
        return None, None, f"Failed to load models: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_model_card(name: str, result, accent: str) -> None:
    """Render a single-model prediction card."""
    label = result.label
    conf = result.confidence
    other = "dog" if label == "cat" else "cat"

    st.markdown(
        f"""
        <div style='
            background: linear-gradient(135deg, {accent}15 0%, {accent}05 100%);
            border: 2px solid {accent}40;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 0.5rem;
        '>
            <div style='font-size: 0.85rem; color: #64748b; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em;'>
                {name}
            </div>
            <div style='font-size: 2.5rem; font-weight: 800; color: {accent};
                        margin: 0.3rem 0; line-height: 1;'>
                {EMOJI[label]} {label.upper()}
            </div>
            <div style='font-size: 1.1rem; color: #475569; font-weight: 600;'>
                {conf * 100:.1f}% confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence bars for both classes
    for cls in CLASS_NAMES:
        p = result.probabilities[cls]
        bar_color = COLOR[cls] if cls == label else "#cbd5e1"
        st.markdown(
            f"""
            <div style='margin-bottom: 0.4rem;'>
                <div style='display: flex; justify-content: space-between;
                            font-size: 0.85rem; color: #475569; margin-bottom: 0.15rem;'>
                    <span>{EMOJI[cls]} {cls.capitalize()}</span>
                    <span style='font-weight: 600;'>{p * 100:.1f}%</span>
                </div>
                <div style='background: #f1f5f9; height: 6px; border-radius: 3px; overflow: hidden;'>
                    <div style='background: {bar_color}; width: {p * 100}%;
                                height: 100%; transition: width 0.4s ease;'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_agreement(lr_result, cnn_result) -> None:
    """Compare the two models and highlight agreement/disagreement."""
    agree = lr_result.label == cnn_result.label
    if agree:
        color, icon, msg = "#16a34a", "✓", "Models agree"
    else:
        color, icon, msg = "#dc2626", "✗", "Models disagree"

    st.markdown(
        f"""
        <div style='
            background: {color}10;
            border-left: 4px solid {color};
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        '>
            <span style='font-size: 1.3rem; font-weight: 700; color: {color};'>{icon} {msg}</span>
            <span style='color: #64748b; margin-left: 0.5rem;'>
                — LogReg says <b>{lr_result.label}</b> ({lr_result.confidence:.0%}),
                CNN says <b>{cnn_result.label}</b> ({cnn_result.confidence:.0%})
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    # Header
    st.markdown(
        """
        <div style='text-align: center; padding-bottom: 1rem;'>
            <h1 style='margin: 0;'>🐾 Dogs vs Cats — Live Model Comparison</h1>
            <p style='color: #64748b; font-size: 1.05rem; margin: 0.5rem 0 0 0;'>
                A 12k-parameter linear baseline vs a 390k-parameter CNN, side-by-side.
                <br/>The gap quantifies what convolutional inductive bias actually buys you.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar — project context
    with st.sidebar:
        st.markdown("### 📊 Test-set performance")
        st.markdown(
            """
            | Metric    | LogReg | **CNN** |
            |-----------|:------:|:-------:|
            | Accuracy  | 58.7%  | **90.1%** |
            | F1-Score  | 55.7%  | **89.7%** |
            | ROC-AUC   | 0.623  | **0.965** |
            """
        )
        st.markdown("---")
        st.markdown("### 🔬 How it works")
        st.markdown(
            """
            Both models share the exact same:
            - Data pipeline
            - Loss function (cross-entropy)
            - Lightning Trainer

            Only the **architecture** differs. Upload your own image to see how
            each model handles it.
            """
        )
        st.markdown("---")
        st.markdown(
            "🐙 [View source on GitHub](https://github.com/Yobapatrick/dogs-vs-cats)"
        )

    # Load models
    logreg, cnn, err = load_predictors()
    if err:
        st.error(err)
        st.info(
            "👉 To run this demo, you need the trained checkpoints. "
            "Clone the repo and run `make train-all`, or download pretrained "
            "weights from the GitHub release page."
        )
        st.stop()

    # File uploader
    st.markdown("### 📷 Upload an image")
    uploaded = st.file_uploader(
        "Drop a cat or dog photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    # Demo samples when no upload
    if uploaded is None:
        samples_dir = PROJECT_ROOT / "app" / "assets"
        sample_files = sorted(samples_dir.glob("*.jp*g")) if samples_dir.exists() else []
        if sample_files:
            st.markdown("…or try one of these samples:")
            cols = st.columns(min(4, len(sample_files)))
            for col, sample in zip(cols, sample_files[:4], strict=False):
                with col:
                    st.image(str(sample), use_column_width=True)
                    if st.button(f"Use {sample.name}", key=sample.name, use_container_width=True):
                        st.session_state["pending_sample"] = str(sample)
                        st.rerun()
        st.stop()

    # Resolve image (from upload or pending sample)
    if "pending_sample" in st.session_state:
        image = Image.open(st.session_state.pop("pending_sample")).convert("RGB")
    else:
        image = Image.open(uploaded).convert("RGB")

    # Layout: image on the left, predictions on the right
    img_col, pred_col = st.columns([1, 1.3])

    with img_col:
        st.image(image, caption="Input image", use_column_width=True)
        st.caption(f"Size: {image.size[0]}×{image.size[1]} px → resized to 64×64 for inference")

    with pred_col:
        with st.spinner("Running inference on both models…"):
            lr_result = logreg.predict(image)
            cnn_result = cnn.predict(image)

        render_agreement(lr_result, cnn_result)

        c1, c2 = st.columns(2)
        with c1:
            render_model_card("Logistic Regression (baseline)", lr_result, "#3b82f6")
        with c2:
            render_model_card("CNN (4-block)", cnn_result, "#f97316")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>
            Built with ⚡ PyTorch Lightning · 🎈 Streamlit
            · <a href='https://github.com/Yobapatrick/dogs-vs-cats' style='color: #94a3b8;'>GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
