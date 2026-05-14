# Streamlit Demo — Dogs vs Cats

Live side-by-side comparison of the Logistic Regression baseline and the CNN.
Upload any image, see what each model predicts, and watch the confidence bars
diverge whenever the linear model gets stuck.

---

## 🏃 Run locally

```bash
# From the project root
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

> 📌 Make sure you have trained checkpoints at `checkpoints/logreg-best.ckpt`
> and `checkpoints/cnn-best.ckpt` first. If you don't, run `make train-all`
> from the project root, or drop pre-trained `.ckpt` files into `checkpoints/`.

---

## 🚀 Deploy to Hugging Face Spaces

Hugging Face Spaces gives you a free, persistent URL like
`https://huggingface.co/spaces/PatrickYoba/dogs-vs-cats`.
Perfect for putting on a CV or LinkedIn.

### Step 1 — Create the Space

1. Go to https://huggingface.co/new-space
2. Choose:
   - **Space name**: `dogs-vs-cats`
   - **License**: MIT
   - **SDK**: **Streamlit**
   - **Hardware**: CPU basic (free) is fine — inference is cheap at 64×64
   - **Visibility**: Public

### Step 2 — Prepare a deployment-ready copy

Spaces expects a flat layout: `app.py` and `requirements.txt` at the **repo root**,
plus a YAML header inside `README.md`. The script below assembles that layout
without polluting your main repo:

```bash
# From the project root
mkdir -p ../hf-space-deploy
cd ../hf-space-deploy

# Copy source code and assets
cp -r ../dogs-vs-cats/src .
cp -r ../dogs-vs-cats/app/assets .   # demo sample images, if any
cp    ../dogs-vs-cats/app/streamlit_app.py app.py
cp    ../dogs-vs-cats/app/requirements.txt requirements.txt

# Copy checkpoints (this is the heavy step — see Git LFS section below)
mkdir -p checkpoints
cp ../dogs-vs-cats/checkpoints/logreg-best.ckpt checkpoints/
cp ../dogs-vs-cats/checkpoints/cnn-best.ckpt    checkpoints/

# Adapt the import path inside app.py so it works at repo root
sed -i 's|PROJECT_ROOT = Path(__file__).resolve().parents\[1\]|PROJECT_ROOT = Path(__file__).resolve().parent|' app.py
```

### Step 3 — Add the Spaces YAML header to README

Create `README.md` at the root of `hf-space-deploy/`:

```markdown
---
title: Dogs vs Cats — LogReg vs CNN
emoji: 🐾
colorFrom: orange
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# Dogs vs Cats — Live Model Comparison

Side-by-side inference: a 12k-parameter logistic regression baseline
versus a 390k-parameter CNN. Upload an image and see how each model
handles it.

Full source code, training pipeline, and methodology:
👉 https://github.com/PatrickYoba/dogs-vs-cats
```

### Step 4 — Push to Spaces

```bash
# In hf-space-deploy/
git init
git lfs install
git lfs track "*.ckpt"
git add .gitattributes
git add .
git commit -m "Initial Spaces deployment"

git remote add origin https://huggingface.co/spaces/PatrickYoba/dogs-vs-cats
git push -u origin main
```

The Space will build automatically. First build takes 3–5 minutes (downloading
PyTorch wheels). Subsequent builds are much faster thanks to layer caching.

### Step 5 — Verify and share

- Visit `https://huggingface.co/spaces/PatrickYoba/dogs-vs-cats`
- Open the **Logs** tab if the build fails — most errors are missing dependencies
- Embed in your GitHub README:

```markdown
[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/PatrickYoba/dogs-vs-cats)
```

---

## ⚠️ About Git LFS and checkpoint size

PyTorch Lightning checkpoints can be 5–50 MB depending on the architecture.
GitHub blocks pushes over 100 MB. Two options:

1. **Use Git LFS** (recommended for Spaces): `git lfs track "*.ckpt"`.
   Spaces supports LFS natively.

2. **Don't commit checkpoints** — instead, upload them to Hugging Face as a
   model repo (free), then download them on app startup:

   ```python
   from huggingface_hub import hf_hub_download
   ckpt = hf_hub_download(repo_id="PatrickYoba/dogs-vs-cats-models", filename="cnn-best.ckpt")
   ```

   This is cleaner and what most production Spaces do.

---

## 🧪 Customization ideas

- **Add Grad-CAM overlay** for the CNN — show what the model is looking at
- **Add a "disagree" filter** that keeps only the images where the two models predict differently — useful for finding edge cases
- **Log predictions** to a Hugging Face dataset for active learning
- **Add a "your own URL" input** for image URLs instead of uploads
