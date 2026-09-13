---
title: Document Layout Detection
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.38.0
app_file: streamlit_app.py
pinned: false
---

# Constrained Document Layout Detection

RT-DETR fine-tuned on DocLayNet, plus a hand-written reasoning layer for
natural-language questions about a page's layout.

Full source, training scripts, and the FastAPI API this demo shares its
code with: see the main repo.

## Space secrets needed

- `GROQ_API_KEY` - for the Ask tab's reasoning layer
- `MODEL_URL` - direct download link for the trained weights (e.g. a
  GitHub Release asset). Downloaded automatically on first run if
  `MODEL_PATH` doesn't already exist - no need to push the weights file
  into the Space's git repo.
