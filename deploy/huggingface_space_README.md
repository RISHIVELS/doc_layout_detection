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
- `MODEL_PATH` - only if the weights aren't at the default `./weights/best.pt`
