# Feet Grounding Classification

This project classifies whether a seated person’s feet are **Fully Grounded**, **Partially Grounded**, or **Not Grounded** from a side‑profile image. It provides two modes:

- **MediaPipe + Logic**: local pose‑based rules
- **API AI**: Gemini API classification (cloud)

## Project Structure
```
/feet-check/
├── app.py
├── core/
│   ├── detector.py      # MediaPipe pose + Gemini API client
│   ├── grounding.py     # Grounding logic (rule‑based)
│   └── classifier.py    # Skeleton/label rendering helpers
├── utils/
│   └── visualizer.py    # Low‑level drawing utilities
└── requirements.txt
```

## Setup
Create and activate a Python environment (recommended):

```bash
conda create -n feetcheck python=3.11
conda activate feetcheck
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Usage
1. Open the Streamlit UI in your browser.
2. Upload a side‑profile seated image.
3. Click one of the buttons:
   - **MediaPipe + Logic** (local rule‑based)
   - **API AI** (Gemini API)

## Gemini API Setup (optional)
If you want to use **API AI** (Gemini):

1) Create a `.env` file in `/feet-check/`:
```
GEMINI_API_KEY=YOUR_KEY_HERE
```

Or export it in your shell:
```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

> Note: Gemini free tier has quota limits. If you exceed it you will get a 429 error.

## Output
The app returns JSON with:
- `classification`
- `confidence`
- `left_foot` / `right_foot`
- `notes`

MediaPipe mode also shows an annotated image with skeleton + heel/toe points.

## How MediaPipe + Logic Works
1. Detect pose landmarks using MediaPipe (Tasks API).
2. Use heel/toe landmarks (indices 29–32) to infer contact.
3. Estimate a per‑foot ground line from the max of heel/toe y.
4. Apply simple thresholds to classify each foot and the person.

This is fast but can be inaccurate with occlusion or complex shoes.

## Notes / Limitations
- **MediaPipe** can be wrong if heel/toe landmarks are inaccurate.
- **API AI** (Gemini) tends to be more accurate in real‑world photos.
- Images with heavy occlusion, shadows, or unusual footwear are harder.

## Troubleshooting
- If MediaPipe model download fails, ensure network access.
- If Gemini returns 404, list available models and update the model name.
- If Gemini returns 429, you hit free tier limits.

## Requirements
See `/feet-check/requirements.txt`.
