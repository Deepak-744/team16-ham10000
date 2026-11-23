# Skin Lesion Classification Web App

The React frontend sends dermoscopic images to a FastAPI backend that runs Team 16's ConvNeXt-Tiny checkpoint on CPU. The interface shows scores for the seven HAM10000 classes, a Grad-CAM overlay when requested, and a downloadable PDF summary. Predictions are for research and education. The app scores one view per upload. The benchmark averages logits from the original image and its flipped views.

[Live demo](https://skin-lesion-classification-umber.vercel.app/) · [Final report](../report/Team16_Report.pdf)

## Run locally

Use Python 3.12 and Node.js 24.

### 1. Provide the checkpoint

Obtain Team 16's epoch-20 ConvNeXt-Tiny checkpoint. From the repository root, put it at `webapp/backend/models/best_model.pth`, or set `MODEL_CHECKPOINT_PATH` to its absolute path before starting the backend. The team's `.pth.zip` file is a PyTorch checkpoint and can be used directly with `MODEL_CHECKPOINT_PATH`.

### 2. Start the backend

From the repository root:

```sh
cd webapp
python -m venv .venv
```

Activate the environment with `. .venv/bin/activate` on macOS/Linux or `.\.venv\Scripts\Activate.ps1` in PowerShell. Then run:

```sh
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Check [backend health](http://127.0.0.1:8000/api/health) for `"status": "ok"` and `"model_loaded": true`. The [API documentation](http://127.0.0.1:8000/docs) lists the upload and reporting endpoints.

### 3. Start the frontend

In another terminal, from the repository root:

```sh
cd webapp/frontend
npm ci
npm run dev
```

Open the URL printed by Vite. The app uses `http://localhost:8000` by default in local development. To specify a different backend, set `VITE_API_URL` in the frontend environment or `webapp/frontend/.env.local`. Use the base URL without `/api`. The page loads Tailwind CSS and fonts from the external CDNs configured in `index.html`.

### Test checkpoint loading, inference, Grad-CAM, and PDF generation

From `webapp/` with the Python environment activated:

```sh
python backend/test_pipeline.py path/to/dermoscopic-image.jpg
```

Add `--output test_report.pdf` to save the generated report. You can also put `.jpg` demo images in `webapp/sample_images/` to make them selectable from the upload page.
