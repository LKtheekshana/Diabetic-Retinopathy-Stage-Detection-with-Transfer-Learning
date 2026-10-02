# Diabetic Retinopathy Screening Assistant – Prototype

A Streamlit web app that uses the EfficientNetB0 transfer-learning model trained in
`../diabetic-retinopathy-stage-detection.ipynb` (APTOS 2019, Kaggle) to classify retinal
fundus photographs into five DR stages and give a referral recommendation.

## Folder structure
```
prototype/
├── app.py                 # Streamlit user interface
├── predict.py             # command-line inference (batch, CSV export)
├── dr_app/
│   ├── config.py          # paths, class names, thresholds (read from results_summary.json)
│   ├── preprocessing.py   # identical preprocessing pipeline to training + input quality check
│   └── inference.py       # model loading, TTA, Grad-CAM, referral logic
├── models/                # final_dr_model.keras, TFLite model, results_summary.json
├── assets/figures/        # evaluation figures from the training notebook
└── sample_images/         # put example fundus images here for one-click demos
```

## Setup and run
```bash
pip install -r requirements.txt
streamlit run app.py            # or double-click run_app.bat
```
The app opens at http://localhost:8501.

Command line:
```bash
python predict.py sample_images/*.png --csv results.csv
```

## Features
- **Analyse an image**: stage and confidence, probability for every stage, a referral decision
  (P(stage ≥ 2) against a threshold tuned on the validation set), a Grad-CAM heatmap,
  every preprocessing step, and a downloadable text report.
- **Batch screening**: analyse many images at once, see a summary and download a CSV.
- **Model performance**: test-set metrics and the training and evaluation figures.
- **Input quality check**: warns about dark, over-exposed, blurry, low-resolution or non-fundus images.
- **Test-time augmentation** that you can switch on in the sidebar.

*Research prototype for decision support only. It is not a certified medical device.*
