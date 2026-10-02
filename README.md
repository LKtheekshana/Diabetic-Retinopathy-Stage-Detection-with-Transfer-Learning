# Diabetic Retinopathy Stage Detection

A Computer Vision project that classifies retinal fundus photographs into five diabetic retinopathy
stages (No DR, Mild, Moderate, Severe, Proliferative DR) using an **EfficientNetB0** CNN with
transfer learning. The model was trained on the **APTOS 2019 Blindness Detection** dataset from Kaggle.

## Repository contents
```
├── diabetic-retinopathy-stage-detection.ipynb   # full training notebook (run on Kaggle GPU)
├── outputs/
│   ├── figures/        # every figure used in the report (EDA, preprocessing, curves, confusion matrix, Grad-CAM...)
│   ├── models/         # final_dr_model.keras, quantised TFLite, best checkpoints, TF SavedModel export
│   └── logs/           # per-epoch training logs, experiment results, test metrics, results_summary.json
└── prototype/          # Streamlit web app that uses the trained model (see prototype/README.md)
```

## Pipeline
1. **Preprocessing**: crop the dark border, pad to a square, resize to 224×224, bilateral denoising,
   CLAHE contrast enhancement, unsharp-mask edge enhancement and a circular mask.
2. **Augmentation and balancing**: random flips, rotation, zoom, brightness and contrast, plus
   oversampling so that minority stages make up at least 50% of the size of the majority class.
3. **Transfer learning**: three backbones compared (EfficientNetB0, ResNet50V2, MobileNetV2), then a
   hyperparameter search and two-phase training: a frozen backbone first, then full fine-tuning.
4. **Training strategy**: early stopping, ReduceLROnPlateau, and checkpoints chosen on validation
   macro-F1 and QWK.
5. **Evaluation**: accuracy and loss curves, precision, recall, F1-score, confusion matrix, ROC curves,
   binary screening performance, error analysis and Grad-CAM.

## Test-set results
| Metric | Value |
|---|---|
| Accuracy | 0.780 |
| Macro F1 | 0.598 |
| Quadratic weighted kappa | 0.853 |
| Macro ROC-AUC | 0.900 |
| Referable DR sensitivity / specificity | 0.933 / 0.893 |
| Any DR sensitivity / specificity | 0.982 / 0.948 |

## Run the prototype
```bash
cd prototype
pip install -r requirements.txt
streamlit run app.py
```

*Research prototype for decision support only. It is not a certified medical device.*
