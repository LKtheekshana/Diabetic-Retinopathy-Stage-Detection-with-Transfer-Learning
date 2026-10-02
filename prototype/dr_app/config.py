"""
Central configuration for the Diabetic Retinopathy (DR) screening prototype.

Every value here must match the training notebook
(diabetic-retinopathy-stage-detection.ipynb) so that images are processed
and interpreted exactly as they were during training.
"""
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / 'models'
FIGURE_DIR = ROOT_DIR / 'assets' / 'figures'
SAMPLE_DIR = ROOT_DIR / 'sample_images'

KERAS_MODEL_PATH = MODEL_DIR / 'final_dr_model.keras'
TFLITE_MODEL_PATH = MODEL_DIR / 'dr_model_quantized.tflite'
RESULTS_PATH = MODEL_DIR / 'results_summary.json'

# ---------------------------------------------------------------------------
# Model / preprocessing constants (same as CONFIG in the notebook)
# ---------------------------------------------------------------------------
IMG_SIZE = 224                       # EfficientNetB0 native input resolution
PREPROCESS_MODE = 'clahe_edge'       # pipeline used for the final model

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
NUM_CLASSES = len(CLASS_NAMES)

# Short clinical description and suggested follow-up for each stage
# (based on the International Clinical Diabetic Retinopathy severity scale).
STAGE_INFO = {
    'No DR': ('No visible signs of diabetic retinopathy.',
              'Routine annual eye screening.'),
    'Mild': ('Microaneurysms only (small bulges in retinal blood vessels).',
             'Re-screen in 6-12 months; keep blood sugar and blood pressure under control.'),
    'Moderate': ('More than microaneurysms: dot/blot haemorrhages, hard exudates or cotton-wool spots.',
                 'Refer to an ophthalmologist within 3-6 months.'),
    'Severe': ('Extensive haemorrhages, venous beading or intraretinal microvascular abnormalities.',
               'Urgent referral to an ophthalmologist (within weeks).'),
    'Proliferative DR': ('Growth of new abnormal blood vessels (neovascularisation) or vitreous haemorrhage.',
                         'Immediate referral - high risk of vision loss.'),
}

# Colour used for each stage in the user interface (green -> red severity scale)
STAGE_COLOURS = ['#2e7d32', '#9e9d24', '#f9a825', '#ef6c00', '#c62828']


def load_results():
    """Load the evaluation summary written at the end of training (results_summary.json)."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


RESULTS = load_results()

# Decision thresholds chosen on the validation set with Youden's J statistic.
# Fallback values are used only if the summary file is missing.
_screening = RESULTS.get('binary_screening', {})
REFERRAL_THRESHOLD = float(RESULTS.get('referral_threshold', 0.4466))   # P(stage >= 2)
ANY_DR_THRESHOLD = float(_screening.get('Any DR (stage >= 1)', {}).get('Threshold (from val)', 0.1896))
