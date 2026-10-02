"""
Image preprocessing for retinal fundus photographs.

This is the same pipeline used to build the training cache in the notebook:

    crop dark border -> pad to square -> resize (224x224)
    -> bilateral denoise -> CLAHE contrast enhancement -> unsharp-mask edge enhancement
    -> circular field-of-view mask

Using identical preprocessing at inference time is essential: any difference
would shift the input distribution and silently reduce accuracy.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import IMG_SIZE, PREPROCESS_MODE


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_image(file_or_path):
    """Read an uploaded file / path into an RGB uint8 numpy array.

    PIL is used (instead of cv2.imread) so that Streamlit's in-memory uploads work
    and EXIF orientation from phone/camera images is respected.
    """
    img = Image.open(file_or_path)
    img = ImageOps.exif_transpose(img).convert('RGB')
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Individual pipeline steps (identical to the training notebook)
# ---------------------------------------------------------------------------
def crop_dark_border(img, tol=10):
    """Remove the black background around the circular retina."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero((gray > tol).astype(np.uint8))
    if coords is None:          # completely dark image - leave it unchanged
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


def pad_to_square(img):
    """Pad with black so the aspect ratio is preserved when resizing."""
    h, w = img.shape[:2]
    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    return cv2.copyMakeBorder(img, top, size - h - top, left, size - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def denoise(img):
    """Edge-preserving bilateral filter to suppress sensor noise."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)


def apply_clahe(img, clip_limit=2.0, tile=8):
    """Contrast Limited Adaptive Histogram Equalisation on the L channel of LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def edge_enhance(img, amount=0.6, sigma=3):
    """Unsharp masking - sharpens vessels, microaneurysms and exudates."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def ben_graham(img, sigma=None):
    """Alternative 'Ben Graham' local-colour normalisation (not used by the final model)."""
    sigma = sigma or img.shape[0] / 30
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)


def apply_circular_mask(img, scale=0.96):
    """Black out the corners so only the circular field of view remains."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * scale), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess_image(img, size=IMG_SIZE, mode=PREPROCESS_MODE):
    """Apply the full training pipeline to an RGB uint8 image and return a (size, size, 3) uint8 array."""
    img = np.ascontiguousarray(img[..., :3])
    img = pad_to_square(crop_dark_border(img))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if mode == 'clahe_edge':
        img = edge_enhance(apply_clahe(denoise(img)))
    elif mode == 'ben_graham':
        img = ben_graham(img)
    elif mode != 'raw':
        raise ValueError(f'Unknown preprocessing mode: {mode}')
    return apply_circular_mask(img)


def pipeline_steps(img, size=IMG_SIZE):
    """Return every intermediate stage of the pipeline, for display in the app."""
    cropped = cv2.resize(pad_to_square(crop_dark_border(img)), (size, size), interpolation=cv2.INTER_AREA)
    den = denoise(cropped)
    clahe = apply_clahe(den)
    edge = edge_enhance(clahe)
    final = apply_circular_mask(edge)
    canny = cv2.Canny(cv2.cvtColor(final, cv2.COLOR_RGB2GRAY), 40, 110)
    return [('1. Crop + pad + resize', cropped), ('2. Bilateral denoise', den),
            ('3. CLAHE contrast', clahe), ('4. Edge enhancement', edge),
            ('5. Final (masked) - model input', final), ('Canny edge map (for reference)', canny)]


# ---------------------------------------------------------------------------
# Input quality check
# ---------------------------------------------------------------------------
def quality_check(img):
    """Simple heuristics that flag images unlikely to be usable fundus photographs.

    Returns a list of warning strings (empty list = no problems detected).
    The model was trained only on fundus photos, so it will still output a
    stage for any image - these checks help the user notice bad inputs.
    """
    warnings = []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2GRAY)
    retina = gray[gray > 10]

    if min(h, w) < IMG_SIZE:
        warnings.append(f'Low resolution ({w}x{h}); at least {IMG_SIZE}x{IMG_SIZE} is recommended.')
    if retina.size < 0.2 * gray.size:
        warnings.append('Image is mostly dark - the retina may be missing or badly under-exposed.')
    elif retina.mean() < 35:
        warnings.append('Retina appears very dark (under-exposed); the prediction may be unreliable.')
    elif retina.mean() > 200:
        warnings.append('Retina appears very bright (over-exposed); the prediction may be unreliable.')
    else:
        # Fundus photographs are dominated by red/orange tones.
        r, g, b = [img[..., c][gray > 10].mean() for c in range(3)]
        if not (r > g and r > b):
            warnings.append('Colours do not look like a typical fundus photograph (red channel is not dominant).')

    sharpness = cv2.Laplacian(cv2.resize(gray, (512, 512)), cv2.CV_64F).var()
    if sharpness < 15:
        warnings.append(f'Image looks blurry (sharpness score {sharpness:.0f}); consider re-capturing.')
    return warnings
