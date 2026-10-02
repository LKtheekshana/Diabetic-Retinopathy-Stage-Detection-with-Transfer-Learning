"""
Model loading, prediction, test-time augmentation, Grad-CAM explanations and
the clinical decision logic (referral recommendation).
"""
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # hide TensorFlow info logs

import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers

from .config import (KERAS_MODEL_PATH, CLASS_NAMES, NUM_CLASSES,
                     REFERRAL_THRESHOLD, ANY_DR_THRESHOLD, STAGE_INFO)
from .preprocessing import apply_circular_mask


class DRClassifier:
    """Wraps the fine-tuned EfficientNetB0 model saved by the training notebook.

    The model expects RGB float32 images in the 0-255 range: EfficientNet in Keras
    rescales internally, so no extra normalisation is applied here.
    """

    def __init__(self, model_path=KERAS_MODEL_PATH):
        self.model = keras.models.load_model(model_path, compile=False)
        self.backbone = self._find_backbone()
        self._build_gradcam_parts()
        # Warm-up call so the first real prediction is not slowed down by graph tracing
        self.model.predict(np.zeros((1, *self.model.input_shape[1:]), np.float32), verbose=0)

    # ------------------------------------------------------------------
    # Set-up helpers
    # ------------------------------------------------------------------
    def _find_backbone(self):
        """The backbone is the nested Functional model inside the classifier (e.g. 'efficientnetb0')."""
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                return layer
        raise ValueError('Could not find the CNN backbone inside the saved model.')

    def _build_gradcam_parts(self):
        """Split the network into: pre-processing layers -> conv feature extractor -> classification head.

        Grad-CAM needs gradients of the class score w.r.t. the last convolutional
        feature map, so the model is rebuilt as two connected sub-models.
        """
        pos = self.model.layers.index(self.backbone)
        self.pre_layers = [l for l in self.model.layers[:pos] if not isinstance(l, layers.InputLayer)]

        last_conv = next(l for l in reversed(self.backbone.layers) if len(l.output.shape) == 4)
        self.last_conv_name = last_conv.name
        self.conv_model = keras.Model(self.backbone.inputs, last_conv.output)

        head_input = keras.Input(shape=last_conv.output.shape[1:])
        x = head_input
        for layer in self.model.layers[pos + 1:]:
            x = layer(x)
        self.head_model = keras.Model(head_input, x)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, images, tta=False):
        """Return class probabilities for a batch of preprocessed uint8 images (N, 224, 224, 3).

        With tta=True the predictions of the original + horizontally / vertically /
        doubly flipped images are averaged (test-time augmentation).
        """
        x = np.asarray(images, dtype=np.float32)
        if x.ndim == 3:
            x = x[None]
        variants = [x] if not tta else [x, x[:, :, ::-1], x[:, ::-1, :], x[:, ::-1, ::-1]]
        probs = [self.model.predict(np.ascontiguousarray(v), verbose=0) for v in variants]
        return np.mean(probs, axis=0)

    def gradcam(self, image, class_index=None):
        """Compute a Grad-CAM heatmap (values 0-1) for one preprocessed image."""
        x = tf.convert_to_tensor(image[None].astype(np.float32))
        for layer in self.pre_layers:
            x = layer(x)
        with tf.GradientTape() as tape:
            conv_out = self.conv_model(x, training=False)
            tape.watch(conv_out)
            preds = self.head_model(conv_out, training=False)
            if class_index is None:
                class_index = int(tf.argmax(preds[0]))
            class_score = preds[:, class_index]
        grads = tape.gradient(class_score, conv_out)
        weights = tf.reduce_mean(grads, axis=(1, 2))               # importance of each feature channel
        cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1))[0]
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return cam.numpy()

    def analyse(self, image, tta=False, explain=True):
        """Full analysis of one preprocessed image -> dictionary used by the user interface."""
        start = time.perf_counter()
        probs = self.predict_proba(image, tta=tta)[0]
        latency_ms = (time.perf_counter() - start) * 1000

        result = interpret(probs)
        result['latency_ms'] = latency_ms
        if explain:
            heat = self.gradcam(image, result['stage_index'])
            result['heatmap'] = heat
            result['overlay'] = overlay_heatmap(image, heat)
        return result


# ---------------------------------------------------------------------------
# Clinical interpretation
# ---------------------------------------------------------------------------
def interpret(probs):
    """Turn the 5 stage probabilities into a stage, confidence and screening decision.

    Two binary screening scores are derived by summing stage probabilities:
      * P(any DR)        = P(Mild) + P(Moderate) + P(Severe) + P(Proliferative)
      * P(referable DR)  = P(Moderate) + P(Severe) + P(Proliferative)
    They are compared with thresholds tuned on the validation set (Youden's J),
    which gave 93% sensitivity / 89% specificity for referable DR on the test set.
    """
    probs = np.asarray(probs, dtype=float)
    stage = int(probs.argmax())
    p_any = float(probs[1:].sum())
    p_ref = float(probs[2:].sum())
    refer = p_ref >= REFERRAL_THRESHOLD

    # Entropy-based uncertainty: 0 = completely certain, 1 = uniform over all 5 classes
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(NUM_CLASSES))
    sorted_p = np.sort(probs)[::-1]
    low_confidence = sorted_p[0] < 0.5 or (sorted_p[0] - sorted_p[1]) < 0.15

    return {
        'probs': probs,
        'stage_index': stage,
        'stage': CLASS_NAMES[stage],
        'confidence': float(probs[stage]),
        'description': STAGE_INFO[CLASS_NAMES[stage]][0],
        'follow_up': STAGE_INFO[CLASS_NAMES[stage]][1],
        'p_any_dr': p_any,
        'any_dr': p_any >= ANY_DR_THRESHOLD,
        'p_referable': p_ref,
        'refer': refer,
        'uncertainty': entropy,
        'low_confidence': bool(low_confidence),
    }


def overlay_heatmap(img, heatmap, alpha=0.45):
    """Blend a JET-coloured Grad-CAM heatmap over the preprocessed image."""
    heat = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img.astype(np.uint8), 1 - alpha, heat, alpha, 0)
    return apply_circular_mask(blended)
