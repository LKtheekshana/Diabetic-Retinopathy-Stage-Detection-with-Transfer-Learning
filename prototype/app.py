"""
Diabetic Retinopathy Screening Assistant - Streamlit prototype.

Run with:   streamlit run app.py

Features
  * Single-image analysis: DR stage, confidence, referral recommendation,
    Grad-CAM explanation and a step-by-step view of the preprocessing pipeline.
  * Batch screening: analyse many images at once and export the results as CSV.
  * Model performance: test-set metrics and figures produced during training.

This is a research prototype for decision support only - not a medical device.
"""
import datetime as dt
import io

import numpy as np
import pandas as pd
import streamlit as st

from dr_app.config import (CLASS_NAMES, STAGE_COLOURS, FIGURE_DIR, SAMPLE_DIR, RESULTS,
                           REFERRAL_THRESHOLD, ANY_DR_THRESHOLD, IMG_SIZE)
from dr_app.preprocessing import load_image, preprocess_image, pipeline_steps, quality_check

IMAGE_TYPES = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']

st.set_page_config(page_title='DR Screening Assistant', page_icon='👁️', layout='wide')


# ---------------------------------------------------------------------------
# Cached resources (loaded once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner='Loading the trained EfficientNetB0 model...')
def get_classifier():
    # Imported here so the page renders immediately while TensorFlow loads
    from dr_app.inference import DRClassifier
    return DRClassifier()


def list_sample_images():
    """Images placed in sample_images/ are offered as one-click examples."""
    if not SAMPLE_DIR.exists():
        return []
    return sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower().lstrip('.') in IMAGE_TYPES)


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def probability_bars(probs, highlight):
    """Render the 5 stage probabilities as coloured horizontal bars."""
    rows = []
    for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
        weight = '700' if i == highlight else '400'
        rows.append(
            f'<div style="display:flex;align-items:center;margin:4px 0;font-weight:{weight}">'
            f'<div style="width:130px">{i}. {name}</div>'
            f'<div style="flex:1;background:rgba(128,128,128,.15);border-radius:4px;height:18px;margin:0 8px">'
            f'<div style="width:{p * 100:.1f}%;background:{STAGE_COLOURS[i]};height:100%;border-radius:4px"></div></div>'
            f'<div style="width:55px;text-align:right">{p:.1%}</div></div>')
    st.markdown(''.join(rows), unsafe_allow_html=True)


def decision_banner(result):
    """Colour-coded referral recommendation based on P(referable DR)."""
    p_ref = result['p_referable']
    if result['refer']:
        st.error(f"### 🔴 REFER to an ophthalmologist\n"
                 f"Probability of referable DR (moderate or worse): **{p_ref:.0%}** "
                 f"(referral threshold {REFERRAL_THRESHOLD:.0%})")
    elif result['any_dr']:
        st.warning(f"### 🟡 Early signs possible - re-screen sooner\n"
                   f"P(any DR) = **{result['p_any_dr']:.0%}** is above the {ANY_DR_THRESHOLD:.0%} screening "
                   f"threshold, but P(referable DR) = **{p_ref:.0%}** is below the referral threshold.")
    else:
        st.success(f"### 🟢 No referral needed - routine screening\n"
                   f"Probability of referable DR: **{p_ref:.0%}** (below the {REFERRAL_THRESHOLD:.0%} threshold)")


def build_text_report(name, result, warnings):
    """Plain-text report the user can download and attach to a patient record."""
    lines = [
        'DIABETIC RETINOPATHY SCREENING REPORT (research prototype)',
        '=' * 60,
        f'Generated     : {dt.datetime.now():%Y-%m-%d %H:%M}',
        f'Image         : {name}',
        f'Model         : {RESULTS.get("config", {}).get("BACKBONE", "EfficientNetB0")} (transfer learning, APTOS 2019)',
        '',
        f'Predicted stage : {result["stage_index"]} - {result["stage"]}  (confidence {result["confidence"]:.1%})',
        f'Description     : {result["description"]}',
        f'Suggested action: {result["follow_up"]}',
        '',
        f'P(any DR, stage >= 1)       : {result["p_any_dr"]:.1%}  (threshold {ANY_DR_THRESHOLD:.1%})',
        f'P(referable DR, stage >= 2) : {result["p_referable"]:.1%}  (threshold {REFERRAL_THRESHOLD:.1%})',
        f'Referral recommended        : {"YES" if result["refer"] else "NO"}',
        f'Prediction uncertainty      : {result["uncertainty"]:.2f} (0 = certain, 1 = uniform)',
        '',
        'Stage probabilities:',
        *[f'  {i}. {n:<18} {p:6.1%}' for i, (n, p) in enumerate(zip(CLASS_NAMES, result['probs']))],
    ]
    if warnings:
        lines += ['', 'Image quality warnings:', *[f'  - {w}' for w in warnings]]
    lines += ['', 'DISCLAIMER: Decision-support prototype only. Not a certified medical device.',
              'All results must be confirmed by a qualified eye-care professional.']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Sidebar - settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title('👁️ DR Screening Assistant')
    st.caption('EfficientNetB0 + transfer learning, trained on the APTOS 2019 Kaggle dataset.')

    st.subheader('Settings')
    use_tta = st.toggle('Test-time augmentation (TTA)', value=bool(RESULTS.get('use_tta', False)),
                        help='Average predictions over 4 flipped copies of the image. '
                             'Slightly more robust, ~4x slower.')
    show_gradcam = st.toggle('Grad-CAM explanation', value=True,
                             help='Highlight the retinal regions that most influenced the prediction.')
    show_pipeline = st.toggle('Show preprocessing steps', value=True)

    st.divider()
    tm = RESULTS.get('test_metrics', {})
    ref = RESULTS.get('binary_screening', {}).get('Referable DR (stage >= 2)', {})
    if tm:
        st.subheader('Test-set performance')
        c1, c2 = st.columns(2)
        c1.metric('Accuracy', f"{tm['Accuracy']:.1%}")
        c2.metric('QWK', f"{tm['QWK']:.3f}")
        c1.metric('Macro F1', f"{tm['Macro F1']:.3f}")
        c2.metric('ROC-AUC', f"{tm['Macro ROC-AUC']:.3f}")
        if ref:
            c1.metric('Referral sens.', f"{ref['Sensitivity']:.1%}")
            c2.metric('Referral spec.', f"{ref['Specificity']:.1%}")

    st.divider()
    st.caption('⚠️ Research prototype for decision support only. '
               'Not a certified medical device - always confirm with an eye-care professional.')


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title('Diabetic Retinopathy Stage Detection')
tab_single, tab_batch, tab_perf, tab_about = st.tabs(
    ['🔍 Analyse an image', '📂 Batch screening', '📊 Model performance', 'ℹ️ About'])

# ---------------------------------------------------------------- single image
with tab_single:
    col_in, col_hint = st.columns([2, 1])
    with col_in:
        upload = st.file_uploader('Upload a retinal fundus photograph', type=IMAGE_TYPES, key='single')
    samples = list_sample_images()
    with col_hint:
        sample_choice = None
        if samples:
            sample_choice = st.selectbox('...or choose a sample image', ['-'] + [p.name for p in samples])
        else:
            st.info('Tip: copy a few fundus images into the `sample_images/` folder '
                    'to get one-click examples here.')

    source, source_name = None, None
    if upload is not None:
        source, source_name = upload, upload.name
    elif sample_choice and sample_choice != '-':
        source, source_name = SAMPLE_DIR / sample_choice, sample_choice

    if source is None:
        st.markdown('Upload a colour fundus photograph (PNG/JPG) to obtain the predicted DR stage, '
                    'a referral recommendation and a visual explanation of the decision.')
    else:
        try:
            raw = load_image(source)
        except Exception as e:  # corrupted / unsupported file
            st.error(f'Could not read the image: {e}')
            st.stop()

        warnings = quality_check(raw)
        processed = preprocess_image(raw)
        classifier = get_classifier()
        with st.spinner('Analysing...'):
            result = classifier.analyse(processed, tta=use_tta, explain=show_gradcam)

        for w in warnings:
            st.warning(f'Image quality: {w}')

        # --- Images: original / model input / Grad-CAM
        cols = st.columns(3 if show_gradcam else 2)
        cols[0].image(raw, caption=f'Original ({raw.shape[1]}x{raw.shape[0]})', width='stretch')
        cols[1].image(processed, caption=f'Preprocessed model input ({IMG_SIZE}x{IMG_SIZE})', width='stretch')
        if show_gradcam:
            cols[2].image(result['overlay'], caption='Grad-CAM: red = most influential regions', width='stretch')

        # --- Result
        st.divider()
        left, right = st.columns([1, 1])
        with left:
            colour = STAGE_COLOURS[result['stage_index']]
            st.markdown(f"<h2 style='margin-bottom:0'>Stage {result['stage_index']}: "
                        f"<span style='color:{colour}'>{result['stage']}</span></h2>"
                        f"<p style='font-size:1.1rem;margin-top:4px'>Confidence {result['confidence']:.1%}</p>",
                        unsafe_allow_html=True)
            st.markdown(f"**What it means:** {result['description']}")
            st.markdown(f"**Suggested follow-up:** {result['follow_up']}")
            if result['low_confidence']:
                st.info('ℹ️ The model is uncertain between neighbouring stages - '
                        'a manual review by a specialist is recommended.')
            st.caption(f"Inference time: {result['latency_ms']:.0f} ms"
                       f"{' (with TTA)' if use_tta else ''} | uncertainty {result['uncertainty']:.2f}")
        with right:
            st.markdown('**Probability of each stage**')
            probability_bars(result['probs'], result['stage_index'])

        decision_banner(result)

        st.download_button('⬇️ Download report (.txt)', build_text_report(source_name, result, warnings),
                           file_name=f'DR_report_{source_name.rsplit(".", 1)[0]}.txt', mime='text/plain')

        # --- Preprocessing pipeline
        if show_pipeline:
            with st.expander('Preprocessing pipeline (same steps as used in training)', expanded=False):
                steps = pipeline_steps(raw)
                for row in (steps[:3], steps[3:]):
                    for c, (title, im) in zip(st.columns(3), row):
                        c.image(im, caption=title, width='stretch')
                st.caption('Crop the black border → pad to square → resize to 224×224 → bilateral denoising → '
                           'CLAHE on the L channel (LAB) → unsharp-mask edge enhancement → circular mask.')

# ---------------------------------------------------------------- batch screening
with tab_batch:
    st.markdown('Upload several fundus images to screen them in one go. '
                'Results can be exported as a CSV file for record keeping.')
    files = st.file_uploader('Upload fundus photographs', type=IMAGE_TYPES,
                             accept_multiple_files=True, key='batch')
    if files:
        classifier = get_classifier()
        rows, thumbs = [], []
        progress = st.progress(0.0, text='Analysing images...')
        for i, f in enumerate(files):
            try:
                raw = load_image(f)
                processed = preprocess_image(raw)
                r = classifier.analyse(processed, tta=use_tta, explain=False)
                rows.append({'Image': f.name, 'Stage': r['stage_index'], 'Stage name': r['stage'],
                             'Confidence': r['confidence'], 'P(any DR)': r['p_any_dr'],
                             'P(referable)': r['p_referable'], 'Refer': 'YES' if r['refer'] else 'no',
                             'Quality warnings': len(quality_check(raw))})
                thumbs.append((processed, f.name, r))
            except Exception as e:
                rows.append({'Image': f.name, 'Stage name': f'ERROR: {e}'})
            progress.progress((i + 1) / len(files), text=f'Analysed {i + 1}/{len(files)}')
        progress.empty()

        df = pd.DataFrame(rows)
        n_ref = int((df['Refer'] == 'YES').sum()) if 'Refer' in df else 0
        c1, c2, c3 = st.columns(3)
        c1.metric('Images analysed', len(df))
        c2.metric('Referrals', n_ref)
        c3.metric('Referral rate', f'{n_ref / max(len(df), 1):.0%}')

        st.dataframe(df, hide_index=True, column_config={
            'Confidence': st.column_config.ProgressColumn(format='percent', min_value=0, max_value=1),
            'P(any DR)': st.column_config.NumberColumn(format='percent'),
            'P(referable)': st.column_config.NumberColumn(format='percent'),
        })
        if 'Stage name' in df:
            st.bar_chart(df['Stage name'].value_counts().reindex(CLASS_NAMES, fill_value=0))

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button('⬇️ Download results (.csv)', buf.getvalue(),
                           file_name=f'DR_batch_{dt.datetime.now():%Y%m%d_%H%M}.csv', mime='text/csv')

        with st.expander('Preview images'):
            for start in range(0, len(thumbs), 5):
                for c, (im, name, r) in zip(st.columns(5), thumbs[start:start + 5]):
                    c.image(im, caption=f"{name}\n{r['stage']} ({r['confidence']:.0%})", width='stretch')

# ---------------------------------------------------------------- model performance
with tab_perf:
    if tm:
        st.subheader('Held-out test set (15% of APTOS 2019, never seen during training)')
        metrics = {k: v for k, v in tm.items() if k != 'Model'}
        cols = st.columns(len(metrics))
        for c, (k, v) in zip(cols, metrics.items()):
            c.metric(k, f'{v:.3f}')

        screening = RESULTS.get('binary_screening', {})
        if screening:
            st.markdown('**Binary screening performance** (thresholds tuned on the validation set)')
            st.dataframe(pd.DataFrame(screening).T.round(4))

    figures = [('11_training_curves', 'Accuracy & loss curves (phase 1 frozen backbone, phase 2 fine-tuning)'),
               ('12_confusion_matrix', 'Confusion matrix on the test set'),
               ('13_per_class_metrics', 'Per-stage precision, recall and F1-score'),
               ('14_roc_multiclass', 'One-vs-rest ROC curves'),
               ('15_binary_screening', 'Binary screening (any DR / referable DR)'),
               ('16_error_analysis', 'Error analysis'),
               ('18_gradcam_per_stage', 'Grad-CAM examples for each stage'),
               ('09_backbone_comparison', 'Backbone comparison (EfficientNetB0 vs ResNet50V2 vs MobileNetV2)')]
    for name, caption in figures:
        path = FIGURE_DIR / f'{name}.png'
        if path.exists():
            with st.expander(caption, expanded=name == '11_training_curves'):
                st.image(str(path), width='stretch')

# ---------------------------------------------------------------- about
with tab_about:
    cfg = RESULTS.get('config', {})
    st.markdown(f"""
### How the prototype works
1. **Input check** - simple heuristics flag dark, over-exposed, blurry or non-fundus images.
2. **Preprocessing** - exactly the training pipeline: border crop, square padding, resize to {IMG_SIZE}×{IMG_SIZE},
   bilateral denoising, CLAHE contrast enhancement, unsharp-mask edge enhancement and a circular mask.
3. **Classification** - {cfg.get('BACKBONE', 'EfficientNetB0')} pre-trained on ImageNet, fine-tuned in two phases
   (frozen backbone, then full fine-tuning with a low learning rate) to predict the 5 DR stages.
4. **Screening decision** - stage probabilities are summed into *P(referable DR)* = P(Moderate)+P(Severe)+P(Proliferative)
   and compared with a threshold of **{REFERRAL_THRESHOLD:.1%}** chosen on the validation set (Youden's J).
5. **Explanation** - Grad-CAM on the last convolutional layer shows which retinal regions drove the prediction.

### DR stages (International Clinical DR scale)
| Stage | Name | Typical findings |
|---|---|---|
| 0 | No DR | No abnormalities |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | Haemorrhages, hard exudates, cotton-wool spots |
| 3 | Severe NPDR | Extensive haemorrhages, venous beading, IRMA |
| 4 | Proliferative DR | Neovascularisation, vitreous haemorrhage |

### Limitations
* Trained on a single public dataset (APTOS 2019, India) - performance may drop on other cameras or populations.
* Minority stages (Mild, Severe) have far fewer training images, so they are predicted less reliably;
  most errors are confusions between neighbouring stages.
* Not validated clinically. Intended only to demonstrate decision support for screening programmes.
""")
