"""
Command-line inference for one or more fundus images (no user interface needed).

Examples:
    python predict.py path/to/image.png
    python predict.py sample_images/*.png --tta --csv results.csv
"""
import argparse
import glob

import pandas as pd

from dr_app.preprocessing import load_image, preprocess_image, quality_check


def main():
    parser = argparse.ArgumentParser(description='Predict the diabetic retinopathy stage of fundus images.')
    parser.add_argument('images', nargs='+', help='Image file(s) or glob pattern(s)')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation (flip averaging)')
    parser.add_argument('--csv', help='Optional path to save the results as CSV')
    args = parser.parse_args()

    # Expand glob patterns (Windows shells do not do this automatically)
    paths = [p for pattern in args.images for p in (glob.glob(pattern) or [pattern])]

    from dr_app.inference import DRClassifier   # heavy import - after argument parsing
    classifier = DRClassifier()

    rows = []
    for path in paths:
        raw = load_image(path)
        r = classifier.analyse(preprocess_image(raw), tta=args.tta, explain=False)
        rows.append({'image': path, 'stage': r['stage_index'], 'stage_name': r['stage'],
                     'confidence': round(r['confidence'], 4), 'p_referable': round(r['p_referable'], 4),
                     'refer': r['refer'], 'quality_warnings': '; '.join(quality_check(raw))})
        print(f"{path}: stage {r['stage_index']} ({r['stage']}, {r['confidence']:.1%}) | "
              f"P(referable)={r['p_referable']:.1%} -> {'REFER' if r['refer'] else 'routine'}")

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print('Saved', args.csv)


if __name__ == '__main__':
    main()
