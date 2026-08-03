# Input images

Drop Instagram screenshots / poster images here (`.jpg .jpeg .png .bmp .webp .tif`).
Subfolders are searched too.

    python run_ocr.py data/input --lang te+en --formats txt json csv

Results land in `outputs/`, one file per image per format, plus
`outputs/_batch_summary.json` when more than one image is processed.
