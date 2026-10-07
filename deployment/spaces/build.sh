#!/bin/bash
# Assemble the Hugging Face Space contents in deployment/spaces/build.
# Then sync it into your Space checkout, e.g.:
#   rsync -a --delete --exclude .git --exclude .gitattributes deployment/spaces/build/ hf-space/
set -e
cd "$(dirname "$0")/../.."
OUT=deployment/spaces/build
rm -rf "$OUT"
mkdir -p "$OUT/backend/app" "$OUT/ml/models/weights" "$OUT/ml/tennis_analysis" "$OUT/ml/train"
cp deployment/spaces/{app.py,download_models.py,Dockerfile,requirements.txt,README.md} "$OUT/"
cp backend/__init__.py "$OUT/backend/"
cp backend/app/{__init__,main,analytics,shots}.py "$OUT/backend/app/"
cp ml/__init__.py "$OUT/ml/"
cp ml/models/__init__.py ml/models/shot_classifier.py "$OUT/ml/models/"
cp ml/tennis_analysis/{tracknet,bounce_detector}.py "$OUT/ml/tennis_analysis/"
# Raquette's own models (small; .pt files go through LFS per the Space's .gitattributes)
cp ml/models/weights/{bounce_classifier.cbm,serve_detector.pt,rally_classifier.pt} "$OUT/ml/models/weights/"
cp ml/train/pose_landmarker_lite.task "$OUT/ml/train/"
cp yolov8n.pt "$OUT/"
printf 'uploads/\nbackend/uploads/\n__pycache__/\n*.py[cod]\n.DS_Store\n' > "$OUT/.gitignore"
echo "Space files ready in $OUT"
