#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT_DIR/models"
if [ ! -d "$MODELS_DIR" ]; then
    echo "Creating models directory: $MODELS_DIR"
    mkdir -p "$MODELS_DIR"
fi
BASE_URL="https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
MODELS=(
    "tokenizer.json:$BASE_URL/tokenizer.json"
    "text_model.onnx:$BASE_URL/onnx/text_model.onnx"
    "vision_model.onnx:$BASE_URL/onnx/vision_model.onnx"
)
echo "=== Checking and downloading ONNX CLIP models ==="
echo "Target folder: $MODELS_DIR"
for ITEM in "${MODELS[@]}"; do
    NAME="${ITEM%%:*}"
    URL="${ITEM#*:}"
    TARGET_PATH="$MODELS_DIR/$NAME"
    if [ -f "$TARGET_PATH" ]; then
        echo "✓ $NAME already exists, skipping."
    else
        echo "-> Downloading $NAME..."
        curl -L --fail --retry 3 --create-dirs "$URL" -o "$TARGET_PATH"
        echo "✓ $NAME successfully downloaded."
    fi
done
echo "=== All model files are verified and ready! ==="
