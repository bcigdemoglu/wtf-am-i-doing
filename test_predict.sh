#!/bin/bash
# Test script for FastVLM prediction

set -e

FASTVLM_DIR="$HOME/.wtf-am-i-doing/ml-fastvlm"
MODEL_PATH="$FASTVLM_DIR/checkpoints/llava-fastvithd_0.5b_stage3"
TEST_IMAGE="/tmp/test_screenshot.png"

echo "=== FastVLM Prediction Test ==="
echo

# Use existing screenshot
echo "Using test image: $TEST_IMAGE"
if [ ! -f "$TEST_IMAGE" ]; then
    echo "ERROR: Test image not found. Please place test_screenshot.png on Desktop first."
    exit 1
fi
echo

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi
echo "Model found: $MODEL_PATH"
echo

# Check if conda env exists
if ! conda env list | grep -q "fastvlm"; then
    echo "ERROR: fastvlm conda environment not found"
    exit 1
fi
echo "Conda environment 'fastvlm' found"
echo

# Run prediction
echo "Running prediction..."
echo "Command: conda run -n fastvlm python predict.py --model-path $MODEL_PATH --image-file $TEST_IMAGE --prompt 'Describe what is on the screen.'"
echo

cd "$FASTVLM_DIR"
conda run -n fastvlm python predict.py \
    --model-path "$MODEL_PATH" \
    --image-file "$TEST_IMAGE" \
    --prompt "Describe what is on the screen."

echo
echo "=== Test Complete ==="
