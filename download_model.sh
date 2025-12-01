#!/bin/bash

# Qwen3-VL-235B-A22B-Thinking-FP8 Model Download Script
# This script downloads the FP8 quantized model from Hugging Face

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Qwen3-VL-235B-A22B-Thinking-FP8 Model Download${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Model information
MODEL_REPO="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
LOCAL_DIR="/home/naresh/models/qwen3-vl-235b-thinking-fp8"

echo -e "${YELLOW}Model Information:${NC}"
echo "  Repository: $MODEL_REPO"
echo "  Local Path: $LOCAL_DIR"
echo "  Size: ~240 GB (FP8 quantized)"
echo "  Files: ~48 safetensor shards (half the size of BF16 version)"
echo ""

# Check if model already exists
if [ -d "$LOCAL_DIR" ] && [ "$(ls -A $LOCAL_DIR)" ]; then
    echo -e "${YELLOW}⚠️  Warning: Model directory already exists and is not empty${NC}"
    echo "  Path: $LOCAL_DIR"
    echo ""
    read -p "Do you want to continue? This will update/overwrite existing files (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Download cancelled${NC}"
        exit 0
    fi
fi

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}Error: huggingface-cli not found${NC}"
    echo "Please install it with: pip install huggingface-hub"
    exit 1
fi

# Check disk space
echo -e "${YELLOW}Checking disk space...${NC}"
MODELS_DIR="/home/naresh/models"
AVAILABLE_SPACE=$(df -BG "$MODELS_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED_SPACE=250  # ~240 GB + some buffer

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo -e "${RED}Error: Insufficient disk space${NC}"
    echo "  Available: ${AVAILABLE_SPACE} GB"
    echo "  Required: ${REQUIRED_SPACE} GB"
    exit 1
else
    echo -e "${GREEN}✓ Sufficient disk space available (${AVAILABLE_SPACE} GB free)${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Starting Download...${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Note: This will take a while (1-4 hours depending on connection speed)${NC}"
echo -e "${YELLOW}Progress will be shown below...${NC}"
echo ""

# Create directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Download the model
# Using huggingface-cli with resume support
huggingface-cli download "$MODEL_REPO" \
  --local-dir "$LOCAL_DIR" \
  --resume-download

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Verify download
echo -e "${YELLOW}Verifying download...${NC}"

# Check for essential files
REQUIRED_FILES=("config.json" "tokenizer.json" "preprocessor_config.json")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$LOCAL_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

# Check for model weight files (safetensors)
SAFETENSOR_COUNT=$(ls -1 "$LOCAL_DIR"/model-*.safetensors 2>/dev/null | wc -l)

if [ "$SAFETENSOR_COUNT" -eq 0 ]; then
    MISSING_FILES+=("model safetensor files")
fi

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All essential files present${NC}"
    echo "  Config files: OK"
    echo "  Model weights: $SAFETENSOR_COUNT safetensor shards found"
    echo ""
    echo -e "${GREEN}Model ready to use!${NC}"
    echo "  Path: $LOCAL_DIR"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Start the service: ./start_service.sh"
    echo "  2. Wait for model loading (5-10 minutes)"
    echo "  3. Test the service with the API examples in README.md"
else
    echo -e "${RED}⚠️  Warning: Some files may be missing:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo -e "${YELLOW}The download may have been incomplete. Try running this script again.${NC}"
fi

# Show disk usage
echo ""
echo -e "${YELLOW}Disk usage:${NC}"
du -sh "$LOCAL_DIR" 2>/dev/null || echo "  Unable to calculate size"
echo ""




