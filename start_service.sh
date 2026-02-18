#!/bin/bash

# Qwen3-VL-235B-A22B-Thinking-FP8 vLLM Service Startup Script
# This script starts the vLLM 0.11.0 server for the FP8 quantized model

# Exit on any error
set -e

# Resolve the directory this script lives in (the project root), regardless of
# where it is called from or who the current user is.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Qwen3-VL-235B-A22B-Thinking-FP8 vLLM Service...${NC}"

# Load configuration
if [ -f "$SCRIPT_DIR/config.env" ]; then
    source "$SCRIPT_DIR/config.env"
    echo -e "${YELLOW}Loaded configuration from config.env${NC}"
else
    echo -e "${RED}Error: config.env not found at $SCRIPT_DIR/config.env${NC}"
    exit 1
fi

# IMPORTANT: vLLM 0.11.0 only has V1 engine (V0 removed)
# The V1 engine is automatically used - no need to set VLLM_USE_V1
# V1 engine is optimized for large models including MoE architectures
echo -e "${YELLOW}Using vLLM 0.11.0 with V1 engine (default and only option)${NC}"

# Note: PYTORCH_CUDA_ALLOC_CONF and CUDA paths are set from config.env

# Activate virtual environment
# VENV_DIR is set in config.env (defaults to $HOME/venvs/qwen3-vl-fp8)
source "$VENV_DIR/bin/activate"

echo -e "${YELLOW}Virtual environment activated${NC}"

# Verify vllm is installed in the virtual environment
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vllm not found in virtual environment at $VENV_DIR${NC}"
    echo -e "${YELLOW}Install dependencies with: pip install -r $SCRIPT_DIR/requirements.txt${NC}"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA may not be installed correctly.${NC}"
    exit 1
fi

# Check CUDA runtime version (determined by the host NVIDIA driver)
CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
MIN_CUDA_MAJOR=12
MIN_CUDA_MINOR=6

if [ "$CUDA_MAJOR" -lt "$MIN_CUDA_MAJOR" ] || \
   { [ "$CUDA_MAJOR" -eq "$MIN_CUDA_MAJOR" ] && [ "$CUDA_MINOR" -lt "$MIN_CUDA_MINOR" ]; }; then
    echo -e "${RED}Error: CUDA $CUDA_VERSION detected. This model requires CUDA >= $MIN_CUDA_MAJOR.$MIN_CUDA_MINOR.${NC}"
    echo -e "${RED}FP8 block scaling (FlashInfer CUTLASS MoE kernel) requires CUDA $MIN_CUDA_MAJOR.$MIN_CUDA_MINOR+.${NC}"
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)
    echo -e "${YELLOW}Your NVIDIA driver: $DRIVER_VERSION (supports up to CUDA $CUDA_VERSION)${NC}"
    echo -e "${YELLOW}Ask your DevOps team to upgrade the host NVIDIA driver to 560+ (for CUDA $MIN_CUDA_MAJOR.$MIN_CUDA_MINOR+).${NC}"
    exit 1
fi
echo -e "${GREEN}CUDA $CUDA_VERSION detected (>= $MIN_CUDA_MAJOR.$MIN_CUDA_MINOR required) ✓${NC}"

# Check GPU availability
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}Detected $GPU_COUNT GPU(s)${NC}"

if [ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]; then
    echo -e "${RED}Error: Requested $TENSOR_PARALLEL_SIZE GPUs but only $GPU_COUNT available${NC}"
    echo -e "${YELLOW}This model requires 8x H100 GPUs for optimal performance (236B parameters, FP8)${NC}"
    echo -e "${YELLOW}Ensure all GPUs are available and not in use by other processes.${NC}"
    echo -e "${YELLOW}Check GPU status with: nvidia-smi${NC}"
    exit 1
fi

# Display detailed GPU information
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}GPU Memory Information${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Get GPU details for the GPUs we'll use
for i in $(seq 0 $((TENSOR_PARALLEL_SIZE - 1))); do
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits -i $i)
    GPU_INDEX=$(echo $GPU_INFO | awk -F',' '{print $1}')
    GPU_NAME=$(echo $GPU_INFO | awk -F',' '{print $2}' | xargs)
    GPU_TOTAL=$(echo $GPU_INFO | awk -F',' '{print $3}' | xargs)
    GPU_USED=$(echo $GPU_INFO | awk -F',' '{print $4}' | xargs)
    GPU_FREE=$(echo $GPU_INFO | awk -F',' '{print $5}' | xargs)
    
    echo -e "GPU $GPU_INDEX: ${GPU_NAME}"
    echo -e "  Total: ${GPU_TOTAL} MiB  |  Used: ${GPU_USED} MiB  |  Free: ${GPU_FREE} MiB"
done

# Calculate total VRAM
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | head -1)
TOTAL_VRAM_ALL=$((TOTAL_VRAM * TENSOR_PARALLEL_SIZE))
# Convert GPU_MEMORY_UTIL float (e.g. 0.90) to integer percentage using awk (no bc needed)
GPU_MEM_PCT=$(awk -v v="$GPU_MEMORY_UTIL" 'BEGIN {printf "%.0f", v * 100}')
USABLE_VRAM=$((TOTAL_VRAM_ALL * GPU_MEM_PCT / 100))

echo ""
echo -e "${YELLOW}Total VRAM across $TENSOR_PARALLEL_SIZE GPUs: $TOTAL_VRAM_ALL MiB (~$((TOTAL_VRAM_ALL / 1024)) GB)${NC}"
echo -e "${YELLOW}Usable VRAM (${GPU_MEMORY_UTIL} utilization): $USABLE_VRAM MiB (~$((USABLE_VRAM / 1024)) GB)${NC}"
echo ""

# Estimate memory requirements for FP8 model
MODEL_WEIGHTS_GB=236  # FP8 is ~50% smaller than BF16
KV_CACHE_GB=$((MAX_MODEL_LEN * MAX_NUM_SEQS / 4096))  # Rough estimate
TOTAL_REQUIRED_GB=$((MODEL_WEIGHTS_GB + KV_CACHE_GB))

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Memory Requirements Estimate (FP8 Model)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Model Weights: ~${MODEL_WEIGHTS_GB} GB (FP8 quantized - 50% reduction)"
echo -e "KV Cache (est): ~${KV_CACHE_GB} GB (${MAX_MODEL_LEN} tokens × ${MAX_NUM_SEQS} seqs)"
echo -e "Total Required: ~${TOTAL_REQUIRED_GB} GB"
echo -e "Available:      ~$((USABLE_VRAM / 1024)) GB"

if [ $((USABLE_VRAM / 1024)) -lt $TOTAL_REQUIRED_GB ]; then
    echo -e "${RED}⚠️  WARNING: Available VRAM may be insufficient!${NC}"
    echo -e "${YELLOW}Consider reducing GPU_MEMORY_UTIL, MAX_MODEL_LEN, or MAX_NUM_SEQS${NC}"
else
    echo -e "${GREEN}✓ Available VRAM appears sufficient${NC}"
fi
echo ""

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port $PORT is already in use${NC}"
    echo "Please stop the existing service or choose a different port"
    exit 1
fi

# Check if model exists locally
if [ ! -d "$MODEL_PATH" ] || [ ! "$(ls -A $MODEL_PATH)" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    echo "Please ensure FP8 model weights are downloaded"
    echo "Run: ./download_model.sh"
    exit 1
fi

echo -e "${GREEN}Model found at: $MODEL_PATH${NC}"

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Service Configuration${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Model Settings:${NC}"
echo "  Path: $MODEL_PATH"
echo "  Architecture: Qwen3VLMoeForConditionalGeneration (MoE)"
echo "  Total Parameters: 236B | Active per forward: ~22B"
echo "  Quantization: FP8 (fine-grained, block size 128)"
echo "  Data Type: $DTYPE"
echo ""
echo -e "${YELLOW}Server Settings:${NC}"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  vLLM Version: 0.11.0 (V1 engine auto-detected)"
echo ""
echo -e "${YELLOW}GPU Configuration:${NC}"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE GPUs"
echo "  GPU Memory Utilization: ${GPU_MEMORY_UTIL} ($((TOTAL_VRAM * GPU_MEM_PCT / 100 / 1024)) GB per GPU)"
echo ""
echo -e "${YELLOW}Capacity Settings:${NC}"
echo "  Max Model Length: $MAX_MODEL_LEN tokens"
echo "  Max Concurrent Sequences: $MAX_NUM_SEQS"
echo "  Multimodal Limits: $LIMIT_MM_PER_PROMPT"
echo ""
echo -e "${YELLOW}Generation Defaults:${NC}"
echo "  Temperature: $TEMPERATURE | Top-P: $TOP_P | Top-K: $TOP_K"
echo "  Max Tokens: $MAX_TOKENS"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Starting vLLM Server...${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}📍 Service URL: http://localhost:$PORT${NC}"
echo -e "${YELLOW}📄 Logs: $SCRIPT_DIR/logs/service.log${NC}"
echo -e "${YELLOW}⏱️  Model loading typically takes 5-10 minutes...${NC}"
echo ""

# Start vLLM server with logging (output to both console and log file)
vllm serve "$MODEL_PATH" \
  --port "$PORT" \
  --host "$HOST" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --enable-expert-parallel \
  --trust-remote-code \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --uvicorn-log-level info \
  2>&1 | tee "$SCRIPT_DIR/logs/service.log"
















