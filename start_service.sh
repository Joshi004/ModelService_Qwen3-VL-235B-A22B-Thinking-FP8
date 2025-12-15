#!/bin/bash

# Qwen3-VL-235B-A22B-Thinking-FP8 vLLM Service Startup Script
# This script starts the vLLM 0.11.0 server for the FP8 quantized model

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Qwen3-VL-235B-A22B-Thinking-FP8 vLLM Service...${NC}"

# Load configuration
if [ -f "/home/naresh/qwen3-vl-fp8-service/config.env" ]; then
    source /home/naresh/qwen3-vl-fp8-service/config.env
    echo -e "${YELLOW}Loaded configuration from config.env${NC}"
else
    echo -e "${RED}Error: config.env not found${NC}"
    exit 1
fi

# IMPORTANT: vLLM 0.11.0 only has V1 engine (V0 removed)
# The V1 engine is automatically used - no need to set VLLM_USE_V1
# V1 engine is optimized for large models including MoE architectures
echo -e "${YELLOW}Using vLLM 0.11.0 with V1 engine (default and only option)${NC}"

# Note: PYTORCH_CUDA_ALLOC_CONF and CUDA paths are set from config.env

# Media server configuration - serves both videos/ and audios/ subdirectories
MEDIA_DIR="/home/naresh/datasets"
MEDIA_PORT=8080

# Check if port 8080 is already in use
if lsof -Pi :$MEDIA_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}HTTP server already running on port $MEDIA_PORT - reusing existing server${NC}"
    echo "  Media files accessible at: http://localhost:$MEDIA_PORT/"
else
    echo -e "${GREEN}Starting HTTP server for media files...${NC}"
    echo "  Directory: $MEDIA_DIR"
    echo "  Port: $MEDIA_PORT"
    echo "  Serving: videos/, audios/, and other subdirectories"
    
    # Check if media directory exists
    if [ ! -d "$MEDIA_DIR" ]; then
        echo -e "${YELLOW}Warning: Directory $MEDIA_DIR does not exist. Creating it...${NC}"
        mkdir -p "$MEDIA_DIR"
    fi
    
    # Start Python HTTP server in background with nohup for proper detachment
    cd "$MEDIA_DIR"
    nohup python3 -m http.server $MEDIA_PORT > /tmp/media_server.log 2>&1 &
    MEDIA_SERVER_PID=$!
    echo $MEDIA_SERVER_PID > /tmp/media_server.pid
    echo -e "${GREEN}HTTP server started with PID: $MEDIA_SERVER_PID${NC}"
    echo "  Video URLs: http://localhost:$MEDIA_PORT/videos/<filename>"
    echo "  Audio URLs: http://localhost:$MEDIA_PORT/audios/<filename>"
    
    # Return to service directory
    cd /home/naresh/qwen3-vl-fp8-service
    
    # Give the HTTP server a moment to start and verify it's running
    sleep 3
    
    # Verify the server is actually running
    if lsof -Pi :$MEDIA_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${GREEN}HTTP server verified running on port $MEDIA_PORT${NC}"
    else
        echo -e "${YELLOW}Warning: HTTP server may not have started properly. Check /tmp/media_server.log${NC}"
        if [ -f /tmp/media_server.log ]; then
            echo "  Last few lines of server log:"
            tail -5 /tmp/media_server.log | sed 's/^/    /'
        fi
    fi
fi

echo ""

# Activate virtual environment
source /home/naresh/venvs/qwen3-vl-fp8-service/bin/activate

echo -e "${YELLOW}Virtual environment activated${NC}"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA may not be installed correctly.${NC}"
    exit 1
fi

# Check GPU availability
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}Detected $GPU_COUNT GPU(s)${NC}"

if [ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]; then
    echo -e "${RED}Error: Requested $TENSOR_PARALLEL_SIZE GPUs but only $GPU_COUNT available${NC}"
    echo -e "${YELLOW}This model requires 8x H100 GPUs for optimal performance (236B parameters, FP8)${NC}"
    echo -e "${YELLOW}Allocate GPUs using: salloc -p p_naresh --job-name=qwen-vl-fp8 --gres=gpu:8 --mem=320G --cpus-per-task=32 --time=24:00:00${NC}"
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
USABLE_VRAM=$(echo "scale=0; $TOTAL_VRAM_ALL * $GPU_MEMORY_UTIL / 1" | bc)

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
mkdir -p /home/naresh/qwen3-vl-fp8-service/logs

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
echo "  GPU Memory Utilization: ${GPU_MEMORY_UTIL} ($(echo "scale=0; $TOTAL_VRAM * $GPU_MEMORY_UTIL / 1024 / 1" | bc) GB per GPU)"
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
echo -e "${YELLOW}📄 Logs: /home/naresh/qwen3-vl-fp8-service/logs/service.log${NC}"
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
  2>&1 | tee /home/naresh/qwen3-vl-fp8-service/logs/service.log
















