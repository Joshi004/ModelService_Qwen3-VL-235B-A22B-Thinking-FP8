#!/bin/bash

# Qwen3-VL-235B-A22B-Thinking-FP8 Service Stop Script
# This script gracefully stops the vLLM server

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Qwen3-VL-235B-A22B-Thinking-FP8 Service...${NC}"

PORT=8010

# Find the vLLM process
VLLM_PID=$(lsof -ti:$PORT)

if [ -z "$VLLM_PID" ]; then
    echo -e "${YELLOW}No service found running on port $PORT${NC}"
    exit 0
fi

echo -e "${GREEN}Found vLLM server running on port $PORT (PID: $VLLM_PID)${NC}"

# Try graceful shutdown first (SIGTERM)
echo -e "${YELLOW}Attempting graceful shutdown...${NC}"
kill -TERM $VLLM_PID

# Wait for process to exit (max 30 seconds)
for i in {1..30}; do
    if ! ps -p $VLLM_PID > /dev/null 2>&1; then
        echo -e "${GREEN}Service stopped successfully${NC}"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo -e "${YELLOW}Process did not exit gracefully. Forcing shutdown...${NC}"

# Force kill if still running
if ps -p $VLLM_PID > /dev/null 2>&1; then
    kill -9 $VLLM_PID
    sleep 2
    
    if ps -p $VLLM_PID > /dev/null 2>&1; then
        echo -e "${RED}Failed to stop service (PID: $VLLM_PID)${NC}"
        exit 1
    else
        echo -e "${GREEN}Service force-stopped${NC}"
    fi
fi

# Clean up any orphaned vllm processes for this specific service
echo -e "${YELLOW}Checking for orphaned vLLM processes...${NC}"
pkill -f "vllm serve.*qwen3-vl-235b-thinking-fp8" && echo -e "${GREEN}Cleaned up orphaned processes${NC}" || echo -e "${GREEN}No orphaned processes found${NC}"

echo -e "${GREEN}Shutdown complete${NC}"
echo ""
echo -e "${YELLOW}Note: HTTP server on port 8080 is shared and was not stopped.${NC}"
echo -e "${YELLOW}To stop it manually: lsof -ti:8080 | xargs kill -9${NC}"




