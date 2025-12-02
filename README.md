# Qwen3-VL-235B-A22B-Thinking-FP8 Service

Inference service for the **FP8 quantized** Qwen3-VL-235B-A22B-Thinking vision-language model with reasoning capabilities.

## Model Overview

**Qwen3-VL-235B-A22B-Thinking-FP8** is the FP8 quantized version of the state-of-the-art vision-language model:

- **Architecture**: Mixture-of-Experts (MoE) with 236B total parameters, ~22B active per forward pass
- **Quantization**: Fine-grained FP8 with block size 128
- **Performance**: Nearly identical to BF16 version (~98% retention)
- **Memory Advantage**: ~50% reduction (236GB vs 472GB)
- **Context Length**: 256K tokens native, expandable to 1M
- **Capabilities**:
  - Advanced image and video understanding
  - Visual reasoning and analysis
  - OCR in 32 languages
  - Spatial perception and 3D grounding
  - Visual coding (HTML/CSS/JS generation)
  - GUI understanding for agent tasks
  - Long-form video analysis with temporal modeling

## Why FP8?

The FP8 quantized version offers significant advantages:

| Aspect | BF16 Version | FP8 Version |
|--------|-------------|-------------|
| Model Size | ~472 GB | ~236 GB |
| Min GPUs (tight) | 8x H100 | 4x H100 |
| Recommended GPUs | 8x H100 | 8x H100 |
| Inference Speed | Baseline | **~1.3-1.5x faster** |
| Memory Bandwidth | Baseline | **~50% reduction** |
| Quality | 100% | **~98%** |

## Hardware Requirements

- **GPUs**: 8x NVIDIA H100 80GB (640GB total VRAM) - recommended
  - *Minimum: 4x H100 80GB (320GB) with reduced settings*
- **Memory**: 320GB system RAM (recommended)
- **Storage**: ~250GB for model weights
- **CUDA**: Version 12.9 or compatible

## Quick Start

### 1. Download the Model

```bash
cd /home/naresh/qwen3-vl-fp8-service
./download_model.sh
```

This will download ~240GB of FP8 quantized model weights from Hugging Face.

### 2. Allocate SLURM Resources

```bash
salloc -p p_naresh --job-name=qwen-vl-fp8 --gres=gpu:8 --mem=320G --cpus-per-task=32 --time=24:00:00
```

### 3. Start the Service

```bash
cd /home/naresh/qwen3-vl-fp8-service
./start_service.sh
```

The service will:
- Start on port **8010**
- Check/start HTTP server on port **8080** for media files (if not running)
- Take **5-10 minutes** to load the model across 8 GPUs
- Show detailed GPU distribution and memory usage
- Be ready to accept requests via OpenAI-compatible API

### 4. Test the Service

Once the service shows "Application startup complete", test it:

```bash
curl http://localhost:8010/v1/models
```

### 5. Stop the Service

```bash
# Press Ctrl+C in the service terminal
# OR use the stop script
./stop_service.sh
```

## Configuration

All configuration is in `config.env`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PORT` | 8010 | API server port |
| `TENSOR_PARALLEL_SIZE` | 8 | Number of GPUs (recommended: 8) |
| `MAX_MODEL_LEN` | 32768 | Maximum context length |
| `GPU_MEMORY_UTIL` | 0.90 | GPU memory utilization (90%) |
| `MAX_NUM_SEQS` | 4 | Concurrent requests |
| `DTYPE` | float16 | Data type (FP8 loaded as float16) |
| `TEMPERATURE` | 0.7 | Sampling temperature |
| `MAX_TOKENS` | 8192 | Max generation length |

Edit `config.env` to customize, then restart the service.

## API Usage

The service provides an OpenAI-compatible API at `http://localhost:8010/v1`.

### Example: Image Understanding

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "qwen3-vl-fp8",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in detail."
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Example: Image URL

```python
import requests

response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "qwen3-vl-fp8",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    }
                ]
            }
        ],
        "max_tokens": 256
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Example: Video Analysis

The service automatically starts an HTTP server on port 8080 for local media files:

```python
import requests

response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "qwen3-vl-fp8",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "http://localhost:8080/videos/sample_video.mp4"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Summarize what happens in this video."
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Example: Text-Only (No Vision)

```python
import requests

response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "qwen3-vl-fp8",
        "messages": [
            {
                "role": "user",
                "content": "Explain the concept of mixture of experts."
            }
        ],
        "max_tokens": 512
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Example: Multi-Turn Conversation

```python
import requests

response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "qwen3-vl-fp8",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/chart.png"}
                    },
                    {"type": "text", "text": "What data is shown in this chart?"}
                ]
            },
            {
                "role": "assistant",
                "content": "This chart shows sales data over Q1-Q4..."
            },
            {
                "role": "user",
                "content": "What was the trend in Q3?"
            }
        ]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## Performance Tuning

### For Higher Throughput

After initial testing, increase concurrency:

```bash
# In config.env
GPU_MEMORY_UTIL=0.95  # Use 95% of GPU memory
MAX_NUM_SEQS=8        # Handle 8 concurrent requests
```

### For Longer Context

To process longer videos or documents:

```bash
# In config.env
MAX_MODEL_LEN=65536   # Support up to 65K tokens
MAX_NUM_SEQS=2        # Reduce concurrency for memory
```

### For Faster Responses (Lower Quality)

```bash
# In config.env
TEMPERATURE=0.3       # More deterministic
MAX_TOKENS=512        # Shorter responses
```

### For Creative Tasks

```bash
# In config.env
TEMPERATURE=1.0       # More creative
TOP_P=0.95
TOP_K=50
```

## Monitoring

### Check Service Status

```bash
# Check if service is running
lsof -i :8010

# View logs
tail -f /home/naresh/qwen3-vl-fp8-service/logs/service.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### GPU Memory Usage

Expected memory usage per GPU with FP8:
- **Model weights**: ~30GB per GPU (236GB total / 8)
- **KV cache + activations**: ~10-15GB per GPU (varies with batch size)
- **Total**: ~40-45GB per GPU at 90% utilization

*Note: This is significantly lower than the BF16 version (~70-75GB per GPU)*

## Troubleshooting

### Service Won't Start

**Issue**: `Error: Requested 8 GPUs but only X available`

**Solution**: Ensure you're in a SLURM allocation with 8 GPUs:
```bash
salloc -p p_naresh --job-name=qwen-vl-fp8 --gres=gpu:8 --mem=320G --cpus-per-task=32 --time=24:00:00
```

### Out of Memory Errors

**Issue**: `CUDA out of memory`

**Solutions**:
1. Reduce `GPU_MEMORY_UTIL` to 0.85
2. Reduce `MAX_NUM_SEQS` to 2
3. Reduce `MAX_MODEL_LEN` to 16384

*Note: FP8 version should have fewer OOM issues than BF16*

### Slow Inference

**Issue**: Requests taking too long

**Causes**:
- Very long input context (large images, long videos)
- High `MAX_TOKENS` setting
- Multiple concurrent requests

**Solutions**:
1. Use smaller images (resize before encoding)
2. Reduce `MAX_TOKENS` for faster generation
3. Lower `MAX_NUM_SEQS` to reduce GPU contention

### Port Already in Use

**Issue**: `Error: Port 8010 is already in use`

**Solution**:
```bash
# Find and stop the existing process
lsof -ti:8010 | xargs kill -9

# Or use the stop script
./stop_service.sh
```

### Model Not Found

**Issue**: `Error: Model not found at /home/naresh/models/qwen3-vl-235b-thinking-fp8`

**Solution**: Download the model:
```bash
./download_model.sh
```

## File Structure

```
qwen3-vl-fp8-service/
├── config.env              # Configuration file
├── start_service.sh        # Service startup script
├── stop_service.sh         # Service shutdown script
├── download_model.sh       # Model download script
├── README.md               # This file
└── logs/
    └── service.log         # Service logs
```

## API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion (recommended)
- `POST /v1/completions` - Text completion
- `GET /health` - Health check

## Advanced Features

### Visual Coding

Generate HTML/CSS/JS from wireframe images:

```python
response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                {"type": "text", "text": "Generate HTML/CSS code for this design."}
            ]
        }],
        "max_tokens": 2048
    }
)
```

### Spatial Understanding

Analyze object positions and relationships:

```python
response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                {"type": "text", "text": "Describe the spatial relationships between objects in this image."}
            ]
        }]
    }
)
```

### OCR (32 Languages)

Extract text from images:

```python
response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                {"type": "text", "text": "Extract all text from this image."}
            ]
        }],
        "temperature": 0.2  # Lower temperature for accuracy
    }
)
```

## Comparison with BF16 Version

If you're running both versions, here's how they compare:

| Feature | BF16 (port 7001) | FP8 (port 8010) |
|---------|------------------|-----------------|
| Model Size | 472 GB | 236 GB |
| Memory/GPU | ~60 GB | ~30 GB |
| Speed | Baseline | 1.3-1.5x faster |
| Quality | 100% | ~98% |
| Throughput | Good | Better |

**Recommendation**: Use FP8 version for production workloads. The minimal quality loss is offset by better speed and efficiency.

## Technical Details

### FP8 Quantization

- **Method**: Fine-grained FP8 quantization
- **Block Size**: 128
- **Quantization Scheme**: Per-block scaling factors
- **Activation**: FP16 (weights are FP8, computations in FP16)

### Performance Characteristics

- **Inference Latency**: ~30% faster than BF16
- **Memory Bandwidth**: ~50% reduction
- **Throughput**: Higher due to reduced memory pressure
- **Quality**: 98-99% retention on benchmarks

## References

- **Model**: [Qwen/Qwen3-VL-235B-A22B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8)
- **Original Model**: [Qwen/Qwen3-VL-235B-A22B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)
- **vLLM**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Qwen Technical Report**: [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

## Support

For issues or questions:
1. Check the logs: `tail -f logs/service.log`
2. Review this README
3. Check GPU status: `nvidia-smi`
4. Verify SLURM allocation: `squeue`

## License

This service uses the Qwen3-VL-235B-A22B-Thinking-FP8 model, which is licensed under Apache 2.0.












