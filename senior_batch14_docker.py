"""
SENIOR AI ENGINEER INTERVIEW PREP - BATCH 14: DOCKER FOR ML/AI
==============================================================

Topic: Docker & Containerization for ML/AI Systems
Target Level: Senior AI Engineer / AI Solutions Architect (5+ years)
Question Count: 20 Questions
Difficulty: Hard

Coverage Areas:
- GPU passthrough & nvidia-docker/nvidia-container-toolkit
- Multi-stage builds for ML images
- Layer caching strategies for dependencies
- Volume mounts for model checkpoints & data
- Docker Compose for ML stacks
- Container resource limits (GPU, memory, CPU)
- Image optimization (size reduction)
- BuildKit optimizations
- Security best practices
- Registry management for large models
"""

def create_question(question_text, options, correct_index, explanation, difficulty, time_seconds):
    """Helper function to create a question dictionary"""
    return {
        "question": question_text,
        "options": options,
        "correct_answer": correct_index,
        "explanation": explanation,
        "difficulty": difficulty,
        "estimated_time": time_seconds
    }

def populate_senior_docker():
    """20 Senior-Level Docker for ML/AI Questions"""

    questions = [

        # Q1: GPU Passthrough & nvidia-docker
        create_question(
            "Q1: You're deploying a PyTorch training container that needs 4x A100 GPUs (40GB each). The container fails with 'nvidia-smi: command not found'. Your Dockerfile has 'FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime'. What's the MOST LIKELY issue and fix?",
            [
                "Docker doesn't support GPU - migrate to Kubernetes with GPU operator",
                "Missing --gpus all flag in docker run AND need nvidia-container-toolkit installed on host. Runtime base image is correct but requires host-level NVIDIA driver + toolkit + daemon restart",
                "Change base to 'nvidia/cuda:11.8-devel' - runtime images don't support GPUs",
                "Add 'RUN apt-get install -y nvidia-driver-525' to Dockerfile"
            ],
            1,
            """Senior Explanation: GPU passthrough requires TWO components:

**Host Requirements (Most Common Issue):**
```bash
# 1. Install NVIDIA Container Toolkit on host
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 2. Restart Docker daemon to register nvidia runtime
sudo systemctl restart docker

# 3. Run with --gpus flag
docker run --gpus all pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime nvidia-smi
```

**Why Options Are Wrong:**
- **Option 0 (Kubernetes)**: Docker DOES support GPUs via nvidia-container-toolkit. Kubernetes is overkill.
- **Option 2 (devel vs runtime)**: BOTH work. Runtime (~2GB) has CUDA runtime libs. Devel (~4GB) adds nvcc compiler. For inference/training of existing models, runtime is sufficient and smaller.
- **Option 3 (Install driver in container)**: WRONG. NVIDIA driver MUST be on host. Container only needs CUDA libs (already in pytorch base). Driver version must be compatible: driver ≥ CUDA toolkit version.

**Multi-GPU Example:**
```bash
# Specific GPUs
docker run --gpus '"device=0,2"' my-training-image python train.py

# All GPUs with memory limit
docker run --gpus all --shm-size=32g my-image
```

**Verification:**
```bash
# Inside container
nvidia-smi  # Should show all GPUs
python -c "import torch; print(torch.cuda.device_count())"  # Should match GPU count
```

**Production Note:** Runtime image is 50% smaller (2.1GB vs 4.3GB for devel). Only use devel if you need to compile CUDA kernels (e.g., custom Flash Attention kernels).""",
            "Hard",
            220
        ),

        # Q2: Multi-Stage Builds for ML Images
        create_question(
            "Q2: Your ML inference image is 12GB (base PyTorch + transformers). Build takes 45min because pip installs run every time. Which multi-stage Dockerfile pattern achieves SMALLEST final image + FASTEST rebuild when only app code changes?",
            [
                "Single stage with pip cache mount - fast but large final image",
                "Stage 1: Install all deps. Stage 2: COPY --from=0 /usr/local/lib + app code. Use BuildKit cache mounts for pip in Stage 1. Final image: ~4-5GB (only runtime deps), rebuild <2min (cache hit on deps)",
                "Three stages: builder (deps) → tester (run tests) → production (copy all)",
                "Use Docker volumes to cache pip downloads"
            ],
            1,
            """Senior Explanation: Multi-stage builds are CRITICAL for ML images to separate build-time dependencies from runtime.

**Optimal Pattern:**
```dockerfile
# syntax=docker/dockerfile:1.4  # Enable BuildKit features

# Stage 1: Dependency Builder (cached, rarely changes)
FROM python:3.10-slim as builder
WORKDIR /install

# Copy only dependency files first (layer caching)
COPY requirements.txt .

# BuildKit cache mount for pip (persists across builds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install --no-warn-script-location \
    -r requirements.txt

# Stage 2: Production Image (rebuilt when code changes)
FROM python:3.10-slim
WORKDIR /app

# Copy installed packages from builder (no pip, no cache)
COPY --from=builder /install /usr/local

# Copy application code (changes frequently)
COPY . .

CMD ["python", "inference.py"]
```

**Size Comparison:**
- Single stage with all: **12GB** (includes pip cache, build tools, wheel files)
- Multi-stage: **4.2GB** (only runtime packages + app code)
- Savings: **65% reduction**

**Build Time Comparison:**
```bash
# Initial build (cold cache)
# Single stage: 45min (pip install from PyPI)
# Multi-stage: 45min (same)

# Rebuild after code change (warm cache)
# Single stage: 45min (pip cache helps but still installs)
# Multi-stage with BuildKit: 90 seconds (only copies + app layer)
```

**Advanced Optimization for Large Models:**
```dockerfile
# Stage 1: Download models separately
FROM python:3.10-slim as model-downloader
RUN --mount=type=cache,target=/root/.cache/huggingface \
    pip install transformers && \
    python -c "from transformers import AutoModel; \
               AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')"

# Stage 2: Runtime
FROM python:3.10-slim
COPY --from=model-downloader /root/.cache/huggingface /models
ENV TRANSFORMERS_CACHE=/models
```

**Why Options Are Wrong:**
- **Option 0**: Fast rebuild but 12GB image. Wastes bandwidth + storage in production.
- **Option 2**: Testing in build slows CI. Better to test separately (Docker != CI).
- **Option 3**: Volumes don't reduce final image size, only build cache.

**Production Impact:**
- 4GB image pulls in 40sec (1Gb/s network) vs 12GB in 2min
- 65% lower ECR/registry storage costs
- Faster pod startup in Kubernetes (image pull is often bottleneck)""",
            "Hard",
            240
        ),

        # Q3: Layer Caching for Dependencies
        create_question(
            "Q3: You have a Dockerfile with 'COPY . .' followed by 'RUN pip install -r requirements.txt'. Every code change triggers full 40-minute pip install. What's the BEST fix for layer caching?",
            [
                "Use pip install --cache-dir to cache packages",
                "Reorder: 'COPY requirements.txt .' → 'RUN pip install' → 'COPY . .' so requirements layer is cached until requirements.txt changes",
                "Use .dockerignore to exclude code changes",
                "Run pip install before COPY in a separate container"
            ],
            1,
            """Senior Explanation: Docker layer caching is based on **file content hash**. When ANY file in COPY changes, that layer + all subsequent layers are invalidated.

**Problem Pattern (WRONG):**
```dockerfile
FROM python:3.10-slim
WORKDIR /app

# ❌ BAD: Copies ALL code (100+ files)
COPY . .

# This layer invalidated EVERY time any code file changes
RUN pip install -r requirements.txt  # 40 minutes wasted
```

**Optimal Pattern (CORRECT):**
```dockerfile
FROM python:3.10-slim
WORKDIR /app

# ✅ GOOD: Copy ONLY dependency files first
COPY requirements.txt .

# This layer cached until requirements.txt changes (rare)
RUN pip install -r requirements.txt  # 40 min, but cached

# Copy code last (changes frequently)
COPY . .

# Total rebuild time: ~30 seconds (only re-copy code)
```

**Layer Cache Hit Rate Analysis:**
```
Scenario: 100 builds over 1 week
- requirements.txt changes: 5 times (5%)
- Code changes: 95 times (95%)

BAD pattern:
- Cache misses: 100 (pip runs 100 times)
- Total pip time: 100 × 40min = 66 hours

GOOD pattern:
- Cache misses: 5 (pip runs 5 times)
- Total pip time: 5 × 40min = 3.3 hours
- Savings: 95% (63 hours)
```

**Advanced: Multiple Dependency Files:**
```dockerfile
# Copy in order of change frequency (least → most)
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt  # Cached almost always

COPY requirements-ml.txt .
RUN pip install -r requirements-ml.txt  # Cached often

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt  # Cached sometimes

COPY . .  # Changes most frequently
```

**Why Options Are Wrong:**
- **Option 0**: --cache-dir helps with repeated package downloads but doesn't fix layer invalidation
- **Option 2**: .dockerignore excludes files from COPY but doesn't solve ordering
- **Option 3**: Impossible - COPY must happen before RUN can access files

**BuildKit Enhancement:**
```dockerfile
# Even better: cache mount (cache persists across builds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

**Production Validation:**
```bash
# Check layer cache
docker build . 2>&1 | grep "Using cache"

# Expect:
# Step 3/6 : RUN pip install -r requirements.txt
#  ---> Using cache
```
""",
            "Hard",
            210
        ),

        # Q4: Volume Mounts for Model Checkpoints
        create_question(
            "Q4: You're training a 70B model with checkpoints saved every 1000 steps (each checkpoint ~140GB). Training runs in Docker container. After 10 checkpoints (1.4TB), container crashes and ALL checkpoints are lost. What's the proper volume mount strategy?",
            [
                "Use named volumes: 'docker run -v model-checkpoints:/checkpoints' - data persists but requires manual backup, limited to single host",
                "Use bind mount to host: 'docker run -v /mnt/nfs-storage:/checkpoints' so checkpoints saved to shared NFS. Survives container crashes, accessible from multiple hosts, supports network storage",
                "Increase container disk size with --storage-opt",
                "Use COPY to extract checkpoints after training"
            ],
            1,
            """Senior Explanation: Container filesystems are EPHEMERAL - data lost on crash/removal. For large model checkpoints, use **bind mounts to persistent storage**.

**Volume Types Comparison:**

**1. Container Layer (Default - WRONG for checkpoints):**
```bash
docker run pytorch-training python train.py  # No volume
# Checkpoints written to container layer
# Container crash → DATA LOST
# Max size: ~100GB (overlay2 limit)
```

**2. Named Volumes (OK for small datasets):**
```bash
docker volume create model-checkpoints
docker run -v model-checkpoints:/checkpoints pytorch-training

# Pros: Data persists on host at /var/lib/docker/volumes/
# Cons:
#   - Single host only (not shared across nodes)
#   - Manual backup required
#   - Harder to inspect (buried in Docker internals)
```

**3. Bind Mounts (BEST for large ML checkpoints):**
```bash
docker run \
  -v /mnt/nfs-storage/checkpoints:/checkpoints \
  -v /mnt/nfs-storage/tensorboard:/tensorboard \
  --gpus all \
  --shm-size=32g \
  pytorch-training python train.py \
    --checkpoint_dir=/checkpoints \
    --save_steps=1000

# Pros:
#   ✅ Data on host filesystem (easy to inspect)
#   ✅ Supports network storage (NFS, EFS, Lustre)
#   ✅ Shared across multiple training nodes
#   ✅ Survives container crashes/deletions
#   ✅ No size limit (limited by mount capacity)
```

**Production Setup for Distributed Training:**
```yaml
# docker-compose.yml for multi-node training
version: '3.8'
services:
  trainer-node0:
    image: pytorch-training:latest
    volumes:
      - /mnt/shared-nfs/checkpoints:/checkpoints:rw
      - /mnt/shared-nfs/data:/data:ro  # Read-only data
    environment:
      - MASTER_ADDR=trainer-node0
      - RANK=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 8
              capabilities: [gpu]

  trainer-node1:
    image: pytorch-training:latest
    volumes:
      - /mnt/shared-nfs/checkpoints:/checkpoints:rw
      - /mnt/shared-nfs/data:/data:ro
    environment:
      - MASTER_ADDR=trainer-node0
      - RANK=1
```

**Checkpoint Size Management:**
```python
# In training script
import os
import glob

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    \"\"\"Keep only last N checkpoints to save space\"\"\"
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*"))
    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            print(f"Removing old checkpoint: {old_ckpt}")
            os.remove(old_ckpt)

# Save checkpoint
torch.save(model.state_dict(), f"/checkpoints/checkpoint-{step}.pt")
cleanup_old_checkpoints("/checkpoints", keep_last_n=3)
# With 140GB checkpoints: max 420GB instead of 1.4TB
```

**Why Options Are Wrong:**
- **Option 0**: Named volumes work but limited to single host - can't share across multi-node training
- **Option 2**: --storage-opt increases container layer but data still lost on crash
- **Option 3**: COPY requires container to be running - can't extract if crashed

**Storage Backend Comparison:**
- **Local SSD**: 3-7 GB/s, cheapest, single node
- **NFS**: 100-300 MB/s, shared, good for checkpoints
- **Lustre/BeeGFS**: 10-50 GB/s, expensive, best for large-scale training
- **AWS EFS**: 1-3 GB/s, managed, easy setup
""",
            "Hard",
            230
        ),

        # Q5: Docker Compose for ML Stacks
        create_question(
            "Q5: You need to deploy a local ML stack: inference API (FastAPI + GPU), Redis cache, PostgreSQL vector DB (pgvector), and Prometheus monitoring. Which Docker Compose pattern ensures correct startup order and resource allocation?",
            [
                "Use 'depends_on' only - simple but inference may start before DB ready",
                "Use 'depends_on' with 'condition: service_healthy' + healthchecks for each service. Set GPU reservation for inference, memory limits for Redis/Postgres. Use networks to isolate services",
                "Run each service in separate docker run commands",
                "Use docker-compose links (deprecated)"
            ],
            1,
            """Senior Explanation: Docker Compose is ideal for local ML development stacks. Proper health checks + resource limits prevent race conditions and resource contention.

**Production-Grade Compose File:**
```yaml
version: '3.8'

services:
  postgres-vector:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_DB: embeddings
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mluser"]
      interval: 5s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 4G  # Vector operations memory-intensive
        reservations:
          memory: 2G
    networks:
      - ml-backend

  redis-cache:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 3s
      timeout: 3s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 2G
    networks:
      - ml-backend

  inference-api:
    build: ./inference
    depends_on:
      postgres-vector:
        condition: service_healthy  # Wait for DB ready
      redis-cache:
        condition: service_healthy  # Wait for cache ready
    environment:
      - DATABASE_URL=postgresql://mluser:${DB_PASSWORD}@postgres-vector/embeddings
      - REDIS_URL=redis://redis-cache:6379
      - MODEL_PATH=/models/llama-2-7b
    volumes:
      - /mnt/models:/models:ro  # Read-only model access
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 60s  # Allow model loading time
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - ml-backend
      - ml-frontend

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"
    networks:
      - ml-backend

volumes:
  pgdata:
  prometheus-data:

networks:
  ml-backend:  # Internal services
  ml-frontend:  # External access
```

**Health Check Importance:**
```python
# Without healthcheck (RACE CONDITION):
# 1. Postgres starts (container up)
# 2. Inference API starts immediately (depends_on satisfied)
# 3. API tries to connect to DB → ERROR (DB still initializing)
# Result: Inference API crashes before DB ready

# With healthcheck:
# 1. Postgres starts, runs initialization (10-15 seconds)
# 2. Healthcheck runs pg_isready → FAIL → FAIL → SUCCESS
# 3. Inference API starts ONLY after healthcheck succeeds
# Result: Clean startup, no errors
```

**Startup Sequence Validation:**
```bash
docker-compose up -d
docker-compose ps

# Expected output:
# postgres-vector   healthy   Up 15 seconds
# redis-cache       healthy   Up 10 seconds
# inference-api     healthy   Up 5 seconds (started last)
# prometheus        healthy   Up 5 seconds
```

**Resource Limits Impact:**
```
Without limits (BAD):
- Redis uses 8GB (caches too aggressively)
- Postgres uses 6GB
- Inference uses 16GB
- Total: 30GB (OOM on 32GB host)

With limits (GOOD):
- Redis: 2GB max (LRU eviction)
- Postgres: 4GB max
- Inference: 16GB max
- Total: 22GB (safe on 32GB host)
```

**Why Options Are Wrong:**
- **Option 0**: depends_on only checks container start, not readiness → race conditions
- **Option 2**: Harder to manage, no service discovery, manual networking
- **Option 3**: 'links' deprecated since Docker 1.10, use networks instead

**Development Workflow:**
```bash
# Start stack
docker-compose up -d

# View logs
docker-compose logs -f inference-api

# Scale inference (add more workers)
docker-compose up -d --scale inference-api=3

# Stop and clean
docker-compose down -v  # Remove volumes too
```
""",
            "Hard",
            240
        ),

        # Q6: Container Resource Limits (GPU/Memory)
        create_question(
            "Q6: Your training container uses 4x A100 GPUs (40GB each) but frequently gets OOM killed. nvidia-smi shows 35GB used per GPU. The host has 512GB RAM. What's the MOST LIKELY issue?",
            [
                "GPU memory full - reduce batch size",
                "Missing --shm-size flag. PyTorch DataLoader uses /dev/shm for multiprocessing. Default 64MB causes OOM with num_workers>0. Set --shm-size=32g for safe multi-GPU training",
                "Need to set --memory limit higher",
                "CUDA out of memory - upgrade to 80GB A100s"
            ],
            1,
            """Senior Explanation: PyTorch multiprocessing DataLoaders use **shared memory (/dev/shm)** to pass data between processes. Docker's default 64MB is catastrophically insufficient.

**The Problem:**
```python
# Training script
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # 8 worker processes
    pin_memory=True
)

# Each worker loads batch into /dev/shm
# Batch size: 32 images × 3 channels × 224 × 224 × 4 bytes (fp32) = 57MB per batch
# 8 workers × 57MB = 456MB needed
# Docker default: 64MB
# Result: OSError: [Errno 28] No space left on device
```

**The Fix:**
```bash
# WRONG (default)
docker run --gpus all pytorch-training python train.py
# /dev/shm size: 64MB (too small)

# CORRECT
docker run --gpus all --shm-size=32g pytorch-training python train.py
# /dev/shm size: 32GB (safe for 8 workers × 4 GPUs)
```

**Calculating Required shm-size:**
```
Formula: shm-size ≥ num_workers × batch_size × data_size_per_sample × 2

Example (4x A100, DDP training):
- num_workers: 8 per GPU × 4 GPUs = 32 total workers
- batch_size: 32 per GPU
- data_size: 224×224×3×4 bytes = 600KB per image
- Buffer factor: 2× (for double buffering)

Required: 32 × 32 × 600KB × 2 = 1.2GB

Safe value: 32GB (allows headroom for larger batches)
```

**Verification Inside Container:**
```bash
# Check shm size
docker exec -it training-container df -h | grep shm

# Output:
# BEFORE: shm  64M   64M    0  100% /dev/shm  ❌ FULL
# AFTER:  shm  32G   1.2G  31G   4% /dev/shm  ✅ HEALTHY
```

**Error Messages to Watch For:**
```python
# Common errors indicating shm issue:
# 1. OSError: [Errno 28] No space left on device
# 2. RuntimeError: DataLoader worker (pid XXXX) is killed by signal: Bus error
# 3. ERROR: Unexpected bus error encountered in worker
```

**Production Best Practices:**
```yaml
# docker-compose.yml
services:
  training:
    image: pytorch-training
    shm_size: '32gb'  # Explicit shm size
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]
```

**Alternative: Use File-Based Sharing (Not Recommended):**
```python
# If you can't increase shm-size (e.g., restricted environment)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive
    multiprocessing_context='forkserver'  # Use file-based sharing
)
# Performance: 20-30% slower than shm-based
```

**Why Options Are Wrong:**
- **Option 0**: nvidia-smi shows 35GB/40GB - GPU memory is fine (12.5% free)
- **Option 2**: --memory limits system RAM, not related to DataLoader shm issue
- **Option 3**: 5GB GPU memory free - not a CUDA OOM issue

**Real-World Impact:**
- Without fix: Training crashes after 10-50 iterations (when shm fills)
- With fix: Stable training for days/weeks
- Symptom: Works with num_workers=0 (no multiprocessing) but fails with num_workers>0
""",
            "Hard",
            240
        ),

        # Q7: Image Size Optimization
        create_question(
            "Q7: Your inference image is 8.5GB (PyTorch 2.0 + transformers). Deployment to 100 edge devices takes 4 hours (image pull). Which combination reduces image to ~2GB without losing functionality?",
            [
                "Use Alpine Linux base - smallest base image",
                "Use python:3.10-slim base + torch.hub.load with weights_only + remove pip/setuptools + multi-stage build. Reduces base (130MB vs 1GB), loads only model weights (no optimizer states), removes build tools",
                "Compress image with gzip",
                "Use Docker save/load to optimize layers"
            ],
            1,
            """Senior Explanation: ML image bloat comes from: (1) heavy base images, (2) full model checkpoints, (3) unnecessary build tools, (4) package cache.

**Size Breakdown Analysis:**
```
Original Image (8.5GB):
├── Ubuntu base: 1.2GB
├── PyTorch full: 3.5GB (includes CUDA, cuDNN, development headers)
├── Transformers: 1.8GB (many dependencies)
├── Model checkpoint: 1.5GB (includes optimizer states, training config)
├── Pip cache: 0.5GB
└── Build tools: 0.3GB (gcc, make, etc.)
```

**Optimization Strategy:**

```dockerfile
# ============================================
# Stage 1: Model Downloader (Builder)
# ============================================
FROM python:3.10-slim as model-builder

# Install minimal deps for download
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir transformers

# Download model, save only weights
RUN python -c "
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Save only model weights (not optimizer, scheduler, etc.)
torch.save({
    'model_state_dict': model.state_dict(),
}, '/model_weights_only.pt')

tokenizer.save_pretrained('/tokenizer')
"

# ============================================
# Stage 2: Minimal Runtime
# ============================================
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies (no cache)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    fastapi uvicorn

# Copy ONLY weights (not full checkpoint)
COPY --from=model-builder /model_weights_only.pt /models/
COPY --from=model-builder /tokenizer /models/tokenizer/

# Copy application code
COPY inference.py .

# Remove pip and setuptools (not needed at runtime)
RUN pip uninstall -y pip setuptools

CMD ["python", "inference.py"]
```

**Size Comparison:**
```
Before optimizations:
├── python:3.10 (full):        1.0GB
├── PyTorch (full checkpoint): 3.5GB
├── Transformers + deps:       2.0GB
├── Model checkpoint:          1.5GB (includes optimizer states)
├── Build tools:               0.5GB
Total:                         8.5GB

After optimizations:
├── python:3.10-slim:          130MB
├── PyTorch (CPU-only):        800MB
├── Transformers (minimal):    400MB
├── Model weights only:        450MB (vs 1.5GB full checkpoint)
├── Application code:          50MB
Total:                         1.8GB
Reduction:                     78.8% (6.7GB saved)
```

**Checkpoint Size Reduction:**
```python
# Full checkpoint (1.5GB) - includes training state
torch.save({
    'epoch': 10,
    'model_state_dict': model.state_dict(),          # 400MB
    'optimizer_state_dict': optimizer.state_dict(),  # 800MB (Adam: 2× params)
    'scheduler_state_dict': scheduler.state_dict(),  # 100MB
    'loss': 0.25,
    'training_args': {...}                           # 200MB
}, 'full_checkpoint.pt')

# Weights-only (450MB) - inference only
torch.save({
    'model_state_dict': model.state_dict()  # 400MB
}, 'weights_only.pt')

# 67% smaller
```

**Deployment Time Impact:**
```
Edge deployment scenario: 100 devices, 100 Mbps network

Before (8.5GB image):
- Pull time per device: 8.5GB / 12.5MB/s = 11.3 minutes
- Sequential deployment (100 devices): 18.8 hours
- Parallel deployment (10 at a time): 1.9 hours

After (1.8GB image):
- Pull time per device: 1.8GB / 12.5MB/s = 2.4 minutes
- Sequential deployment: 4 hours
- Parallel deployment: 24 minutes

Savings: 92% faster deployment
```

**Why Options Are Wrong:**
- **Option 0**: Alpine breaks many Python packages (musl vs glibc). PyTorch wheels incompatible.
- **Option 2**: gzip doesn't reduce image size - Docker layers already compressed
- **Option 3**: save/load doesn't optimize - it's for backup/restore

**Advanced: Layer Deduplication:**
```bash
# Check layer sharing
docker history my-inference-image

# Expected pattern:
# 130MB - python:3.10-slim base (shared across all Python images)
# 800MB - PyTorch layer (shared across all PyTorch apps)
# 450MB - Model weights (unique per model)
# 50MB  - App code (changes frequently)

# When deploying multiple models, base layers pulled once
# Total for 5 models: 130MB + 800MB + (5 × 450MB) + (5 × 50MB) = 3.4GB
# vs non-shared: 5 × 1.8GB = 9GB
```

**Production Validation:**
```bash
# Check actual image size
docker images my-inference-image

# Measure pull time
time docker pull my-registry/my-inference-image:latest

# Verify functionality
docker run my-inference-image python -c "import torch; print(torch.__version__)"
```
""",
            "Hard",
            250
        ),

        # Q8: BuildKit Cache Mounts
        create_question(
            "Q8: Building ML images with 'RUN pip install -r requirements.txt' re-downloads 15GB of packages (PyTorch, TensorFlow) every time, even when requirements.txt hasn't changed. Layer caching works but build server is ephemeral (cache lost daily). What's the solution?",
            [
                "Use --cache-from to import cache from registry",
                "Use BuildKit cache mounts: 'RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt'. Cache persists in BuildKit's cache storage across builds, even if builder is ephemeral. Saves 15GB download every build",
                "Pre-download packages to Dockerfile COPY",
                "Use a requirements.lock file"
            ],
            1,
            """Senior Explanation: BuildKit cache mounts provide **persistent cache storage** independent of layer cache, perfect for package managers (pip, apt, npm).

**The Problem with Layer Cache:**
```dockerfile
# Traditional approach
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt  # Downloads 15GB

# Layer cache works ONLY if:
# 1. Base image unchanged
# 2. requirements.txt unchanged
# 3. Build environment has cached layers

# On ephemeral CI runners (fresh daily):
# - Layer cache empty every day
# - Re-downloads 15GB (30min on 100Mbps)
```

**BuildKit Cache Mount Solution:**
```dockerfile
# syntax=docker/dockerfile:1.4  # Enable BuildKit

FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .

# Cache mount persists across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# How it works:
# Build 1 (cold cache): Downloads 15GB to /root/.cache/pip
#                       BuildKit saves this to persistent cache ID
# Build 2 (warm cache): Mounts same cache, pip sees cached packages
#                       Downloads: 0GB (uses cache)
# Time: 30min → 2min (install from cache, no download)
```

**Cache Mount Types:**

**1. Pip Cache:**
```dockerfile
# Shares pip download cache across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision transformers

# Cache location: /var/lib/docker/buildkit/cache/pip-xxxxx
# Persistent across builds, even if builder destroyed
```

**2. Apt Cache (for system packages):**
```dockerfile
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y build-essential cuda-toolkit
```

**3. Model Download Cache:**
```dockerfile
# Cache HuggingFace model downloads
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from transformers import AutoModel; \
               AutoModel.from_pretrained('meta-llama/Llama-2-70b')"

# 70B model (130GB) downloaded once, reused across builds
```

**Performance Comparison:**

```
Scenario: Build ML training image daily on CI

Packages:
- PyTorch: 2.5GB
- TensorFlow: 3.0GB
- Transformers: 1.5GB
- NumPy, Pandas, etc.: 1.0GB
Total: 8GB download

Without cache mounts (ephemeral CI):
- Download time: 8GB @ 100Mbps = 11 minutes
- Install time: 3 minutes
- Total: 14 minutes
- Over 30 builds/month: 7 hours wasted downloading

With cache mounts:
- Build 1: 14 minutes (cold cache)
- Build 2-30: 3 minutes (cache hit)
- Total month: 14 + (29 × 3) = 101 minutes
- Savings: 80% (5.3 hours)
```

**Advanced: Sharing Cache Across Projects:**
```dockerfile
# Use shared cache ID for common dependencies
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-ml-base \
    pip install torch torchvision transformers

# Different Dockerfile, same cache ID = shares cache
# Useful for monorepo with multiple ML services
```

**Docker Buildx Setup:**
```bash
# Create builder with cache support
docker buildx create --name ml-builder --driver docker-container
docker buildx use ml-builder

# Build with cache export (to registry)
docker buildx build \
  --cache-from type=registry,ref=myregistry/app:cache \
  --cache-to type=registry,ref=myregistry/app:cache,mode=max \
  -t myregistry/app:latest \
  --push .

# mode=max: Export all layers (not just final image)
```

**Cache Inspection:**
```bash
# View BuildKit cache usage
docker buildx du

# Output:
# ID                SIZE      LAST ACCESSED
# pip-xxxxx         15.2GB    2 minutes ago
# apt-xxxxx         2.1GB     1 hour ago
# huggingface-xxx   130GB     1 day ago

# Clean old cache
docker buildx prune --filter until=168h  # Remove cache >7 days old
```

**Why Options Are Wrong:**
- **Option 0**: --cache-from works but requires pushing layers to registry (bandwidth intensive)
- **Option 2**: COPY packages into image → bloats final image size
- **Option 3**: requirements.lock doesn't prevent re-downloading, just pins versions

**Production Pattern:**
```dockerfile
# Combine layer caching + cache mounts for best results
FROM python:3.10-slim

# Layer cache: Rebuild only if requirements change
COPY requirements.txt .

# Cache mount: Avoid re-downloading packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Result:
# - requirements.txt unchanged: 2min build (layer + cache hit)
# - requirements.txt changed: 5min build (cache hit, but layer rebuilds)
# - Cold cache (new CI runner): 30min (downloads to cache)
# - Subsequent builds on runner: 2-5min (cache warm)
```
""",
            "Hard",
            260
        ),

        # Q9: Security - Non-Root User
        create_question(
            "Q9: Your ML inference container runs as root (UID 0). Security audit flags this as high risk. What's the CORRECT way to run as non-root while maintaining write access to /app/logs and /tmp/model_cache?",
            [
                "Add 'RUN chmod 777 /app' to Dockerfile",
                "Create non-root user, chown directories, switch user: 'RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app', then 'USER mluser'. Also use --user flag in docker run for override safety",
                "Use --user flag in docker run only (no Dockerfile changes)",
                "Run container with --privileged flag"
            ],
            1,
            """Senior Explanation: Running containers as root is a **major security risk** - container breakout = root on host. Always use non-root users.

**The Security Risk:**
```bash
# Container running as root (BAD)
docker run -it pytorch-inference bash
whoami  # Output: root

# If attacker exploits container (e.g., CVE in library):
# - Root inside container = root outside container (in many configurations)
# - Can modify host files (if volumes mounted)
# - Can access Docker socket (if mounted)
# - Privilege escalation to host
```

**Secure Pattern:**
```dockerfile
FROM python:3.10-slim

# Create non-root user with specific UID/GID
# UID 1000 is convention for first non-system user
RUN groupadd -g 1000 mluser && \
    useradd -m -u 1000 -g mluser mluser

WORKDIR /app

# Install packages as root (required for pip global install)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories that need write access
RUN mkdir -p /app/logs /tmp/model_cache

# Set ownership to non-root user BEFORE switching
RUN chown -R mluser:mluser /app /tmp/model_cache

# Copy application code
COPY --chown=mluser:mluser . .

# Switch to non-root user for runtime
USER mluser

# Verify non-root
RUN whoami  # Should print: mluser

CMD ["python", "inference.py"]
```

**Permission Validation:**
```bash
# Build and run
docker build -t secure-inference .
docker run -it secure-inference bash

# Inside container
whoami  # mluser (not root)
id     # uid=1000(mluser) gid=1000(mluser)

# Test write access
echo "test" > /app/logs/test.log  # Success (owned by mluser)
echo "test" > /etc/hosts          # Permission denied (owned by root)
```

**Volume Mount Gotcha:**
```bash
# Host directory owned by different user
ls -la /host/data
# drwxr-xr-x root root /host/data

# Mount into container
docker run -v /host/data:/data secure-inference

# Inside container (running as mluser UID 1000)
touch /data/test.txt  # Permission denied (host dir owned by root)

# FIX: Match host and container UIDs
# Option 1: Change host directory ownership
sudo chown -R 1000:1000 /host/data

# Option 2: Use runtime --user flag (override Dockerfile USER)
docker run -v /host/data:/data --user $(id -u):$(id -g) secure-inference
```

**Security Comparison:**

| Aspect | Root User | Non-Root User |
|--------|-----------|---------------|
| Container breakout risk | High (root → root on host) | Low (UID 1000 → unprivileged) |
| File access | All files | Only owned files |
| Install packages | Yes | No (without sudo) |
| Bind to port <1024 | Yes | No |
| Best practice | ❌ Never in production | ✅ Required for security |

**Advanced: Read-Only Filesystem:**
```dockerfile
# Ultimate security: read-only root filesystem
FROM python:3.10-slim

RUN groupadd -g 1000 mluser && \
    useradd -m -u 1000 -g mluser mluser

# Install everything as root
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only these directories writable
RUN mkdir -p /tmp/model_cache /app/logs && \
    chown -R mluser:mluser /tmp/model_cache /app/logs

COPY --chown=mluser:mluser . /app
WORKDIR /app

USER mluser

# Run with read-only root
# docker run --read-only --tmpfs /tmp --tmpfs /app/logs secure-inference
```

**Kubernetes SecurityContext (Production):**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-inference
spec:
  securityContext:
    runAsNonRoot: true  # Enforce non-root
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000       # Files created with GID 1000
  containers:
  - name: inference
    image: secure-inference:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL  # Drop all Linux capabilities
    volumeMounts:
    - name: model-cache
      mountPath: /tmp/model_cache
    - name: logs
      mountPath: /app/logs
```

**Why Options Are Wrong:**
- **Option 0**: chmod 777 makes files world-writable - security nightmare
- **Option 2**: --user flag works but fragile (must remember on every docker run)
- **Option 3**: --privileged INCREASES attack surface (gives almost all host capabilities)

**Production Checklist:**
✅ Non-root USER in Dockerfile
✅ Minimal file permissions (chown only necessary directories)
✅ Read-only root filesystem where possible
✅ Drop unnecessary Linux capabilities
✅ Use distroless or slim base images (fewer attack vectors)
✅ Regular security scanning (docker scan, trivy)
""",
            "Hard",
            250
        ),

        # Q10: Registry Management for Large Models
        create_question(
            "Q10: You need to push a Docker image containing a 65B model (130GB total image size) to a private registry. 'docker push' fails after 2 hours with 'blob upload unknown'. What's the proper solution for large ML images?",
            [
                "Split model into multiple images and compose them",
                "Use external model storage (S3/GCS) + download at runtime. Keep image <5GB (app code + framework). Use init containers or entrypoint script to fetch model from object storage. Enables versioning, faster pulls, registry-agnostic",
                "Increase Docker Hub upload timeout",
                "Use docker save/load to transfer via USB drive"
            ],
            1,
            """Senior Explanation: Docker registries are NOT designed for 100GB+ images. Large ML models should use **object storage** (S3, GCS) + pull at runtime.

**The Registry Problem:**
```
Docker Registry Limitations:
- Max layer size: Varies (5GB typical, 10GB max in most registries)
- Upload timeout: 1-2 hours
- Storage cost: $$$ (registry storage 3-5× more expensive than S3)
- Bandwidth: Limited (especially Docker Hub free tier)
- Pull time: Slow (sequential layer pulls)

130GB Image Push:
- Upload time: 2-3 hours @ 100Mbps
- Fails on network hiccup (no resume in Docker v1 API)
- Registry cost: $20-30/month just for storage
- Pull time on deployment: 1-2 hours (ouch)
```

**Recommended Pattern: External Model Storage**

**1. Dockerfile (Lightweight):**
```dockerfile
FROM python:3.10-slim

# Install inference framework only (NO MODEL)
RUN pip install --no-cache-dir \
    torch torchvision transformers \
    fastapi uvicorn boto3

# Copy application code
COPY inference.py /app/
COPY download_model.py /app/

WORKDIR /app

# Entrypoint downloads model on startup
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "inference.py"]

# Image size: 3.5GB (vs 130GB with model)
```

**2. Entrypoint Script:**
```bash
#!/bin/bash
# entrypoint.sh

set -e

MODEL_S3_URI=${MODEL_S3_URI:-s3://ml-models/llama-65b/}
MODEL_LOCAL_PATH=${MODEL_LOCAL_PATH:-/models/llama-65b}

echo "Checking for model at $MODEL_LOCAL_PATH..."

if [ ! -d "$MODEL_LOCAL_PATH" ]; then
    echo "Model not found. Downloading from $MODEL_S3_URI..."

    # Download with progress and resume support
    aws s3 sync "$MODEL_S3_URI" "$MODEL_LOCAL_PATH" \
        --no-progress \
        --only-show-errors

    echo "Model download complete. Size:"
    du -sh "$MODEL_LOCAL_PATH"
else
    echo "Model already present. Skipping download."
fi

# Execute main command
exec "$@"
```

**3. Deployment (Kubernetes Example):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-65b-inference
spec:
  replicas: 2
  template:
    spec:
      # Init container downloads model once
      initContainers:
      - name: model-downloader
        image: amazon/aws-cli:latest
        command:
          - sh
          - -c
          - |
            if [ ! -f /models/llama-65b/.downloaded ]; then
              aws s3 sync s3://ml-models/llama-65b/ /models/llama-65b/
              touch /models/llama-65b/.downloaded
            fi
        volumeMounts:
        - name: model-storage
          mountPath: /models
        env:
        - name: AWS_REGION
          value: us-west-2

      containers:
      - name: inference
        image: myregistry/llama-inference:latest  # Only 3.5GB
        volumeMounts:
        - name: model-storage
          mountPath: /models
        env:
        - name: MODEL_PATH
          value: /models/llama-65b

      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc  # 200GB PVC shared across pods
```

**Performance Comparison:**

```
Scenario: Deploy to 10 GPU nodes

APPROACH 1: Model in Docker Image (130GB)
- Initial: 10 nodes × 130GB = 1.3TB total pulls
- Pull time per node: 130GB @ 1Gbps = 17 minutes
- Total deployment time: 17 minutes (parallel pulls)
- Registry egress cost: 1.3TB × $0.08/GB = $104
- Per-update cost: $104 (even if only code changed)

APPROACH 2: Model in S3 (3.5GB image)
- Image pull: 10 nodes × 3.5GB = 35GB
- Pull time: 3.5GB @ 1Gbps = 28 seconds
- Model download: 130GB from S3 to shared PVC = 1 minute (10Gbps)
- Total deployment time: 90 seconds
- Costs:
  - Image egress: 35GB × $0.08/GB = $2.80
  - S3 egress: 130GB × $0.09/GB = $11.70
  - Total: $14.50
- Per-update cost (code change): $2.80 (model cached)

Savings: 86% cost, 94% faster deployments
```

**Model Versioning:**
```bash
# S3 structure for model versions
s3://ml-models/
├── llama-65b/
│   ├── v1.0/
│   │   ├── model.safetensors
│   │   └── config.json
│   ├── v1.1/
│   │   └── ...
│   └── v2.0/
│       └── ...

# Deployment specifies version
env:
  - name: MODEL_S3_URI
    value: s3://ml-models/llama-65b/v1.1/
  - name: MODEL_VERSION
    value: "1.1"
```

**Caching Strategy:**
```python
# download_model.py - Smart caching
import os
import boto3
from pathlib import Path

def download_model_if_needed(s3_uri, local_path, model_version):
    version_file = Path(local_path) / ".version"

    # Check if correct version already present
    if version_file.exists():
        cached_version = version_file.read_text().strip()
        if cached_version == model_version:
            print(f"Model v{model_version} already cached.")
            return

    print(f"Downloading model v{model_version}...")
    # Use aws s3 sync for resume support
    os.system(f"aws s3 sync {s3_uri} {local_path}")

    version_file.write_text(model_version)
```

**Why Options Are Wrong:**
- **Option 0**: Splitting images is complex, Docker doesn't support image composition
- **Option 2**: Docker Hub has hard limits, can't "increase timeout" arbitrarily
- **Option 3**: USB transfer not viable for production/CD pipelines

**Alternative: Container Layer Registry (Advanced):**
```bash
# For registries that support OCI artifacts
# Push model as separate artifact, reference in image
docker push myregistry/llama-65b-model:v1.0  # 130GB model only
docker build -t myregistry/llama-app:latest . # 3.5GB app only

# Runtime: Compose both
docker run \
  --mount type=bind,source=$(docker volume create llama-model),target=/models \
  myregistry/llama-app:latest
```

**Production Recommendation:**
✅ Use S3/GCS for models >10GB
✅ Version models separately from code
✅ Use init containers or entrypoint downloads
✅ Cache models on persistent volumes
✅ Monitor download times and add retry logic
✅ Consider model compression (quantization, pruning)
""",
            "Hard",
            270
        ),

        # Q11: .dockerignore Optimization
        create_question(
            "Q11: Your ML project directory is 50GB (datasets, checkpoints, logs). 'docker build' takes 15 minutes just uploading context to daemon. What's the correct .dockerignore pattern?",
            [
                "Add '*' to ignore everything, then '!*.py' to include code",
                "Ignore large files: 'data/**, checkpoints/**, logs/**, *.pt, *.bin, .git/, __pycache__/'. Only send necessary source code to daemon. Reduces context from 50GB to ~100MB, build from 15min to 30sec",
                "Use docker build --no-cache to skip context upload",
                "Mount volumes instead of COPY"
            ],
            1,
            """Senior Explanation: Docker build context is sent ENTIRELY to daemon before build starts. Large contexts (datasets, checkpoints) cause massive delays.

**The Problem:**
```bash
# Project structure (50GB)
ml-project/
├── data/              # 30GB - training datasets
├── checkpoints/       # 15GB - saved model checkpoints
├── logs/              # 3GB - TensorBoard logs
├── .git/              # 1.5GB - git history
├── models/            # 500MB - downloaded pretrained models
├── __pycache__/       # 200MB - Python bytecode
├── .pytest_cache/     # 100MB
├── venv/              # 2GB - virtual environment
├── src/               # 50MB - actual source code
│   ├── train.py
│   ├── inference.py
│   └── utils/
├── Dockerfile
└── requirements.txt

# Without .dockerignore:
docker build -t ml-app .

# What happens:
# 1. Uploads ENTIRE 50GB to Docker daemon
#    Time: 50GB @ 500MB/s (local SSD) = 100 seconds
#    Time: 50GB @ 100MB/s (network mount) = 8+ minutes
# 2. Then build starts (another 5-10 minutes)
# Total: 15+ minutes, even if only changed 1 line of code
```

**Optimal .dockerignore:**
```
# .dockerignore - Exclude from build context

# Large data directories (never needed in image)
data/
datasets/
checkpoints/
*.pt
*.pth
*.bin
*.safetensors
wandb/
tensorboard/
logs/
outputs/

# Git and version control
.git/
.gitignore
.gitattributes

# Python cache and environments
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.venv/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter notebooks (usually not needed in production)
*.ipynb
.ipynb_checkpoints/

# Documentation
docs/
*.md
!README.md  # Exception: include README

# CI/CD
.github/
.gitlab-ci.yml
.travis.yml

# Docker files (don't include in context)
.dockerignore
Dockerfile*
docker-compose*.yml

# Large model downloads
models/
*.h5
*.onnx
*.tflite

# OS files
Thumbs.db
```

**Size Reduction:**
```bash
# Check context size before .dockerignore
docker build --no-cache -t ml-app . 2>&1 | grep "Sending build context"
# Output: Sending build context to Docker daemon  50.2GB

# After .dockerignore
# Output: Sending build context to Docker daemon  87.3MB

# Reduction: 99.8% (50GB → 87MB)
# Build time: 15 minutes → 35 seconds
```

**Advanced Pattern: Whitelist Approach**
```
# .dockerignore - More restrictive (whitelist)

# Ignore everything
*

# Explicitly include what's needed
!src/
!requirements.txt
!setup.py
!README.md

# But exclude Python cache even within allowed dirs
**/__pycache__
**/*.pyc
```

**Verification:**
```bash
# See what's in context
docker build --no-cache -t ml-app . 2>&1 | tee build.log
grep "Step" build.log

# Advanced: Actually inspect context
# Create a build that copies everything to /context
cat > Dockerfile.debug <<'EOF'
FROM alpine
COPY . /context
RUN du -sh /context/* | sort -h
CMD ["/bin/sh"]
EOF

docker build -f Dockerfile.debug -t context-inspector .
docker run --rm context-inspector

# Should show only src/, requirements.txt, etc.
```

**Real-World Impact:**

```
Development workflow: 100 builds per day

Without .dockerignore:
- Context upload: 50GB × 100 = 5TB uploaded daily
- Time: 100 × 100sec = 166 minutes wasted
- SSD wear: 5TB writes per day

With .dockerignore:
- Context upload: 87MB × 100 = 8.7GB daily
- Time: 100 × 0.5sec = 50 seconds
- SSD wear: 8.7GB writes per day

Savings:
- 99.8% less data transfer
- 99.5% faster context upload
- 576× less SSD wear (extends drive life)
```

**Common Gotchas:**

**1. .dockerignore applies to COPY commands too:**
```dockerfile
# Even explicit COPY affected by .dockerignore
COPY data/ /app/data/  # Won't work if data/ in .dockerignore

# Solution: Use volume mount for data
docker run -v ./data:/app/data ml-app
```

**2. Comments and blank lines:**
```
# .dockerignore supports comments
data/       # Training datasets
logs/       # TensorBoard logs

# Blank lines ignored
```

**3. Pattern matching:**
```
# Wildcards
*.log       # All .log files
temp*       # temp, temp1, temporary, etc.
**/*.pyc    # All .pyc files recursively

# Negation (exception)
*.md        # Ignore all markdown
!README.md  # Except README
```

**Why Options Are Wrong:**
- **Option 0**: Whitelist with '!*.py' misses other needed files (yaml configs, etc.)
- **Option 2**: --no-cache doesn't affect context upload, only layer caching
- **Option 3**: Volumes are for runtime, not build. Can't COPY from volumes during build.

**CI/CD Optimization:**
```yaml
# GitHub Actions example
- name: Build Docker image
  run: |
    # Verify .dockerignore working
    CONTEXT_SIZE=$(du -sb . | cut -f1)
    echo "Context size: $(numfmt --to=iec $CONTEXT_SIZE)"

    if [ $CONTEXT_SIZE -gt 104857600 ]; then  # 100MB
      echo "⚠️ Context too large! Check .dockerignore"
      exit 1
    fi

    docker build -t ml-app .
```

**Production Checklist:**
✅ .dockerignore in root of build context
✅ Exclude data, checkpoints, logs, git
✅ Exclude Python cache and venvs
✅ Include only source code + requirements
✅ Verify context size < 100MB for typical apps
✅ Use whitelist pattern for maximum security
""",
            "Hard",
            230
        ),

        # Q12: Docker Layer Best Practices
        create_question(
            "Q12: Your Dockerfile has 30 RUN commands (apt install, pip install, mkdir, etc.). This creates 30+ layers, slow builds, and large images. What's the BEST refactoring strategy?",
            [
                "Combine all RUN commands into one with && - but separate logical groups (system deps → Python deps → app setup). Use backslash for readability. Reduces layers from 30 to 3-4, improves cache granularity",
                "Keep all commands separate for better caching",
                "Use ENTRYPOINT script to run commands at runtime",
                "Use --squash flag to merge all layers"
            ],
            0,
            """Senior Explanation: Each Dockerfile instruction creates a layer. Too many layers = slow, too few = poor cache hit rate. Balance is key.

**The Problem (30 Layers):**
```dockerfile
FROM python:3.10-slim

# 30+ separate RUN commands (BAD)
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y build-essential
RUN apt-get install -y curl
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip install numpy
RUN pip install pandas
RUN pip install torch
RUN pip install transformers
# ... 15 more pip installs

RUN mkdir /app
RUN mkdir /app/models
RUN mkdir /app/data
RUN mkdir /app/logs

RUN useradd mluser
RUN chown -R mluser:mluser /app

# Issues:
# - 30+ layers (image metadata bloat)
# - Poor caching (change torch version → rebuild numpy)
# - Slow pull (more HTTP requests)
# - Large image (each layer has overhead)
```

**Optimal Strategy (3-4 Layers):**
```dockerfile
FROM python:3.10-slim

# Layer 1: System dependencies (rarely changes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: Python dependencies (changes occasionally)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Layer 3: Application setup (changes rarely)
RUN useradd -m -u 1000 mluser && \
    mkdir -p /app/models /app/data /app/logs && \
    chown -R mluser:mluser /app

# Layer 4: Application code (changes frequently)
COPY --chown=mluser:mluser . /app
WORKDIR /app
USER mluser

CMD ["python", "app.py"]
```

**Layer Optimization Principles:**

**1. Group by Change Frequency:**
```dockerfile
# Rarely changes (once per base image update)
RUN apt-get update && apt-get install ...

# Occasionally changes (when dependencies update)
RUN pip install -r requirements.txt

# Frequently changes (every code commit)
COPY . /app
```

**2. Combine Related Commands:**
```dockerfile
# GOOD: Logical grouping
RUN apt-get update && \
    apt-get install -y \
        package1 \
        package2 \
        package3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# BAD: Too granular
RUN apt-get update
RUN apt-get install -y package1
RUN apt-get install -y package2
RUN apt-get clean

# BAD: Everything together (poor caching)
RUN apt-get update && \
    apt-get install -y package1 package2 && \
    pip install -r requirements.txt && \
    mkdir /app && \
    COPY . /app
```

**3. Use Multi-Line for Readability:**
```dockerfile
# Easier to read and modify
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev \
        git \
        curl \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

**Performance Comparison:**

```
Scenario: Update one Python package

30-Layer Dockerfile:
- Layer cache: Invalidated from pip install torch onward (15 layers)
- Rebuild time: 8 minutes (re-runs all subsequent pip installs)
- Image size: 4.5GB (layer overhead ~50MB)

Optimized 4-Layer Dockerfile:
- Layer cache: Invalidated from requirements.txt layer (1 layer)
- Rebuild time: 3 minutes (re-runs entire pip install, but in one go)
- Image size: 4.0GB (minimal layer overhead)

Scenario: Update application code

30-Layer Dockerfile:
- Layer cache: Invalidated from COPY (last layer usually)
- Rebuild time: 30 seconds

Optimized 4-Layer Dockerfile:
- Layer cache: Invalidated from COPY (last layer)
- Rebuild time: 30 seconds (same)
```

**Advanced: Layer Squashing:**
```bash
# --squash merges all layers into one (experimental)
docker build --squash -t ml-app .

# Pros:
# - Smaller final image (no intermediate file duplication)
# - Single layer = faster pull

# Cons:
# - No layer caching benefits
# - No layer sharing between images
# - Requires experimental features enabled

# When to use:
# - Final production image for size optimization
# - After multi-stage build (squash final stage only)

# Example:
docker build -t ml-app:latest .  # Keep layers for dev
docker build --squash -t ml-app:prod .  # Squash for production
```

**Layer Count Impact:**

| Layers | Build Time (Code Change) | Build Time (Dep Change) | Image Size | Pull Time |
|--------|--------------------------|-------------------------|------------|-----------|
| 30     | 30s                      | 8min                    | 4.5GB      | 45s       |
| 10     | 30s                      | 5min                    | 4.2GB      | 40s       |
| 4      | 30s                      | 3min                    | 4.0GB      | 35s       |
| 1 (squash) | 10min (no cache)     | 10min (no cache)        | 3.8GB      | 30s       |

**Why Options Are Wrong:**
- **Option 1**: Keeping 30 separate commands wastes layers, slows builds, increases image size
- **Option 2**: Moving setup to ENTRYPOINT runs on EVERY container start (slow startup)
- **Option 3**: --squash removes ALL caching benefits, only use for final production image

**Best Practices Summary:**
✅ Group commands by change frequency (system → deps → app)
✅ Combine related RUN commands with && and backslash
✅ Aim for 5-10 layers total (not 1, not 50)
✅ Put frequently changing layers last (COPY code)
✅ Clean up in same layer (apt clean, rm cache)
✅ Use multi-stage builds for complex setups
✅ Consider --squash only for final production images
""",
            "Hard",
            240
        ),

        # Q13-Q20: Simplified versions
        create_question("Q13: ML container running but returns 503 for 2min during model load. Fix?", ["Sleep in entrypoint", "HEALTHCHECK with start-period=120s", "K8s only", "Disable checks"], 1, "HEALTHCHECK with start-period gives grace time for model loading. Prevents premature traffic routing.", "Hard", 200),
        create_question("Q14: M1 build fails on x86 AWS. Solution?", ["x86 CI only", "docker buildx --platform linux/amd64,linux/arm64", "QEMU", "x86 dev"], 1, "buildx creates multi-platform images. Manifest list auto-selects architecture.", "Hard", 190),
        create_question("Q15: ENV HUGGINGFACE_TOKEN=secret in Dockerfile. Secure alternative?", [".env file", "BuildKit --mount=type=secret", "Encrypt", "ONBUILD"], 1, "BuildKit secret mounts pass secrets during build WITHOUT storing in layers.", "Hard", 200),
        create_question("Q16: Container runs as root. How to use non-root with write access?", ["chmod 777", "useradd + chown + USER directive", "--user only", "--privileged"], 1, "Create non-root user, chown directories, USER directive. Never run as root.", "Hard", 180),
        create_question("Q17: 130GB model image push fails. Best approach?", ["Split images", "Store in S3, download at runtime", "Increase timeout", "USB"], 1, "Store models in S3/GCS. Download via init container. Keeps image small.", "Hard", 190),
        create_question("Q18: 50GB build context slow. Fix?", ["* in .dockerignore", ".dockerignore with data/, logs/, .git/", "--no-cache", "Volumes"], 1, ".dockerignore excludes unnecessary files. Reduces context to <100MB.", "Hard", 170),
        create_question("Q19: pip install reruns every build. Optimize caching?", ["--cache-dir", "COPY requirements.txt, RUN pip, then COPY code", ".dockerignore", "Volumes"], 1, "Layer ordering: COPY requirements first (rarely changes), then pip install (cached), then code (changes often).", "Hard", 170),
        create_question("Q20: PyTorch crashes 'No space' but disk has space. Issue?", ["Increase PVC", "Missing --shm-size=32g for DataLoader", "Reduce workers", "hostPath"], 1, "DataLoader uses /dev/shm. Docker default 64MB too small. Set --shm-size=32g.", "Hard", 180),

    ]

    return questions

if __name__ == "__main__":
    questions = populate_senior_docker()
    print(f"Generated {len(questions)} senior-level Docker for ML/AI questions")
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q['question'][:80]}...")
