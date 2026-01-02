"""
Senior AI Engineer Interview Questions - Batch 8: Model Quantization
Topics: GPTQ, AWQ, INT8/INT4 Quantization, Dynamic vs Static, Post-Training Quantization
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_quantization():
    """20 Senior-Level Quantization Questions"""
    questions = [
        # GPTQ (POST-TRAINING QUANTIZATION) (Questions 1-6)
        create_question(
            "Q1: GPTQ quantizes models to 4-bit/3-bit post-training. What is the core algorithm?",
            [
                "K-means clustering to find optimal quantization centroids",
                "Layer-wise optimal quantization minimizing reconstruction error using Hessian inverse (second-order information)",
                "Gradient-based search for quantization parameters",
                "Random quantization with fine-tuning"
            ],
            1,
            "Senior Explanation: GPTQ uses Optimal Brain Quantization (OBQ) approach: For each layer, minimize ||WX - W_quantX||² where X is calibration data. Uses Hessian H = X^T X (second-order curvature) to find optimal per-weight quantization that minimizes reconstruction error. Algorithm: (1) Compute H for layer, (2) Quantize weights one-by-one, updating remaining weights to compensate using H^{-1} (optimal update direction). Complexity: O(n³) for Hessian inverse, but approximations make it O(n²). For 7B model: ~1-2 hours on single GPU. Option A too simple. Option C requires backprop (GPTQ is post-training only). Production: GPTQ achieves 4-bit with <1% perplexity degradation vs fp16 (3-bit: 1-3% degradation). Trade-off: Calibration time (hours) for zero-shot quantization (no training needed).",
            "Hard",
            240
        ),
        create_question(
            "Q2: For GPTQ quantization, how much calibration data is typically needed?",
            [
                "Millions of samples - need to cover full distribution",
                "128-1024 samples (few seconds of text) - captures sufficient statistics",
                "Full training dataset - ensures accuracy",
                "No calibration data - purely algorithmic"
            ],
            1,
            "Senior Explanation: GPTQ needs small calibration set (128-1024 samples, ~5K-40K tokens total) to compute Hessian H = X^T X. More data doesn't significantly improve quality (Hessian converges quickly). Typical: 1024 samples from C4 dataset (~2MB text). Quantization time: Dominated by Hessian computation and optimization, not data volume. For 7B model: 1024 samples → ~2 hours quantization. Option A wasteful (diminishing returns). Option C infeasible (hours/days). Production: C4 subset (open-source corpus) commonly used. Even random Wikipedia text works (task-agnostic). Trade-off: Minimal calibration data requirement makes GPTQ practical for any model. No need for original training data (often unavailable).",
            "Medium",
            180
        ),
        create_question(
            "Q3: GPTQ quantizes a 7B model to 4-bit. What is the inference speedup on GPU vs fp16?",
            [
                "~4× faster - 4-bit means 4× less data",
                "~2× faster - memory bandwidth limited, not compute",
                "~1.5-2× faster - kernel optimization immature, GPU designed for fp16/fp32",
                "No speedup - same compute, just less memory"
            ],
            2,
            "Senior Explanation: Theoretical: 4-bit = 4× less memory bandwidth → 4× faster (if memory-bound). Reality: (1) GPUs optimize for fp16/fp32 (Tensor Cores), not int4. (2) Dequantization overhead: 4-bit weights loaded → dequantized to fp16 → matmul in fp16. (3) Kernel immaturity: Custom CUDA kernels for 4-bit slower than highly-optimized cuBLAS for fp16. Actual speedup: ~1.5-2× on A100/H100 with ExLlama/AutoGPTQ kernels. For batch=1 (latency-critical): ~2× speedup. Larger batches: ~1.5× (compute-bound). Option A 'junior trap' - assumes ideal speedup. Production: Primary benefit is MEMORY reduction (4× less), enabling larger batch sizes (→ higher throughput). Trade-off: Modest latency improvement (1.5-2×) but huge capacity increase (4× larger models on same GPU).",
            "Hard",
            220
        ),
        create_question(
            "Q4: GPTQ quantization is 'asymmetric' (uses zero-point + scale). Why asymmetric vs symmetric?",
            [
                "Asymmetric is faster - simpler computation",
                "Asymmetric better handles skewed weight distributions - can shift zero point to minimize quantization error",
                "Symmetric required for GPU acceleration",
                "No difference - same quality"
            ],
            1,
            "Senior Explanation: Symmetric: quantize to [-127, 127] with scale only (assumes weights centered at 0). Asymmetric: quantize to [0, 255] or [-128, 127] with scale + zero_point (shifts range). Benefit: If weight distribution is [0.5, 3.5] (not centered), asymmetric sets zero_point=128, scale=(3.5-0.5)/255, uses full int8 range. Symmetric would waste half the range (negative values unused). Quality: Asymmetric ~0.5-1% better perplexity for layers with skewed weights (layer norm scales, some attention weights). Overhead: One extra addition per weight (x_quant = clip((x - zero_point) / scale)). Negligible. Option A wrong - asymmetric slightly slower. Production: Most quantization schemes (GPTQ, AWQ, TensorRT-LLM) use asymmetric for better quality. Trade-off: Tiny compute overhead for better accuracy.",
            "Hard",
            200
        ),
        create_question(
            "Q5: After GPTQ quantization, can the model be fine-tuned further?",
            [
                "No - quantized weights are fixed integers",
                "Yes via quantization-aware training (QAT) - simulate quantization during training",
                "Yes with QLoRA - fine-tune LoRA adapters on top of quantized base",
                "Only specific layers can be unfrozen"
            ],
            2,
            "Senior Explanation: GPTQ produces integer weights (not trainable in standard frameworks). Fine-tuning options: (1) QLoRA: Keep GPTQ 4-bit base frozen, add LoRA adapters (bf16), train adapters only. (2) QAT: Dequantize to fp16, fine-tune with fake quantization, re-quantize (complex, less common). Option C (QLoRA on GPTQ) is standard practice. Memory: 7B GPTQ base (3.5GB) + LoRA (30MB) + optimizer (60MB) = ~4GB (fits on 12GB GPU). Option B possible but overkill (GPTQ already near-optimal). Production: Fine-tune quantized LLaMA with QLoRA for domain adaptation. Trade-off: Quantization + LoRA = 2 types of compression, may compound quality loss (~2-3% total). But enables fine-tuning on minimal hardware.",
            "Hard",
            200
        ),
        create_question(
            "Q6: GPTQ quantization is 'group-wise'. What does this mean and why?",
            [
                "Quantize weights in groups (e.g., 128 weights share scale/zero-point) - reduces overhead while maintaining quality",
                "Quantize different model components (attention, FFN) separately",
                "Process layers in groups for faster quantization",
                "Batch multiple samples for Hessian computation"
            ],
            0,
            "Senior Explanation: Group-wise: Divide weight matrix into groups (e.g., 128 weights per group), each group has own scale and zero_point. Benefits: (1) Better quality than per-tensor quantization (one scale/zero-point for entire layer) - adapts to local weight distributions. (2) Less overhead than per-weight quantization. Group size=128: For 4096×4096 matrix (16.8M weights), need 16.8M/128 = 131K scales (262KB in fp16). Overhead: 262KB / 8.4MB = 3%. Quality: group=128 nearly matches per-channel quantization. Option B describes per-layer (coarser). Production: GPTQ uses group=128 by default. Smaller groups (64, 32) → better quality but more overhead. Trade-off: Group size hyperparameter - 128 balances quality and efficiency.",
            "Hard",
            220
        ),

        # AWQ (ACTIVATION-AWARE WEIGHT QUANTIZATION) (Questions 7-12)
        create_question(
            "Q7: AWQ differs from GPTQ by being 'activation-aware'. What does this mean?",
            [
                "Quantizes activations in addition to weights",
                "Analyzes activation distributions to identify important weights (high activation magnitude), protects them from aggressive quantization",
                "Uses activations as calibration data",
                "Requires activation checkpointing during quantization"
            ],
            1,
            "Senior Explanation: AWQ observes that weights contributing to large-magnitude activations are more important (higher impact on output). Algorithm: (1) Run calibration data, record activation magnitudes per channel. (2) Identify 'salient' channels (top 1-5% activation magnitude). (3) Apply per-channel scaling - scale up salient weights before quantization (gets more quantization bins), scale down non-salient weights. (4) Quantize all to 4-bit. Result: Salient weights quantized more accurately. Quality: AWQ slightly better than GPTQ (0.1-0.3% perplexity) for same bit-width. Efficiency: AWQ quantization faster (~10-30 min vs GPTQ 1-2 hours for 7B) - no Hessian computation. Option A wrong - AWQ quantizes weights only (activations stay fp16). Production: TinyChat, vLLM support AWQ. Trade-off: Faster quantization, slightly better quality, but less mature than GPTQ (fewer model support).",
            "Hard",
            240
        ),
        create_question(
            "Q8: AWQ applies per-channel scaling before quantization. How is this scaling factor computed?",
            [
                "Based on weight magnitude - larger weights get larger scale",
                "Based on activation magnitude - channels with larger activations get scaling factor s to optimize quantization error",
                "Learned via gradient descent",
                "Fixed scale (e.g., 1.5) for all salient channels"
            ],
            1,
            "Senior Explanation: AWQ scaling: s_c = (avg_activation_magnitude_c)^α where α ∈ [0, 1] (hyperparameter, typically 0.5). High-activation channels get s > 1 (scale up before quantization), low-activation get s < 1 (scale down). Intuition: High-activation channels' errors amplified in final output → need more precision. Quantize: W_quant = quantize(s × W), then dequantize: W_dequant = dequantize(W_quant) / s. Inference: Absorb scaling into adjacent layer (fuse s into layer norm or previous layer's output). Zero overhead at inference. Option C too expensive (AWQ is post-training). Production: α=0.5 (square root of activation magnitude) works well empirically. Trade-off: Calibration requires forward passes to collect activations (~5-10 min), but much faster than GPTQ's Hessian.",
            "Hard",
            220
        ),
        create_question(
            "Q9: AWQ claims to be 'training-free' like GPTQ. What calibration is still needed?",
            [
                "No calibration - purely based on weight statistics",
                "Forward passes on calibration data to collect activation statistics",
                "Backward passes to compute gradients",
                "Full fine-tuning on small dataset"
            ],
            1,
            "Senior Explanation: AWQ needs forward passes only (no backward, no training). Process: (1) Run 128-1024 samples through model, (2) Collect activation magnitudes per channel in each layer, (3) Compute scaling factors, (4) Quantize. Time: ~10-30 min for 7B model on single GPU. GPTQ also forward-only but computes Hessian (more expensive). Option C/D require gradients/training (AWQ doesn't). Production: Both GPTQ and AWQ are post-training quantization (PTQ) - no training needed, works on pre-trained checkpoints directly. Trade-off: PTQ convenient (no training data/code needed) but limited quality (up to ~4-bit reliably). For 2-3 bit, quantization-aware training (QAT) needed.",
            "Medium",
            180
        ),
        create_question(
            "Q10: For AWQ, what is the typical perplexity degradation for 4-bit quantization of LLaMA-7B?",
            [
                "<0.5% - negligible degradation",
                "1-2% - small acceptable degradation",
                "5-10% - noticeable but usable",
                "15%+ - significant quality loss"
            ],
            0,
            "Senior Explanation: AWQ on LLaMA-7B achieves <0.5% perplexity increase (e.g., fp16: 5.68, AWQ 4-bit: 5.71). For 3-bit: ~1-2% degradation. GPTQ similar (<1% for 4-bit). For comparison, naive round-to-nearest 4-bit: ~10-20% degradation. Option A correct for 4-bit. Option B for 3-bit. Production: 4-bit quantization considered 'production-ready' (minimal quality impact). 3-bit usable for many tasks. 2-bit degrades significantly (5-15%), only for extreme compression needs. Trade-off: 4-bit = 4× memory reduction with <0.5% quality loss (excellent ROI). Common deployment: Serve 4-bit models to maximize throughput.",
            "Medium",
            180
        ),
        create_question(
            "Q11: AWQ quantization supports 'group size' like GPTQ. What group size is typically used?",
            [
                "group=32 - fine-grained quantization",
                "group=128 - standard balanced choice",
                "group=1024 - coarse-grained for efficiency",
                "group=1 (per-weight) - maximum quality"
            ],
            1,
            "Senior Explanation: AWQ typically uses group=128 (same as GPTQ). Smaller groups (32, 64) → better quality (~0.1-0.2% improvement) but more overhead (more scales to store/load). Larger groups (256, 512) → worse quality. For LLaMA-7B: group=128 has ~1.5% overhead (scales storage), group=64 has ~3% overhead. Quality difference: group=64 vs group=128 ≈ 0.1% perplexity. Not worth 2× overhead. Option D (per-weight) impractical (overhead = 100%+ of weights). Production: group=128 default in AWQ, GPTQ, AutoGPTQ libraries. Trade-off: Diminishing returns for group <128, negligible quality gain for significant overhead.",
            "Medium",
            180
        ),
        create_question(
            "Q12: For inference, AWQ 4-bit model on A100 GPU vs fp16 - what is the throughput improvement for batch=32?",
            [
                "~4× - directly proportional to memory reduction",
                "~2-3× - limited by compute and kernel efficiency",
                "~1.2-1.5× - minimal improvement for large batches",
                "No improvement - same throughput, just less memory"
            ],
            1,
            "Senior Explanation: Large batch (32): Compute-bound (not memory-bound). AWQ 4-bit weights dequantized to fp16, matmuls in fp16 (Tensor Cores). Throughput gain from: (1) Larger effective batch fits in VRAM (4× less model memory), (2) Dequantization overhead ~10-20%. For batch=32, fp16 LLaMA-7B: Model 14GB + activations 10GB = 24GB (needs 40GB for KV cache). AWQ: Model 3.5GB + activations 10GB = 13.5GB (can fit batch=64 in 40GB). Throughput: batch=32 → ~2× (better GPU utilization). batch=64 (only possible with AWQ) → ~3× vs fp16 batch=32. Option A assumes memory-bound (true for batch=1). Production: AWQ's main benefit is ENABLING larger batches, not faster per-sample. Trade-off: Latency (batch=1) improvement ~1.5-2×, throughput (batch=32+) ~2-3×.",
            "Hard",
            220
        ),

        # INT8/INT4 QUANTIZATION (Questions 13-16)
        create_question(
            "Q13: LLM.int8() (8-bit quantization) handles outlier features differently. What is the approach?",
            [
                "Removes outlier features before quantization",
                "Uses mixed-precision - keeps ~0.1% of features (outliers) in fp16, quantizes rest to int8",
                "Clips outliers to reduce range",
                "Uses higher bit-width (int16) for outliers"
            ],
            1,
            "Senior Explanation: LLM.int8() observation: ~0.1-0.5% of features have magnitude >6σ (outliers), causing huge quantization error if quantized naively. Solution: Detect outliers (threshold = 6.0), keep in fp16, quantize rest to int8. Matmul: Split into int8 matmul (99.5% of weights) + fp16 matmul (0.5%). Memory: Mostly int8 (2× reduction) with tiny fp16 overhead. Quality: Near-zero degradation (<0.1% perplexity). Compute: int8 matmul fast (Tensor Cores), fp16 matmul small (negligible overhead). Option C (clipping) causes accuracy loss. Production: bitsandbytes library implements LLM.int8(). Used for inference and QLoRA training. Trade-off: Slight complexity (mixed precision) for maintaining quality.",
            "Hard",
            220
        ),
        create_question(
            "Q14: For weight-only quantization (weights in int4, activations in fp16), what is the inference speedup determinant?",
            [
                "Compute speed - int4 matmuls faster",
                "Memory bandwidth - loading 4× less weight data from HBM to compute units",
                "Batch size - only matters for large batches",
                "GPU type - only newer GPUs benefit"
            ],
            1,
            "Senior Explanation: Weight-only quantization: Weights stored int4 in HBM, loaded to GPU, dequantized to fp16, matmul in fp16. Bottleneck: Memory bandwidth (loading weights from HBM). For batch=1, seq=1 (single token generation): Compute = O(H²), memory transfer = O(H²) for weights. Weight-only reduces memory transfer 4× → ~2-3× speedup (not 4× due to dequantization overhead). For large batch: Compute O(B × H²) dominates, memory O(H²) (weights loaded once, reused) → minimal speedup (1.1-1.5×). Option A wrong - matmul in fp16, not int4. Production: Weight-only quantization best for low-batch / latency-critical serving. For high-throughput (batch=32+), need activation quantization too. Trade-off: Simple (weights only) but limited speedup for large batches.",
            "Hard",
            220
        ),
        create_question(
            "Q15: Dynamic quantization vs static quantization for activations - what is the key difference?",
            [
                "Dynamic computes quantization params (scale/zero-point) per-batch at runtime, static uses pre-calibrated constants",
                "Dynamic quantizes during training, static post-training",
                "Dynamic uses different bit-widths, static fixed",
                "No difference - same approach"
            ],
            0,
            "Senior Explanation: Static: Calibration phase collects activation ranges (min, max) for each layer, computes fixed scale/zero-point, stores them. Inference: Uses stored params for all inputs. Dynamic: Runtime computes scale/zero-point for each batch's activations. Benefits: (1) Static faster (no computation overhead), (2) Dynamic more accurate (adapts to input distribution). For LLMs: Activations vary widely by input → dynamic preferred. Overhead: Computing min/max + scale ≈ 1-5% latency increase. Quality: Dynamic ~0.5-1% better than static. Option B confuses with QAT. Production: PyTorch dynamic quantization for NLP models (bert, gpt), static for CV (more stable activations). Trade-off: Dynamic flexibility vs static speed.",
            "Hard",
            200
        ),
        create_question(
            "Q16: For int4 weight quantization, what is the theoretical memory reduction for a 7B model vs fp16?",
            [
                "2× - int4 is half of int8",
                "4× - int4 is quarter of fp16 (16-bit)",
                "~3.5-3.8× accounting for quantization overhead (scales, zero-points)",
                "8× - aggressive compression"
            ],
            2,
            "Senior Explanation: FP16: 7B params × 2 bytes = 14GB. Int4: 7B × 0.5 bytes = 3.5GB. Overhead: Scales + zero-points (fp16) for groups. Group=128: 7B/128 groups × 4 bytes (scale+zero in fp16) = 218MB. Total: 3.5GB + 0.22GB = 3.72GB. Reduction: 14GB / 3.72GB ≈ 3.76×. Option B assumes zero overhead (not realistic). Smaller groups (64): More overhead, ~3.5× reduction. Option A/D wrong. Production: Actual deployment sees ~3.5-3.8× memory reduction. Enables: 7B model (fp16: 14GB) → int4: ~4GB, fits on RTX 3080 (10GB) with room for KV cache. Trade-off: Quantization overhead (scales) small (~5%) but non-negligible for memory planning.",
            "Medium",
            180
        ),

        # DYNAMIC VS STATIC & MIXED PRECISION (Questions 17-20)
        create_question(
            "Q17: For serving a quantized LLM, you observe accuracy degradation for certain prompts. What is the likely cause and fix?",
            [
                "Quantization is fundamentally broken - revert to fp16",
                "Activation outliers for specific inputs - use dynamic quantization or mixed precision (LLM.int8() approach)",
                "Model was poorly quantized - re-run calibration",
                "GPU doesn't support quantized ops properly"
            ],
            1,
            "Senior Explanation: Input-dependent degradation suggests activation outliers. Some prompts trigger extreme activations (magnitude >>typical), causing int8 overflow or large quantization error. Solution: (1) Dynamic quantization (adapts per input), (2) Mixed precision (detect outliers, use fp16 for them), (3) Per-token quantization (instead of per-tensor). Debugging: Log activation ranges per prompt. If max/min vary 10×+ across prompts, outliers present. Option C - re-calibration helps only if calibration set unrepresentative. Production: LLM.int8() specifically designed to handle this (outlier features in fp16). Trade-off: Mixed precision adds complexity but maintains quality for outlier-heavy inputs.",
            "Hard",
            200
        ),
        create_question(
            "Q18: Quantization-aware training (QAT) vs post-training quantization (PTQ) - when is QAT necessary?",
            [
                "Always - QAT always better than PTQ",
                "For aggressive quantization (2-3 bit) or when PTQ degrades quality >5%",
                "For large models only (7B+)",
                "Never - PTQ sufficient for all cases"
            ],
            1,
            "Senior Explanation: PTQ (GPTQ, AWQ) works well for 4-bit+ (typically <1% degradation). For 2-3 bit, PTQ degrades 5-15% → QAT needed. QAT: Train with fake quantization (simulate int4 in fp32), learns to be robust to quantization. Benefit: ~3-5% better quality than PTQ at 3-bit. Cost: Requires training (data, compute, days/weeks). For 4-bit, PTQ sufficient (QAT improves only 0.1-0.3%). Option A too strong - QAT expensive, only use when necessary. Production: 4-bit PTQ standard. 3-bit PTQ for less critical tasks. 2-bit requires QAT or significant quality loss. Trade-off: PTQ fast and easy (hours) vs QAT slow and complex (days) but higher quality at low bits.",
            "Hard",
            220
        ),
        create_question(
            "Q19: For a 175B model (GPT-3 scale), what is the minimum VRAM needed for 4-bit inference with batch=1, seq=2048?",
            [
                "~40 GB - model weights dominate",
                "~80 GB - model + KV cache",
                "~150 GB - model + KV cache + activations",
                "~200 GB - needs multi-GPU"
            ],
            1,
            "Senior Explanation: 175B model 4-bit: 175B × 0.5 bytes ≈ 87.5GB (with overhead ~90GB). KV cache (batch=1, seq=2048, 96 layers, H=12288): ~20GB. Activations (batch=1): ~5-10GB. Total: 90 + 20 + 10 ≈ 120GB. Option B reasonable (80GB tight, may OOM). Single A100 (80GB): Can't fit. 2× A100: Fits. Single H100 (80GB): Tight, need optimizations (Flash Attention, offloading). Option A underestimates KV cache. Production: 175B 4-bit needs 2× 80GB GPUs minimum, or 1× H100 with optimizations. For batch>1 or seq>2048, need more GPUs. 8-bit would need ~2× (160GB). Trade-off: 4-bit enables serving large models on fewer GPUs (cost savings), but still requires high-end hardware.",
            "Hard",
            240
        ),
        create_question(
            "Q20: What is 'GPTQ-for-LLaMA' vs 'AutoGPTQ' - what is the difference?",
            [
                "Different quantization algorithms - GPTQ-for-LLaMA uses unique approach",
                "Same algorithm (GPTQ), different implementations - GPTQ-for-LLaMA for LLaMA only, AutoGPTQ general-purpose library",
                "GPTQ-for-LLaMA is research code, AutoGPTQ is production",
                "AutoGPTQ is newer, improved algorithm"
            ],
            1,
            "Senior Explanation: Both implement GPTQ algorithm. GPTQ-for-LLaMA: Original community implementation (qwopqwop200/GPTQ-for-LLaMA), supports LLaMA/LLaMA-2. Less maintained. AutoGPTQ: General library (AutoGPTQ/AutoGPTQ), supports many models (LLaMA, GPT-J, OPT, BLOOM), actively maintained, easier API, integrates with Transformers. Both produce similar quality (same algorithm). AutoGPTQ preferred for new projects. Production: AutoGPTQ standard choice. Integrates with Hugging Face (load quantized models with from_pretrained). GPTQ-for-LLaMA historical importance but superseded. Trade-off: AutoGPTQ more dependencies and complexity, but better ecosystem integration. For research/experimentation, GPTQ-for-LLaMA sufficient.",
            "Medium",
            180
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_quantization()
    db.add_questions("Senior Quantization - GPTQ, AWQ, INT8/INT4", questions)
    print(f"✓ Successfully added {len(questions)} senior Quantization questions!")
    print(f"✓ Category: Senior Quantization - GPTQ, AWQ, INT8/INT4")
