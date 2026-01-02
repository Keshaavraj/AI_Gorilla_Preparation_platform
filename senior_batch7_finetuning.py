"""
Senior AI Engineer Interview Questions - Batch 7: Fine-Tuning Techniques
Topics: LoRA, QLoRA, Prefix Tuning, Prompt Tuning, PEFT Methods
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_finetuning():
    """20 Senior-Level Fine-Tuning Questions"""
    questions = [
        # LORA (LOW-RANK ADAPTATION) (Questions 1-7)
        create_question(
            "Q1: LoRA decomposes weight updates ΔW into low-rank matrices A and B. For a weight matrix W of size 4096×4096 with LoRA rank r=8, what is the parameter reduction?",
            [
                "~512× reduction - (4096×8 + 8×4096) / (4096×4096) ≈ 0.4%",
                "~8× reduction - rank determines reduction factor",
                "~50% reduction - half the parameters",
                "No reduction - LoRA adds parameters on top of base model"
            ],
            0,
            "Senior Explanation: Full fine-tuning updates W (4096×4096 = 16.8M params). LoRA: W' = W + BA where B is 4096×r, A is r×4096. Trainable params: 4096×8 + 8×4096 = 65,536. Reduction: 16.8M / 65.5K ≈ 256×. For r=8, typical reduction is 100-1000× depending on original matrix size. For 7B model with LoRA on all attention matrices: Full fine-tune ~7B params, LoRA ~4-8M params (~1000× reduction). Memory: Base model frozen (no optimizer states), only LoRA weights need Adam states. Storage: LoRA checkpoint ~10-30MB vs full model ~14GB (fp16). Option B 'junior trap' - rank doesn't directly equal reduction factor. Production: Fine-tune LLaMA-7B on single GPU (24GB) with LoRA, impossible with full fine-tuning (needs 200GB+ for optimizer states).",
            "Hard",
            220
        ),
        create_question(
            "Q2: In LoRA, why are weight updates decomposed as ΔW = BA instead of directly learning a low-rank ΔW?",
            [
                "BA decomposition is faster to compute",
                "Enables scaling: ΔW can be scaled by α/r where α is hyperparameter, making it easy to adjust LoRA strength",
                "BA uses less memory than full ΔW",
                "BA is mathematically proven to converge faster"
            ],
            1,
            "Senior Explanation: LoRA uses scaling factor α/r: W' = W + (α/r)BA. This allows tuning LoRA contribution strength without retraining. Typical: α=16, r=8 → scale by 2×. Higher α → stronger adaptation. During inference, LoRA weights can be merged: W_merged = W + (α/r)BA (single matrix, no inference overhead). Option A wrong - BA requires two matmuls in forward. Option C - both ΔW and BA have similar memory. Production: α is key hyperparameter. For domain adaptation: α=8-16 (mild adaptation). For task-specific fine-tuning: α=32-64 (strong adaptation). Tuning α is faster than retraining with different rank. Trade-off: Need to choose α upfront; changing α after training requires recomputing BA scaling. Code: lora_weight = (alpha / r) * (B @ A).",
            "Hard",
            200
        ),
        create_question(
            "Q3: You apply LoRA to a Transformer. Which matrices should you apply LoRA to for best performance?",
            [
                "Only query and value matrices in attention - most important for adaptation",
                "All four attention matrices (Q, K, V, O) - comprehensive adaptation",
                "Q, V matrices + FFN layers - balances params and performance",
                "Only output projection - minimizes parameters"
            ],
            1,
            "Senior Explanation: Empirical studies show applying LoRA to ALL four attention matrices (Q, K, V, output projection) gives best results, with minimal param increase (4× more LoRA params but still <<1% of model). Ablation: Q+V only: 80-90% of full LoRA performance. Q+V+FFN: 95%+. All attention+FFN: 98-100%. For 7B model: LoRA on all attention (~8M params), LoRA on all attention+FFN (~16M params). Option A common misconception from original LoRA paper (used Q+V only as example). Option D too limited. Production: For production fine-tuning, use all attention matrices at minimum. Add FFN if parameter budget allows (~2× LoRA params). Trade-off: More matrices → more params but better adaptation. Typical: r=8 for Q,K,V,O is ~10-20M params for 7B model (0.3% of total).",
            "Hard",
            220
        ),
        create_question(
            "Q4: During LoRA training, the base model weights are frozen. What is the memory advantage for optimizer states?",
            [
                "No advantage - still need optimizer states for all parameters",
                "~2/3 memory saving - only LoRA parameters have optimizer states (Adam: 2× params for momentum+variance)",
                "~99% saving - base model (7B params) frozen, only LoRA (8M params) needs optimizer states",
                "50% saving - half the parameters don't need gradients"
            ],
            2,
            "Senior Explanation: Adam optimizer stores 2× trainable params (momentum + variance). Full fine-tuning: 7B trainable → 14B optimizer states. LoRA: 8M trainable → 16M optimizer states. Saving: (14B - 16M) / 14B ≈ 99.9%. Memory breakdown: Base model 7B (fp16) = 14GB, LoRA params 8M = 16MB, optimizer states 16M = 32MB, gradients 8M = 16MB. Total: ~14.1GB vs full fine-tune ~42GB (14GB model + 14GB gradients + 28GB optimizer). Option B 'junior trap' - confusing fraction of trainable params with memory. Production: Enables fine-tuning 7B models on 24GB consumer GPUs (RTX 3090/4090). Full fine-tuning needs 80GB A100. Trade-off: Memory savings allow larger batch sizes (better gradient stability) on same hardware.",
            "Hard",
            220
        ),
        create_question(
            "Q5: You train LoRA adapters for 10 different tasks. For inference, what is the overhead of swapping between tasks?",
            [
                "~5-10s - need to reload entire model",
                "~50-200ms - load LoRA weights (~10-30MB) from disk and merge with base model",
                "Negligible (<1ms) - LoRA weights kept in memory, just switch which adapter is active",
                "~1-2s - requires recompiling computation graph"
            ],
            2,
            "Senior Explanation: Multi-task serving: Load base model once (14GB), keep all LoRA adapters in memory (10 tasks × 20MB = 200MB). Switching: Change which LoRA adapter is added to base weights. No disk I/O, no reloading. Overhead: <1ms (pointer switch). Alternative: Merge LoRA into base for each task (W_task = W_base + BA), pre-compute all 10 versions. Memory: 10 × 14GB = 140GB (infeasible). Better: Dynamic merging during forward pass (add BA @ x to output). Overhead: ~5-10% (extra matmul). Option B describes disk loading. Production: Serve 100s of fine-tuned models on single GPU by sharing base model. Example: ChatGPT potentially uses adapter-style approach for different behavior modes. Trade-off: Keeping all adapters in memory (200MB total) vs disk loading (200ms per swap) vs merged models (10× memory).",
            "Hard",
            200
        ),
        create_question(
            "Q6: LoRA rank r is a key hyperparameter. For a 7B model, what rank is typically used?",
            [
                "r=1-2 - minimal parameters for efficiency",
                "r=8-16 - balances performance and parameter efficiency",
                "r=64-128 - high rank for better expressiveness",
                "r=512+ - approach full-rank for best quality"
            ],
            1,
            "Senior Explanation: Typical ranks: r=8 for simple tasks (classification, entity extraction), r=16-32 for complex tasks (instruction tuning, domain adaptation), r=64+ rarely used (diminishing returns). Empirical: r=8 achieves 90-95% of full fine-tuning performance. r=16: 95-98%. r=32: 98-99%. r=64: 99%+ but 8× more LoRA params than r=8. Option A too low (underfit). Option C/D wasteful (diminishing returns, defeats LoRA purpose). Production: LLaMA fine-tuning usually r=8-16. GPT-3.5 fine-tuning (via API) likely uses similar low ranks. Trade-off: Higher rank → better quality but more memory, slower training, larger checkpoint. For most tasks, r=8-16 optimal. Hyperparameter search: Try r=8,16,32 on validation set.",
            "Medium",
            180
        ),
        create_question(
            "Q7: Can LoRA adapters be merged with the base model for inference? What is the advantage?",
            [
                "No - LoRA must be computed dynamically during forward pass",
                "Yes - compute W' = W + BA offline, then inference uses W' with zero overhead",
                "Yes but slower - merging adds latency",
                "Only for specific architectures - not general"
            ],
            1,
            "Senior Explanation: LoRA can be merged: W_merged = W_base + (α/r) × BA. Compute offline (one-time cost ~1-5s for 7B model), then use W_merged for inference. Inference: Zero overhead vs base model (same compute, same latency). Memory: Same as base model (14GB for 7B fp16). Unmerging: Not needed typically, but theoretically possible if original W_base and BA stored. Option A 'junior trap' - dynamic computation possible but unnecessary. Production: Deployed LoRA models often merged for simplicity (single weight file, standard inference code). Un-merged useful for: (1) Multi-task serving (swap adapters), (2) Experimentation (adjust α without retraining). Trade-off: Merged = simple deployment, unmerged = flexibility for multi-task. Code: merged_weight = base_weight + lora_scale * (lora_B @ lora_A).",
            "Medium",
            180
        ),

        # QLORA (QUANTIZED LORA) (Questions 8-13)
        create_question(
            "Q8: QLoRA quantizes base model to 4-bit. For a 7B parameter model, what is the memory reduction vs fp16?",
            [
                "2× reduction - 4-bit vs 8-bit",
                "4× reduction - 4-bit vs 16-bit (fp16)",
                "~3.5× reduction accounting for quantization overhead (scales, zero-points)",
                "8× reduction - aggressive compression"
            ],
            2,
            "Senior Explanation: FP16: 7B × 2 bytes = 14GB. 4-bit: 7B × 0.5 bytes = 3.5GB. Overhead: Quantization scales/zero-points per group (e.g., 64-element groups) add ~2-5% (typically use fp16 for these). Total: ~3.5GB + 5% ≈ 3.7GB. Reduction: 14GB / 3.7GB ≈ 3.8×. Option B assumes perfect 4× (ignores overhead). Option A/D wrong calculations. Additional memory: LoRA adapters (16-32MB fp16/bf16), optimizer states for LoRA only (~32-64MB), activations/gradients (~2-4GB for batch=4). Total QLoRA training: ~8-10GB vs full fp16 fine-tuning ~42GB. Production: QLoRA enables fine-tuning 7B models on consumer GPUs (RTX 3090 24GB, even RTX 3080 12GB with small batch). Trade-off: 4-bit quantization causes ~0.5-1% performance degradation vs fp16, but enables training otherwise impossible.",
            "Hard",
            220
        ),
        create_question(
            "Q9: QLoRA uses NormalFloat4 (NF4) data type instead of standard 4-bit integers. What is the key advantage?",
            [
                "Faster computation - NF4 optimized for GPUs",
                "Information-theoretically optimal for normally distributed weights - assigns more precision to common values near zero",
                "Uses less memory than standard 4-bit",
                "Better numerical stability - prevents overflow"
            ],
            1,
            "Senior Explanation: Neural network weights typically follow normal distribution N(0, σ). NF4 assigns quantization levels such that each bin has equal probability under normal distribution (optimal rate-distortion for Gaussian data). More levels near zero (high density), fewer in tails. Standard uniform 4-bit: Equal spacing (e.g., -8 to +7). Wastes precision in tails. NF4: ~0.3-0.5% better perplexity than uniform 4-bit. Option A wrong - NF4 uses same compute as int4 (lookup + dequantize). Option C wrong - same 4 bits. Production: QLoRA paper introduced NF4, now standard for 4-bit quantization. Implementation: Pre-computed lookup table of 16 NF4 values, quantize via nearest value. Dequantize to fp16/bf16 for computation. Trade-off: Minimal implementation complexity for measurable quality improvement.",
            "Hard",
            200
        ),
        create_question(
            "Q10: In QLoRA, LoRA adapters are trained in what precision?",
            [
                "4-bit - matches base model quantization",
                "8-bit - balances efficiency and quality",
                "16-bit (bf16/fp16) - full precision for trainable parameters",
                "Mixed - gradients in fp16, weights in 4-bit"
            ],
            2,
            "Senior Explanation: QLoRA: Base model frozen in 4-bit, LoRA adapters trained in bf16/fp16. During forward: (1) Dequantize 4-bit base weights to bf16, (2) Compute base model output, (3) Add LoRA contribution (BA @ x) in bf16. Backward: Gradients computed in bf16 for LoRA only (base frozen). This maintains training stability - 4-bit insufficient for gradient accumulation (too coarse for small updates). Memory: LoRA in bf16 adds ~30MB (negligible vs 3.5GB base). Option A would cause training instability. Option D partially correct but LoRA weights themselves are bf16. Production: bitsandbytes library (QLoRA implementation) uses this exact setup. Quality: QLoRA achieves 99%+ of full fp16 fine-tuning performance. Trade-off: Tiny memory increase (30MB) for stable training.",
            "Hard",
            200
        ),
        create_question(
            "Q11: QLoRA uses 'double quantization'. What does this mean?",
            [
                "Quantize both weights and activations",
                "Quantize the quantization parameters (scales/zero-points) themselves to save memory",
                "Perform quantization twice for better accuracy",
                "Use 4-bit for weights, 8-bit for gradients"
            ],
            1,
            "Senior Explanation: Standard quantization stores fp16 scales (one per group of 64 weights). For 7B params with 64-element groups: 7B/64 = 109M scales × 2 bytes = 218MB. Double quantization: Quantize scales to 8-bit → 109M × 1 byte = 109MB (50% saving on scales). Nested quantization: Each group of 256 scales has one fp16 'super-scale' + 256 8-bit scales. Memory saved: ~100-150MB (2-3% of total). Negligible compute overhead (one extra dequantization step). Option A describes activation quantization (separate concept). Production: QLoRA uses double quantization by default in bitsandbytes. Contribution to overall savings: Minor (~100MB), but authors found it necessary to fit 65B models on 48GB GPUs. Trade-off: Tiny complexity increase for 100MB savings (can be difference between OOM and success).",
            "Hard",
            220
        ),
        create_question(
            "Q12: For QLoRA training, activations are computed in what precision?",
            [
                "4-bit - matches base model to save memory",
                "8-bit - compressed but sufficient for forward pass",
                "bf16 - full precision for numerical stability during training",
                "Mixed precision - critical layers in fp32"
            ],
            2,
            "Senior Explanation: QLoRA computes activations in bf16/fp16. Process: 4-bit weights dequantized to bf16 → matmul with bf16 activations → bf16 output. Keeping activations in bf16 essential for: (1) Gradient computation (backward pass needs high precision), (2) Numerical stability (small activation values important). Memory: Activations dominate for large batch/sequence. For batch=4, seq=512, 7B model: ~3-4GB activations (bf16). If quantized to 4-bit: ~1GB but training fails (unstable gradients). Option A 'junior trap' - 4-bit activations cause severe degradation. Production: Activation quantization possible for INFERENCE (PTQ, QAT) with careful calibration, but not standard for training. Trade-off: Activation memory (3-4GB) is trade-off for stable training. Reduce via gradient checkpointing (recompute activations, save memory).",
            "Medium",
            180
        ),
        create_question(
            "Q13: You want to fine-tune a 13B model with QLoRA on a 24GB GPU. What batch size and sequence length are feasible?",
            [
                "Batch=16, seq=2048 - standard training setup",
                "Batch=4, seq=512 - memory-constrained but feasible",
                "Batch=1, seq=128 - extremely limited",
                "Batch=8, seq=1024 - balanced"
            ],
            1,
            "Senior Explanation: 13B model 4-bit: ~6.5GB. LoRA params (16M): ~32MB. Optimizer states: ~64MB. Activations (batch=4, seq=512, 40 layers): ~4-6GB. Gradients: ~1-2GB. Total: ~12-15GB (fits in 24GB). Batch=8 or seq=1024: Activations double → ~20-22GB (tight, may OOM). Option A requires ~40GB+. Option C too conservative (could use larger). Production: Typical QLoRA on consumer GPUs: batch=1-4, seq=512-1024 with gradient accumulation (effective batch 16-32). Techniques to increase capacity: (1) Gradient checkpointing (saves 50% activation memory, ~30% slower), (2) Flash Attention (saves 30-50% attention memory). Trade-off: Small batch (1-4) → noisy gradients, use gradient accumulation. 4 steps × batch=4 = effective batch 16 (stable training).",
            "Hard",
            220
        ),

        # PREFIX TUNING & PROMPT TUNING (Questions 14-17)
        create_question(
            "Q14: Prefix Tuning prepends trainable embeddings (prefix) to the input. For a 7B model with prefix_length=20, how many trainable parameters?",
            [
                "~10M - comparable to LoRA",
                "~100K - prefix only (20 × embedding_dim)",
                "~60M - prefix for all layers (20 × hidden_dim × num_layers × 2 for K,V)",
                "~1M - prefix and projection layers"
            ],
            2,
            "Senior Explanation: Prefix tuning adds trainable prefix for K,V in each layer. For LLaMA-7B (32 layers, hidden_dim=4096): Prefix params = prefix_length × hidden_dim × num_layers × 2 (K and V) = 20 × 4096 × 32 × 2 = 5.24M params. Some implementations add reparameterization MLP (smaller prefix projected to hidden_dim): ~2× params ≈ 10M. Option B 'junior trap' - forgets prefix replicated per layer. Option A/D close but depends on reparameterization. Production: Prefix tuning typically 5-10M params (0.1-0.2% of 7B model), similar to LoRA but different mechanism. Trade-off: Prefix tuning modifies attention directly (more disruptive), LoRA modifies weight matrices (more general). Quality: Comparable to LoRA for many tasks, sometimes better for generation tasks.",
            "Hard",
            200
        ),
        create_question(
            "Q15: Prompt Tuning (soft prompts) vs Prefix Tuning - what is the key difference?",
            [
                "Prompt tuning is for classification, prefix tuning for generation",
                "Prompt tuning adds trainable embeddings only at input layer, prefix tuning adds to all layers",
                "Prompt tuning uses discrete tokens, prefix tuning uses continuous vectors",
                "No difference - same technique with different names"
            ],
            1,
            "Senior Explanation: Prompt tuning: Adds trainable embeddings to INPUT only (e.g., prepend 20 tokens to input sequence). Params: prefix_length × embedding_dim = 20 × 4096 = 81,920 (~80K). Prefix tuning: Adds trainable K,V to EVERY layer's attention. Params: ~5-10M. Quality: Prefix tuning generally better (modifies all layers), prompt tuning simpler (fewer params). Option C confuses with hard prompt engineering (discrete tokens). Production: Prompt tuning simpler to implement (just add to input embeddings), prefix tuning more powerful. For T5, prompt tuning with length=100 achieves good results. For GPT-style models, prefix tuning preferred. Trade-off: Prompt tuning 100× fewer params but ~5-10% worse performance than prefix tuning.",
            "Hard",
            200
        ),
        create_question(
            "Q16: For multi-task learning, you train separate prefix adapters for 20 tasks on a 7B model. What is the total parameter overhead?",
            [
                "~10M - shared prefix across tasks",
                "~100M - 20 tasks × 5M params/task",
                "~200M - includes task-specific heads",
                "~1B - separate adapters are expensive"
            ],
            1,
            "Senior Explanation: Each task has independent prefix: 20 tasks × 5M params = 100M total. Base model (7B) shared. Storage: 100M × 2 bytes (fp16) = 200MB total (~14MB per task). Compare to 20 fully fine-tuned models: 20 × 14GB = 280GB. Savings: 280GB / 200MB = 1400×. Memory at runtime: Base model (14GB) + active prefix (10MB) = 14.01GB. Can load all 20 prefixes in memory (200MB) and switch instantly. Option A assumes shared (defeats multi-task purpose). Production: Multi-task serving with prefix/LoRA adapters standard for scalable deployment. Example: Serve 100 specialized models on single GPU. Trade-off: Slight quality loss vs full fine-tuning (3-5%) for massive efficiency.",
            "Medium",
            180
        ),
        create_question(
            "Q17: Prefix tuning often uses a reparameterization MLP. Why?",
            [
                "Reduces number of trainable parameters",
                "Smaller prefix (e.g., 512-dim) projected to hidden_dim (4096-dim) improves optimization and prevents overfitting",
                "Faster inference - MLP can be pre-computed",
                "Required for compatibility with attention mechanism"
            ],
            1,
            "Senior Explanation: Reparameterization: Learn small prefix (e.g., 20 × 512) → MLP projects to (20 × 4096) used as actual K,V prefix. Trainable params: 20×512 (prefix) + 512×4096 (MLP projection) ≈ 2M per layer. Benefits: (1) Lower-dimensional optimization space (easier to train), (2) Regularization (bottleneck prevents overfitting). After training: Can discard MLP and use projected prefix only (inference speedup). Option A wrong - reparameterization adds MLP params (more not fewer). Option C - MLP removed post-training (baked into prefix). Production: Most prefix tuning implementations use reparameterization with bottleneck_dim=512-1024. Trade-off: Training complexity (MLP) for better convergence and final quality.",
            "Hard",
            200
        ),

        # FULL FINE-TUNING VS PEFT (Questions 18-20)
        create_question(
            "Q18: For instruction tuning a 7B model, LoRA vs full fine-tuning - what is the quality gap?",
            [
                "~10-15% degradation - LoRA significantly worse",
                "~1-3% degradation - LoRA nearly matches full fine-tuning",
                "No degradation - LoRA equals or exceeds full fine-tuning",
                "~20-30% degradation - LoRA only for simple tasks"
            ],
            1,
            "Senior Explanation: Empirical results (LLaMA, GPT-3 fine-tuning): LoRA (r=16-32) achieves 97-99% of full fine-tuning performance on instruction following, summarization, QA. Gap: 1-3% absolute (e.g., full fine-tune 85% accuracy, LoRA 82-84%). For some tasks (classification with few classes), LoRA matches or exceeds full fine-tuning (regularization effect from low rank). Option A/D overstate gap. Option C overstates - usually slight degradation. Production: Most commercial LLM fine-tuning (OpenAI, Anthropic likely) uses adapter methods due to cost/efficiency. Quality gap acceptable for most applications. Trade-off: 1-3% quality for 100× faster training, 1000× smaller checkpoints, multi-task serving capability.",
            "Medium",
            180
        ),
        create_question(
            "Q19: You need to fine-tune for a privacy-sensitive task where data cannot leave on-premise servers. Which method is most practical?",
            [
                "Full fine-tuning - ensures best quality",
                "LoRA/QLoRA - enables fine-tuning on consumer GPUs available on-premise",
                "Prompt engineering - no fine-tuning needed",
                "API-based fine-tuning - most secure"
            ],
            1,
            "Senior Explanation: On-premise typically has limited compute (few GPUs, consumer-grade). QLoRA enables fine-tuning 7B-13B models on single RTX 4090 (24GB). Full fine-tuning needs 80GB A100 (expensive, rare on-premise). Option C (prompt engineering) may not achieve task performance requirements. Option D contradicts privacy constraint (data leaves premise). Production scenario: Healthcare/finance fine-tuning on proprietary data. Solution: QLoRA on-premise with 2-4× RTX 4090 GPUs. Cost: ~$8K hardware vs $200K+ for 8× A100 cluster. Trade-off: QLoRA slight quality reduction (~1-2%) acceptable for privacy/cost constraints. Alternative: Differential privacy + cloud fine-tuning (complex, not widely adopted).",
            "Medium",
            180
        ),
        create_question(
            "Q20: For catastrophic forgetting (model forgets original capabilities after fine-tuning), which PEFT method helps most?",
            [
                "Full fine-tuning with regularization",
                "LoRA - base model frozen, preserves original weights and capabilities",
                "Prompt tuning - modifies input only",
                "All methods equally suffer from catastrophic forgetting"
            ],
            1,
            "Senior Explanation: LoRA keeps base model FROZEN - original capabilities fully preserved. Fine-tuned behavior comes from LoRA adapter. Can even remove adapter to recover original model. Full fine-tuning: Modifies all weights → overwrites original knowledge (e.g., fine-tune on code → forgets language understanding). Prompt/prefix tuning: Also freeze base, preserve capabilities. Option A can mitigate (e.g., elastic weight consolidation) but doesn't eliminate forgetting. Production: LoRA/adapters enable fine-tuning without catastrophic forgetting - critical for continual learning, multi-task models. Example: Fine-tune GPT-3 for 100 specialized tasks, each with adapter, base model unchanged. Trade-off: Adapters slightly less performant (1-3%) but preserve original capabilities. For high-stakes deployment, preservation crucial.",
            "Hard",
            200
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_finetuning()
    db.add_questions("Senior Fine-Tuning - LoRA & PEFT Methods", questions)
    print(f"✓ Successfully added {len(questions)} senior Fine-Tuning questions!")
    print(f"✓ Category: Senior Fine-Tuning - LoRA & PEFT Methods")
