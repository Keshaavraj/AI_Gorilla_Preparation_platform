"""
Senior AI Engineer Interview Questions - Batch 6: Advanced Deep Learning Architecture
Topics: Normalization (LayerNorm, BatchNorm, RMSNorm), Activations, Residual Connections, Initialization, Efficient Architectures
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_deep_learning():
    """20 Senior-Level Deep Learning Architecture Questions"""
    questions = [
        # NORMALIZATION TECHNIQUES (Questions 1-5)
        create_question(
            "Q1: For a Transformer model with batch size B=32, sequence length L=512, hidden dim H=768, what is computed during LayerNorm?",
            [
                "Normalize across batch dimension - mean/var computed over all 32 samples at each position",
                "Normalize across hidden dimension - mean/var computed over 768 features per token independently",
                "Normalize across sequence dimension - mean/var over all 512 tokens per sample",
                "Global normalization - mean/var over entire B×L×H tensor"
            ],
            1,
            "Senior Explanation: LayerNorm computes mean and variance across the FEATURE dimension (last dimension H=768) for each token independently. For one token: mean = sum(x_i) / H, var = sum((x_i - mean)²) / H, then normalize: y_i = (x_i - mean) / sqrt(var + eps). Shape: Input (B, L, H), output (B, L, H) with B×L independent normalization operations. Option A describes BatchNorm. Option C would be non-standard (not used). Production: LayerNorm is standard in Transformers (BERT, GPT, T5) because it works with variable sequence lengths and doesn't depend on batch statistics (good for inference with batch=1). BatchNorm fails for batch=1 (no statistics). Trade-off: LayerNorm slightly more compute than BatchNorm (per-token stats) but essential for sequence models.",
            "Hard",
            200
        ),
        create_question(
            "Q2: You're training a CNN with BatchNorm (num_features=256). During evaluation with batch_size=1, how are normalization statistics computed?",
            [
                "Computed from the single sample - mean/var of the one image",
                "Use running statistics accumulated during training - moving averages of mean/var",
                "BatchNorm disabled during evaluation",
                "Use global dataset statistics computed offline"
            ],
            1,
            "Senior Explanation: During training, BatchNorm maintains running_mean and running_var using exponential moving average (EMA): running_mean = momentum × running_mean + (1-momentum) × batch_mean. Typical momentum=0.9. During eval (model.eval()), BatchNorm uses these running stats for normalization (no batch stats computed). This allows batch_size=1 inference. For batch=1, computing stats from single sample would be meaningless (var ≈ 0). Option A 'junior trap'. Option C wrong - still normalizes, just uses running stats. Production: Critical to call model.eval() for inference - using training mode with batch=1 causes poor results (unstable stats). Trade-off: Running stats are approximations (biased toward later training batches) but work for any batch size. For very different test distribution, may need to recompute running stats on validation set.",
            "Hard",
            200
        ),
        create_question(
            "Q3: RMSNorm (used in LLaMA) differs from LayerNorm by removing what?",
            [
                "The learnable scale and bias parameters",
                "The mean centering step - only normalizes by RMS (root mean square)",
                "The variance computation - uses approximate normalization",
                "The epsilon term - more numerically stable"
            ],
            1,
            "Senior Explanation: RMSNorm: y = x / sqrt(mean(x²) + eps) × scale. LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) × scale + bias. RMSNorm SKIPS mean centering (no - mean(x)) and bias term. Normalizes by RMS instead of standard deviation. Compute: RMSNorm ~30% faster (fewer ops: no mean subtraction, no variance computation). Accuracy: Comparable to LayerNorm (<0.5% degradation in perplexity). Option A wrong - RMSNorm still has learnable scale. Option C wrong - normalization is exact, not approximate. Production: LLaMA, GPT-NeoX, many recent LLMs use RMSNorm for efficiency. Trade-off: Small quality reduction for ~30% speedup in normalization (typically ~5-10% of total training time, so ~2-3% overall speedup). Memory: Same as LayerNorm.",
            "Hard",
            200
        ),
        create_question(
            "Q4: In a multi-GPU setup with BatchNorm and batch_size=16 per GPU (4 GPUs, effective batch=64), what batch statistics does BatchNorm use?",
            [
                "Per-GPU batch statistics (batch=16) - each GPU independently normalizes",
                "Global batch statistics (batch=64) - synchronized across GPUs via all-reduce",
                "Running statistics from previous iteration",
                "Mixed - use local stats during forward, sync during backward"
            ],
            0,
            "Senior Explanation: By default, BatchNorm computes stats PER-GPU (local batch=16), not synchronized. This causes issues: (1) High variance in stats (batch=16 vs batch=64), (2) Models behave differently on different GPU counts. Solution: SyncBatchNorm - computes mean/var across all GPUs via all-reduce. With SyncBatchNorm: Each GPU computes local sum, sum_squares → all-reduce to get global sum → compute global mean/var → normalize. Overhead: ~1-5ms per BatchNorm layer for all-reduce. Option A is default (problematic). Option B is SyncBatchNorm (correct practice). Production: Always use SyncBatchNorm for distributed training (PyTorch: nn.SyncBatchNorm, TF: sync_batch_norm=True). Impact: Accuracy improves 1-3% with SyncBatchNorm on small per-GPU batches. Trade-off: Slight communication overhead for better convergence.",
            "Hard",
            220
        ),
        create_question(
            "Q5: For online learning with streaming data (batch_size=1), which normalization is most suitable?",
            [
                "BatchNorm - standard choice for deep learning",
                "LayerNorm or GroupNorm - don't depend on batch statistics",
                "InstanceNorm - normalizes each instance independently",
                "No normalization - not necessary for online learning"
            ],
            1,
            "Senior Explanation: BatchNorm requires batch statistics (mean/var over batch dimension) - fails for batch=1. LayerNorm computes stats per sample over feature dimension (works with batch=1). GroupNorm divides features into groups, computes stats per group per sample (also works with batch=1). InstanceNorm normalizes per channel per sample (used in style transfer, also works with batch=1). Option A 'junior trap' - BatchNorm designed for batch>1. Option D wrong - normalization critical for training stability. Production: For online learning, reinforcement learning, or edge deployment (batch=1), use LayerNorm or GroupNorm. RNNs typically use LayerNorm. Trade-off: LayerNorm slightly different behavior than BatchNorm (trained with batch=32), but necessary for batch=1. For models requiring both batch and online inference, train with GroupNorm (works for any batch size).",
            "Medium",
            180
        ),

        # ACTIVATION FUNCTIONS & GRADIENT FLOW (Questions 6-9)
        create_question(
            "Q6: GELU activation (used in BERT, GPT) vs ReLU. What is the key difference in behavior?",
            [
                "GELU is faster - simpler computation than ReLU",
                "GELU is smooth and non-zero for negative inputs, better gradient flow than ReLU's hard threshold",
                "GELU prevents gradient vanishing completely",
                "GELU uses less memory by being in-place"
            ],
            1,
            "Senior Explanation: GELU(x) ≈ x × Φ(x) where Φ is Gaussian CDF. For x<0: GELU(x) is small but NON-ZERO (e.g., GELU(-1) ≈ -0.16), gradient flows. ReLU(x<0) = 0, gradient = 0 (dead neurons). GELU is SMOOTH (differentiable everywhere), ReLU has kink at 0. Benefits: (1) Better gradient flow, (2) Stochastic regularization effect (small negative inputs sometimes activate). Computation: GELU slower than ReLU (~2-3× due to erf/tanh approximation). Option A wrong - GELU slower. Option C overstates - improves but doesn't eliminate vanishing. Production: GELU standard in Transformers (GPT, BERT, T5). Empirically: ~0.5-1% accuracy improvement over ReLU. Approximation: GELU ≈ 0.5 × x × (1 + tanh(sqrt(2/π) × (x + 0.044715 × x³))) for faster compute. Trade-off: Slightly slower for better performance.",
            "Hard",
            200
        ),
        create_question(
            "Q7: Swish/SiLU activation (x × sigmoid(x)) is used in EfficientNet and modern CNNs. What is the main advantage?",
            [
                "Unbounded above (like ReLU) but smooth, enabling better optimization",
                "Faster than ReLU due to hardware optimization",
                "Uses less memory via in-place computation",
                "Prevents overfitting through built-in regularization"
            ],
            0,
            "Senior Explanation: Swish(x) = x × σ(x). For x>0: Swish(x) ≈ x (unbounded like ReLU, preventing saturation). For x<0: Swish(x) → 0 smoothly (not hard cutoff). Gradient: dSwish/dx = Swish(x) + σ(x) × (1 - Swish(x)) - always defined, never exactly zero. Benefits: (1) Smooth (better optimization landscape), (2) Non-monotonic (slight dip near x=0 acts as regularization), (3) Self-gating (sigmoid term gates x). Performance: ~0.5-2% accuracy improvement over ReLU on ImageNet. Computation: Slower than ReLU (~3-5× due to sigmoid), but benefits outweigh cost. Option B wrong - Swish slower. Option C wrong - requires storing x and sigmoid(x) for backward. Production: EfficientNet, NFNet, some Vision Transformers use Swish. Trade-off: Compute overhead for accuracy gain.",
            "Hard",
            200
        ),
        create_question(
            "Q8: For a 100-layer ResNet, what is the gradient magnitude at layer 1 (near input) compared to layer 100 WITHOUT skip connections?",
            [
                "Similar magnitude - gradients propagate equally through network",
                "~0 (vanishing gradients) - gradient shrinks exponentially with depth (~0.9^100 ≈ 0.000027)",
                "Larger at layer 1 - gradients accumulate during backprop",
                "Exploding - gradients grow exponentially"
            ],
            1,
            "Senior Explanation: Without skip connections, gradient passes through 100 layers. If each layer multiplies gradient by ~0.9 (typical Jacobian eigenvalue <1): gradient × 0.9^100 ≈ 0. Even with ReLU (gradient 0 or 1), ~50 layers cause 0.5^50 ≈ 1e-15 shrinkage. With skip connections (ResNet): gradient flows through residual path (y = x + F(x)), gradient dy/dx = 1 + dF(x)/dx ≈ 1 (even if dF/dx ≈ 0). This preserves gradient magnitude. Option A 'junior trap' - assumes perfect propagation. Option D - exploding happens if Jacobian >1 (e.g., bad initialization), not typical. Production: Skip connections are WHY ResNet trains 100+ layers. Plain CNNs struggle beyond 20-30 layers (vanishing gradients). Benchmark: ResNet-110 trains successfully, plain-110 fails to converge. Trade-off: Skip connections add memory (store x for backward) but enable deep networks.",
            "Hard",
            220
        ),
        create_question(
            "Q9: You implement Mish activation (x × tanh(softplus(x))). What is the computational bottleneck?",
            [
                "Tanh computation - requires exponential operations",
                "Softplus(x) = log(1 + exp(x)) - exp and log are slow transcendental functions",
                "Multiplication - memory bandwidth limited",
                "No bottleneck - Mish is highly optimized"
            ],
            1,
            "Senior Explanation: Softplus(x) = log(1 + exp(x)) requires: (1) exp(x) (~50-100 CPU cycles), (2) addition, (3) log(x) (~50-100 cycles). Total ~100-200 cycles per element. Tanh ~50 cycles. Multiplication ~1 cycle. Softplus dominates. For 10M activations: Mish ~1-2s, ReLU ~10-50ms = 20-200× slower. Option A - tanh is expensive but less than softplus. Option C wrong - compute-bound, not memory-bound for transcendental functions. Production: Mish used in YOLOv4, some detection models. Shows ~1-2% mAP improvement over ReLU. Not widely adopted due to cost. Trade-off: Accuracy gain vs significant compute overhead (20-200× slower). Approximations exist (piecewise polynomial Mish) for 5-10× speedup with <0.1% degradation. Modern trend: GELU or Swish (better cost/benefit than Mish).",
            "Medium",
            180
        ),

        # RESIDUAL CONNECTIONS & SKIP CONNECTIONS (Questions 10-12)
        create_question(
            "Q10: In Transformer's Pre-LN (Pre-LayerNorm) vs Post-LN (Post-LayerNorm) architecture, which is more stable for training deep models (>24 layers)?",
            [
                "Post-LN - original Transformer design is always better",
                "Pre-LN - LayerNorm before sub-layer (attention/FFN) improves gradient flow and stability",
                "Both identical - normalization placement doesn't affect stability",
                "Depends on learning rate - both work with proper tuning"
            ],
            1,
            "Senior Explanation: Post-LN: y = LayerNorm(x + SubLayer(x)). Pre-LN: y = x + SubLayer(LayerNorm(x)). Pre-LN advantages: (1) Gradient flow: gradients pass through skip connection WITHOUT passing through LayerNorm (which has small eigenvalues), preventing attenuation. (2) No learning rate warmup needed (Post-LN requires careful warmup to avoid divergence). Depth: Pre-LN trains 48+ layers easily, Post-LN struggles beyond 24 without tricks. Option A 'junior trap' - original isn't always best. Production: GPT-2 used Post-LN, GPT-3+ switched to Pre-LN for stability. Modern Transformers (T5, BERT variants) use Pre-LN. Trade-off: Pre-LN has slightly worse performance (0.5-1% perplexity) when both converge, but MUCH easier to train deep models. For shallow models (<12 layers), Post-LN competitive.",
            "Hard",
            220
        ),
        create_question(
            "Q11: Dense connections (DenseNet) concatenate all previous layer outputs. For a DenseNet with L=100 layers, growth rate k=32, input channels 64, what is the channel count at layer 100?",
            [
                "64 + 32 = 96 - grows by k each layer",
                "64 + 100 × 32 = 3264 - cumulative concatenation",
                "64 × 32 = 2048 - multiplicative growth",
                "32 - constant after first layer"
            ],
            1,
            "Senior Explanation: DenseNet layer l receives concatenation of all previous layers' outputs. Layer 1 output: k=32 channels. Layer 2 input: 64 (original) + 32 (layer 1) = 96 channels. Layer 2 output: 32 channels. Layer 3 input: 96 + 32 = 128. Pattern: Layer l input channels = 64 + (l-1) × k. Layer 100 input: 64 + 99 × 32 = 3232 channels. Option A 'junior trap' - forgets cumulative concatenation. Memory: Layer 100 processes 3232-channel input - HUGE. Typical DenseNet uses transition layers (1×1 conv + pooling) every ~12 layers to compress channels. Production: DenseNet-121 has 4 dense blocks with transitions. Without transitions, memory explodes (3232 × H × W × 4 bytes). Trade-off: Dense connections improve gradient flow but use massive memory. Growth rate k=12-32 typical (smaller k for deeper networks).",
            "Medium",
            180
        ),
        create_question(
            "Q12: ResNeXt uses grouped convolutions with cardinality C=32 (32 groups). For input channels=256, output=256, kernel=3×3, what is the parameter count vs standard ResNet block?",
            [
                "Same parameters - grouped conv is just a different computation pattern",
                "32× fewer parameters - each group has 1/32 of the parameters",
                "~32× fewer - (256/32) × (256/32) × 3 × 3 per group × 32 groups = 18K vs 590K for standard conv",
                "More parameters - grouped conv adds group-specific weights"
            ],
            2,
            "Senior Explanation: Standard conv: 256 (in) × 256 (out) × 3 × 3 = 589,824 params. Grouped conv with C=32 groups: Each group processes 256/32=8 input channels, produces 8 output channels. Per group: 8 × 8 × 3 × 3 = 576 params. Total: 576 × 32 = 18,432 params (32× reduction). ResNeXt compensates by using more channels or more groups to maintain capacity. Option A 'junior trap' - grouped conv drastically reduces params. Trade-off: Fewer params, less compute (32× faster), but more groups (cardinality) improves accuracy by learning diverse paths. ResNeXt-50 (32×4d): 32 groups, 4 channels per group, outperforms ResNet-50 with fewer FLOPs. Production: Grouped convs used in MobileNet, ShuffleNet, EfficientNet for efficiency. Modern trend: Depthwise separable convs (extreme grouping).",
            "Hard",
            200
        ),

        # MODEL INITIALIZATION (Questions 13-15)
        create_question(
            "Q13: Xavier/Glorot initialization sets weights with variance = 2/(n_in + n_out). For a layer with n_in=512, n_out=256, what is the std dev for weight initialization?",
            [
                "sqrt(2 / 768) ≈ 0.051",
                "sqrt(1 / 512) ≈ 0.044",
                "sqrt(2 / 512) ≈ 0.063",
                "1.0 - standard normal initialization"
            ],
            0,
            "Senior Explanation: Xavier initialization: Var(W) = 2 / (n_in + n_out) = 2 / (512 + 256) = 2/768 ≈ 0.0026. Std = sqrt(0.0026) ≈ 0.051. Sample weights: W ~ N(0, 0.051²) or uniform [-0.088, 0.088] (uniform variant: ±sqrt(3) × std). Purpose: Maintains variance of activations and gradients across layers (prevents vanishing/exploding). Derivation assumes linear activations. For ReLU: Use He initialization (Var = 2/n_in) since ReLU zeros half the activations. Option B is He init. Option C is intermediate. Production: PyTorch defaults: Linear layers use Xavier, Conv layers use He (kaiming). Trade-off: Proper initialization critical for convergence - bad init (e.g., std=1.0) causes exploding activations in first iteration. For 100-layer network, bad init → activations of 1.0^100 = 1 (lucky) or 1.5^100 = overflow.",
            "Hard",
            200
        ),
        create_question(
            "Q14: For Transformer models, how are embedding weights typically initialized?",
            [
                "Xavier initialization - standard for all linear layers",
                "Normal(0, 1/sqrt(embed_dim)) - smaller variance for embeddings",
                "Uniform[-0.1, 0.1] - simple bounded initialization",
                "Pre-trained embeddings (e.g., Word2Vec) - no random initialization"
            ],
            1,
            "Senior Explanation: Transformer embeddings initialized with N(0, 1/sqrt(d_model)) where d_model is embedding dimension (e.g., 768). For d_model=768: std = 1/sqrt(768) ≈ 0.036. This ensures embedding magnitude ~1 on average (sqrt(d_model × (1/d_model)) = 1). Positional encodings have similar magnitude (~1), so they can be summed without one dominating. Option A (Xavier) uses 2/(n_in+n_out) - not standard for embeddings. Option C used in older RNNs. Option D for fine-tuning, not training from scratch. Production: BERT, GPT, T5 all use N(0, 1/sqrt(d_model)). Code: nn.Embedding(vocab_size, d_model); nn.init.normal_(embedding.weight, mean=0, std=1/sqrt(d_model)). Trade-off: Proper scaling ensures embeddings and positional encodings balance in magnitude.",
            "Medium",
            180
        ),
        create_question(
            "Q15: You initialize a 50-layer network and notice layer 50 outputs have std=0.001 (too small). What is the likely cause?",
            [
                "Incorrect initialization - weights too small",
                "Gradient vanishing during forward pass - cumulative effect of activations <1 magnitude through layers",
                "Learning rate too low",
                "Batch size too small causing noisy statistics"
            ],
            1,
            "Senior Explanation: Even with correct weight initialization, activations can shrink through layers if activation functions (ReLU) or normalization reduce magnitude. For ReLU: Half of activations zeroed, reducing magnitude by ~sqrt(2) per layer (if not compensated by He init). Over 50 layers: (1/sqrt(2))^50 ≈ 1e-8 shrinkage. Normalization (BatchNorm/LayerNorm) stabilizes this. Without normalization + wrong init: Activations collapse to ~0. Option A possible but less likely (standard frameworks use good defaults). Option C/D don't affect forward pass magnitude. Production: This is why BatchNorm was revolutionary - enables training very deep networks by preventing activation shrinkage/explosion. Check: Inspect intermediate activations (hooks), look for layers with collapsing magnitude. Fix: (1) Add normalization, (2) Use skip connections, (3) Verify initialization (He for ReLU, Xavier for tanh/sigmoid).",
            "Hard",
            200
        ),

        # EFFICIENT ARCHITECTURES (Questions 16-20)
        create_question(
            "Q16: MobileNetV2 uses inverted residual blocks (expand → depthwise → project). For input channels=24, expansion=6, output=24, what is the memory footprint for activations?",
            [
                "~24 units - input channels dominate",
                "~144 units - expanded dimension (24 × 6) dominates",
                "~48 units - input + output",
                "Constant - depthwise conv doesn't change memory"
            ],
            1,
            "Senior Explanation: Inverted residual: (1) 1×1 conv expands 24 → 144 channels, (2) Depthwise conv (144 channels, spatial), (3) 1×1 conv projects 144 → 24. Activation memory: Input (24) + expanded (144) + output (24). Peak: 144-channel activations after expansion. For spatial size H×W=56×56, batch=32: 32 × 56 × 56 × 144 × 4 bytes = 57MB. Input/output: 32 × 56 × 56 × 24 = 9.6MB. Expanded layer dominates. Option A/C 'junior trap' - forgetting intermediate expansion. Production: Expansion factor 6 is standard (balances accuracy vs memory/compute). Gradient checkpointing: Don't store expanded activations, recompute during backward (saves 57MB → 9.6MB, 6× reduction, ~30% slowdown). Trade-off: High expansion (6-8) better accuracy but more memory; low expansion (2-4) more efficient.",
            "Hard",
            200
        ),
        create_question(
            "Q17: EfficientNet uses compound scaling (depth, width, resolution). For a base model with depth=D, width=W, resolution=R, compound scaling with coefficient φ=1.2 gives what?",
            [
                "D' = 1.2D, W' = 1.2W, R' = 1.2R - uniform scaling",
                "D' = 1.2^α D, W' = 1.2^β W, R' = 1.2^γ R where α, β, γ satisfy αβ²γ² ≈ 2 (FLOPs doubling)",
                "D' = D + 1.2, W' = W + 1.2, R' = R + 1.2 - additive scaling",
                "Randomly sample D', W', R' within ±20% of base"
            ],
            1,
            "Senior Explanation: EfficientNet compound scaling: depth × width² × resolution² ≈ constant (FLOPs budget). With coefficient φ: d = α^φ, w = β^φ, r = γ^φ where αβ²γ² ≈ 2 (doubling FLOPs per φ=1 increase). For EfficientNet: α=1.2, β=1.1, γ=1.15 satisfy 1.2 × 1.1² × 1.15² ≈ 2. With φ=1.2: D' = 1.2^1.2 D ≈ 1.22D, W' = 1.1^1.2 W ≈ 1.12W, R' = 1.15^1.2 R ≈ 1.18R. FLOPs increase: ~2^1.2 ≈ 2.3×. Option A 'junior trap' - ignores quadratic FLOPs impact of width/resolution. Production: EfficientNet-B0 to B7 use φ = 0, 1, 2, ..., 7. B7 (φ=7): ~60× FLOPs of B0, much better accuracy. Trade-off: Balanced scaling (depth+width+resolution) outperforms single-axis scaling (e.g., only depth like ResNet-50→152).",
            "Hard",
            220
        ),
        create_question(
            "Q18: Depthwise separable convolution (MobileNet) splits standard conv into depthwise + pointwise. For input=128 channels, output=256, kernel=3×3, what is the parameter reduction?",
            [
                "2× fewer parameters",
                "~8× fewer - (128 × 3 × 3) + (128 × 256) = 33.4K vs 128 × 256 × 3 × 3 = 295K",
                "No reduction - same parameters, different computation",
                "32× fewer - depthwise drastically reduces params"
            ],
            1,
            "Senior Explanation: Standard conv: 128 (in) × 256 (out) × 3 × 3 = 294,912 params. Depthwise separable: (1) Depthwise (per-channel 3×3): 128 × 3 × 3 = 1,152 params, (2) Pointwise (1×1 conv 128→256): 128 × 256 = 32,768 params. Total: 33,920 params. Reduction: 294,912 / 33,920 ≈ 8.7×. FLOPs reduction similar (~8-9×). Accuracy: Slight degradation (1-3% on ImageNet) vs standard conv. Option A/D wrong calculations. Production: MobileNet, ShuffleNet, EfficientNet heavily use depthwise separable. Enables mobile deployment (1-5M params vs 25-50M for ResNet). Trade-off: Efficiency (8× fewer params/FLOPs) vs slight accuracy loss. For resource-constrained devices (phones, edge), depthwise separable is essential.",
            "Hard",
            200
        ),
        create_question(
            "Q19: Squeeze-and-Excitation (SE) blocks (used in SE-ResNet) apply channel attention. For C=512 channels, reduction ratio r=16, what is the parameter overhead?",
            [
                "Negligible (~1KB) - SE adds minimal parameters",
                "~16K parameters - global pool → FC(512→32) → FC(32→512)",
                "~256K parameters - significant overhead",
                "Zero parameters - SE is parameter-free attention"
            ],
            1,
            "Senior Explanation: SE block: (1) Global average pool (C channels → C scalars), (2) FC layer (C → C/r), (3) ReLU, (4) FC layer (C/r → C), (5) Sigmoid, (6) Scale input channels. Parameters: FC1: C × (C/r) = 512 × 32 = 16,384. FC2: (C/r) × C = 32 × 512 = 16,384. Total: 32,768 params. For ResNet-50 (~25M params), SE adds ~2-3M params (~10% overhead). Compute: Negligible (global pool + 2 small FCs). Accuracy: +1-2% on ImageNet. Option A/D wrong. Production: SE-ResNet, SE-ResNeXt use SE blocks. Trade-off: 10% param overhead for 1-2% accuracy gain - good ROI. Modern variants: ECA (Efficient Channel Attention) reduces params further with 1D conv instead of FCs (similar performance, fewer params).",
            "Medium",
            180
        ),
        create_question(
            "Q20: For a Vision Transformer (ViT) with patch size 16×16, image 224×224, embedding dim 768, what is the sequence length?",
            [
                "14 - number of patches per row",
                "196 - (224/16)² = 14² patches + 1 CLS token = 197",
                "197 - 196 patches + 1 CLS token",
                "224 - matches image height"
            ],
            2,
            "Senior Explanation: Image 224×224 divided into 16×16 patches: 224/16 = 14 patches per side. Total patches: 14 × 14 = 196. ViT adds 1 CLS (classification) token at position 0. Total sequence length L = 196 + 1 = 197. Each patch (16×16×3 = 768 values) is linearly projected to embedding dim (768). Memory: Attention matrix (B, num_heads, 197, 197). For batch=256, 12 heads: 256 × 12 × 197² × 4 bytes = 2.4GB (fp32). Option B forgets CLS token. Production: ViT-Base (patch=16, dim=768), ViT-Large (patch=14, dim=1024). Larger patches (32×32) reduce sequence length (49 patches) but lose fine-grained info. Trade-off: Smaller patches better accuracy but quadratic memory cost. ViT-Huge (patch=14, image=224): L=257, attention ~4GB per batch=256.",
            "Medium",
            180
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_deep_learning()
    db.add_questions("Senior Deep Learning - Advanced Architecture", questions)
    print(f"✓ Successfully added {len(questions)} senior Deep Learning questions!")
    print(f"✓ Category: Senior Deep Learning - Advanced Architecture")
