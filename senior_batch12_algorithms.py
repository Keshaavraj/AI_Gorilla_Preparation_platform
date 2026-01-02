"""
Senior AI Engineer Interview Questions - Batch 12: Algorithms & Complexity for ML
Topics: Space Complexity of Tensors, In-place Operations, Memory Layout, Graph Algorithms, Optimization Complexity
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_algorithms():
    """20 Senior-Level Algorithms & Complexity Questions"""
    questions = [
        # SPACE COMPLEXITY OF TENSOR OPERATIONS (Questions 1-6)
        create_question(
            "Q1: For matrix multiplication C = A @ B where A is (M, K) and B is (K, N), what is the space complexity including inputs, output, and temporary buffers?",
            [
                "O(MK + KN + MN) - just the three matrices",
                "O(MN) - only output matrix (inputs can be streamed)",
                "O(max(MK, KN, MN)) - largest matrix dominates",
                "O(MKN) - requires intermediate products"
            ],
            0,
            "Senior Explanation: Naive matmul: Needs A (M×K elements), B (K×N elements), C (M×N elements). Total: O(MK + KN + MN). For square matrices (M=N=K=n): O(3n²). No additional space needed for standard algorithms (Strassen's algorithm uses O(n²) auxiliary space). Production example: A (1024×2048) @ B (2048×512) requires 1024×2048 + 2048×512 + 1024×512 = 2M + 1M + 0.5M = 3.5M elements × 4 bytes = 14MB. GPU memory: Must fit all three matrices + CUDA kernel workspace (~few MB). Option B wrong - can't stream inputs for matmul (need random access). Trade-off: Larger matrices may need tiling/blocking to fit in cache/VRAM, increasing complexity to O(n²) but maintaining same asymptotic space.",
            "Hard",
            200
        ),
        create_question(
            "Q2: During backpropagation in a 100-layer ResNet, what is the memory complexity for storing activations (without gradient checkpointing)?",
            [
                "O(L) - linear in number of layers",
                "O(L × B × H × W) - layer count × batch size × spatial dimensions",
                "O(B × H × W) - only need final layer activations",
                "O(1) - constant memory with in-place operations"
            ],
            1,
            "Senior Explanation: Must store ALL layer activations for gradient computation. Each layer: Batch × Channels × Height × Width. For ResNet-50 (50 layers, downsample 5×): Layer 1: B×64×56×56, Layer 2-5: varying dimensions. Total: ~50 layers × B×C×H×W. For B=32, typical ResNet-50: ~5-10GB activation memory (fp32). Space: O(L × B × C × H × W) where C, H, W vary per layer. Option A ignores batch/spatial dims. Option D wrong - can't do in-place for activations needed in backward. Gradient checkpointing: Stores subset of activations (every N layers), recomputes others during backward. Reduces to O(sqrt(L) × B × C × H × W) space at cost of ~30% more compute. Production: Activation memory dominates for large batch/high-res images. Trade-off: Memory vs compute.",
            "Hard",
            220
        ),
        create_question(
            "Q3: For self-attention with sequence length L, batch size B, hidden dim H, what is the space complexity of the attention operation?",
            [
                "O(B × L × H) - query/key/value matrices",
                "O(B × L² × H) - attention matrix dominates",
                "O(B × L²) - attention scores (B, num_heads, L, L)",
                "O(H²) - weight matrices"
            ],
            2,
            "Senior Explanation: Attention memory: (1) Q, K, V: 3 × (B, num_heads, L, head_dim) ≈ O(B × L × H), (2) Attention matrix (QK^T): (B, num_heads, L, L) = O(B × L²), (3) Output: O(B × L × H). Dominant term: Attention matrix O(B × L²) for large L. For B=32, L=2048, 12 heads: 32×12×2048×2048×4 bytes = 6.4GB (fp32). Q,K,V: 32×2048×768×4×3 = 0.6GB. Attention matrix 10× larger. This is why Flash Attention critical - avoids materializing L² attention matrix. Option A ignores attention matrix. Production: Standard attention OOMs at L>4096. Flash Attention reduces to O(B × L × H) by computing attention on-the-fly in blocks. Trade-off: O(L²) memory limits context length dramatically.",
            "Hard",
            220
        ),
        create_question(
            "Q4: For element-wise operations (ReLU, sigmoid) on tensor of size (B, C, H, W), what is the space complexity?",
            [
                "O(BCHW) - need output tensor",
                "O(1) - can be computed in-place",
                "O(2 × BCHW) - input + output",
                "O(log(BCHW)) - sublinear"
            ],
            1,
            "Senior Explanation: Element-wise ops CAN be in-place: tensor.relu_() modifies tensor in-place, zero extra space (O(1) auxiliary). Non-in-place: tensor.relu() creates new tensor, O(BCHW) extra space. In training: Must keep input for gradient computation (can't do in-place if input needed in backward). In inference: Can do in-place (no gradients). PyTorch: Operations ending with '_' are in-place (tensor.add_(1), tensor.clamp_(min=0)). Benefits: Save memory (no temporary tensors). Risks: Modifying tensor used elsewhere causes bugs. Production: In-place ops save memory in inference (model(x) can modify x if x not reused). Training: Rarely use in-place on activations (need for backward), but use on gradients (optimizer.step() updates weights in-place). Trade-off: Memory savings vs safety.",
            "Medium",
            180
        ),
        create_question(
            "Q5: For concatenating N tensors of shape (B, C, H, W) along channel dimension, what is the space complexity?",
            [
                "O(N × B × C × H × W) - sum of all input tensors",
                "O(B × N×C × H × W) - output tensor only (inputs can be freed)",
                "O(B × C × H × W) - constant",
                "O((N+1) × B × C × H × W) - inputs + output"
            ],
            3,
            "Senior Explanation: torch.cat([t1, t2, ..., tN], dim=1) creates new tensor with concatenated data. Input tensors: N × (B, C, H, W) = O(N × B × C × H × W). Output: (B, N×C, H, W) = same O(N × B × C × H × W). Total: O(2N × B × C × H × W) ≈ O(N × B × C × H × W). Inputs not freed automatically (Python garbage collection later). Memory-efficient: del t1, t2, ...; result = torch.cat(...) then inputs freed after cat. Production: DenseNet concatenates all previous layers - memory grows linearly. For DenseNet-121 (121 layers, growth rate k=32): Final layer concatenates 121×32 = 3872 channels. With transition layers (compression), manageable. Trade-off: Concatenation creates new tensor (memory cost) but provides flexibility. Alternative: In-place operations where possible.",
            "Medium",
            180
        ),
        create_question(
            "Q6: For gradient accumulation (accumulate gradients over N micro-batches before optimizer step), what is the extra memory overhead?",
            [
                "O(N × model_size) - store N sets of gradients",
                "O(model_size) - gradients accumulated in-place into single gradient buffer",
                "O(N) - only step counter",
                "O(log N) - compressed gradient storage"
            ],
            1,
            "Senior Explanation: Gradient accumulation: loss.backward() ACCUMULATES gradients into parameter.grad tensors (+=, not replace). Grad buffer size: O(number of parameters) regardless of accumulation steps. For 7B model: gradients = 7B × 4 bytes (fp32) = 28GB whether accumulating 1 step or 100 steps. Process: Step 1: loss.backward() writes grads. Step 2-N: loss.backward() adds to existing grads (tensor.grad += new_grad). Step N+1: optimizer.step() uses accumulated grads, optimizer.zero_grad() clears. Extra memory: Zero (grads stored anyway for single-batch). Benefit: Train with effective batch size N × micro_batch_size on limited memory. Production: Train with batch=256 on 16GB GPU via batch=4 × 64 accumulation steps. Trade-off: More steps = more time (64 forward/backward vs 1 batched), but enables large effective batches.",
            "Hard",
            200
        ),

        # IN-PLACE VS OUT-OF-PLACE (Questions 7-10)
        create_question(
            "Q7: For normalizing a tensor (subtract mean, divide by std), when can you safely use in-place operations?",
            [
                "Always - in-place saves memory",
                "Only during inference - training needs original tensor for gradients",
                "Never - numerical stability requires out-of-place",
                "Only for small tensors"
            ],
            1,
            "Senior Explanation: Training: tensor.sub_(mean).div_(std) breaks autograd - can't compute gradient w.r.t original tensor (modified in-place). Use: normalized = (tensor - mean) / std (creates new tensor, original preserved for backward). Inference: Can use in-place (no gradients). Saves memory: Single tensor modified vs creating 3 temporaries (tensor-mean, result/std, normalized). Production: BatchNorm training uses out-of-place (inputs needed for gradient), inference uses in-place or fused kernels. Trade-off: In-place faster and memory-efficient but incompatible with autograd. PyTorch raises error if in-place op breaks gradient computation. Alternative: torch.nn.functional operations (out-of-place by default, autograd-safe).",
            "Hard",
            200
        ),
        create_question(
            "Q8: What is the time complexity of in-place addition (tensor.add_(1)) vs out-of-place (tensor + 1) for tensor of size N?",
            [
                "In-place O(N), out-of-place O(N²) - copying expensive",
                "Both O(N) - same computation, just different memory allocation",
                "In-place O(1), out-of-place O(N)",
                "In-place faster by constant factor but same O(N)"
            ],
            3,
            "Senior Explanation: Both iterate N elements and add 1: O(N) time. Difference: Memory allocation overhead (malloc/free for output tensor in out-of-place). In-place: ~N ops (pure computation). Out-of-place: ~N ops + allocation (~N bytes memcpy) + deallocation. Speedup: 1.2-2× (in-place faster by constant, not asymptotically). For N=1M elements (4MB tensor): In-place ~1ms, out-of-place ~1.5ms (allocation ~0.5ms). GPU: Allocation faster (pre-allocated memory pool) - gap smaller (~10-20% speedup). Production: In-place optimizations matter for tight loops (e.g., optimizer.step() uses in-place updates on all parameters). Trade-off: Marginal speed gain vs safety risks.",
            "Medium",
            180
        ),
        create_question(
            "Q9: For a function f(x) = x² + 2x + 1, which implementation is most memory efficient for large tensor x?",
            [
                "result = x**2 + 2*x + 1 - clean and readable",
                "result = x.pow(2).add(x.mul(2)).add(1) - method chaining",
                "x.pow_(2).add_(x_copy.mul(2)).add_(1) where x_copy = x.clone() - in-place with clone",
                "Fused kernel computing all at once - custom CUDA kernel"
            ],
            3,
            "Senior Explanation: Option A: Creates 3 temporaries (x**2, 2*x, x**2+2*x) before final result. For x size N: Peak memory ~4N (x + 3 temporaries). Option B: Same (method chaining doesn't reduce temporaries). Option C: Clone defeats purpose (x_copy is copy). Option D: Fused kernel reads x once, computes expression, writes once. Memory: 2N (input + output). No temporaries. Speedup: 2-3× (fewer memory transfers, single kernel launch). PyTorch JIT can fuse simple expressions automatically. Production: TorchScript @torch.jit.script or torch.compile fuses operations. Manually write CUDA for custom ops. Trade-off: Fused kernels optimal but complex to implement. For standard ops, rely on framework fusion.",
            "Hard",
            220
        ),
        create_question(
            "Q10: Which operations are guaranteed in-place in PyTorch?",
            [
                "All operations with trailing underscore (e.g., tensor.add_(), tensor.relu_())",
                "Only underscore ops where no gradient needed",
                "All assignment operations",
                "No operations are guaranteed in-place"
            ],
            0,
            "Senior Explanation: PyTorch convention: Operations ending with '_' modify tensor in-place. Examples: tensor.add_(1), tensor.mul_(2), tensor.clamp_(0, 1), tensor.copy_(other). Guaranteed behavior: Modify tensor's data pointer, no new tensor allocated. Autograd: In-place ops tracked, but complex interactions may raise errors (e.g., modifying tensor needed for gradients of other ops). Production: Use in-place ops carefully - verify with tensor.is_contiguous() and check autograd compatibility. Memory benefit: For 1M-element tensor updated 1000 times - in-place ~4MB, out-of-place ~4GB (1000 allocations). Trade-off: In-place efficient but easy to introduce bugs (unintended mutations, autograd errors). Non-underscore ops always out-of-place (safe default).",
            "Medium",
            180
        ),

        # MEMORY LAYOUT & CACHE (Questions 11-14)
        create_question(
            "Q11: For summing elements of a 2D array (1000×1000), which iteration order has better cache performance?",
            [
                "for i in range(1000): for j in range(1000): sum += arr[i][j] - row-major",
                "for j in range(1000): for i in range(1000): sum += arr[i][j] - column-major",
                "No difference - modern CPUs cache efficiently",
                "Depends on array layout (C-contiguous vs F-contiguous)"
            ],
            3,
            "Senior Explanation: C-contiguous (row-major): Rows stored sequentially. Iterate row-wise (outer=i, inner=j) for sequential memory access → cache-friendly. Column-wise iteration jumps 1000 elements per access → cache misses. F-contiguous (column-major): Opposite. Performance: Row-wise on C-contiguous ~10-50× faster than column-wise (cache hits vs misses). Cache line: 64 bytes = 16 float32s. Row-wise loads 16 elements per cache line (amortized). Column-wise loads 1 element per cache line (wastes 15/16 of bandwidth). Production: NumPy default C-contiguous. Always iterate matching layout. Check: arr.flags['C_CONTIGUOUS']. Trade-off: Cache-aware iteration free (just change loop order) for massive speedup. Benchmark: 1000×1000 float32 sum - row-wise ~1ms, column-wise ~50ms.",
            "Hard",
            220
        ),
        create_question(
            "Q12: For matrix multiplication C = A @ B, what memory access pattern is most cache-friendly for naive algorithm?",
            [
                "for i: for j: for k: C[i][j] += A[i][k] * B[k][j] - standard loops",
                "for i: for k: for j: C[i][j] += A[i][k] * B[k][j] - ikj order",
                "Blocking/tiling - divide into cache-sized blocks",
                "No difference - all O(n³)"
            ],
            2,
            "Senior Explanation: Naive loops: Poor cache reuse for B (B[k][j] accessed non-sequentially). Blocking: Divide matrices into tiles (e.g., 64×64) fitting in cache. Compute tile-wise matmul. Cache: 256KB L2 can fit ~4000 float64s = 63×63 matrix. Tiling ensures tiles stay in cache during computation. Speedup: 5-10× vs naive for large matrices. Production: BLAS libraries (OpenBLAS, MKL) use multi-level tiling + vectorization + threading → 100-1000× faster than naive. Example: 1024×1024 matmul - naive ~10s, BLAS ~10ms (1000×). Option B (ikj order) helps but tiling is optimal. Trade-off: Tiled implementation complex but critical for performance. Use BLAS for production, understand tiling for interviews.",
            "Hard",
            220
        ),
        create_question(
            "Q13: For training a CNN, activation memory typically dominates. Why?",
            [
                "Activations larger than weights",
                "Must store ALL layer activations for backprop - grows with batch size and image resolution",
                "Activations not compressed",
                "Activations recomputed multiple times"
            ],
            1,
            "Senior Explanation: ResNet-50: ~25M parameters (weights) = 100MB (fp32). But activations for batch=32, 224×224 images: ~5-10GB (50-100× larger). Why: Each layer stores B×C×H×W activations for gradient computation. 50 layers × varying C,H,W → GBs. Weights reused across batch (same 100MB), activations grow with batch. Solutions: (1) Gradient checkpointing (store subset, recompute others - trades 2× compute for 5-10× memory), (2) Smaller batch (batch=16 vs 32 halves activation memory), (3) Mixed precision (fp16 activations - halves memory). Production: Training large CNNs on ImageNet (batch=256) requires 8× V100 (32GB each) due to activation memory. Trade-off: Larger batch (better accuracy) vs memory constraints. Weights are O(model size), activations O(batch size × input size × depth).",
            "Medium",
            180
        ),
        create_question(
            "Q14: For convolution on image (3×H×W) with 64 filters (3×3 kernels), what is the space complexity?",
            [
                "O(H × W) - output only",
                "O(3×H×W + 64×3×3 + 64×H×W) - input + filters + output",
                "O(64×H×W) - output dominates",
                "O(H×W×64×9) - all intermediate products"
            ],
            1,
            "Senior Explanation: Conv2d memory: Input (3, H, W), filters (64, 3, 3, 3) - 64 filters each 3×3×3, output (64, H, W). Total: 3HW + 64×3×3×3 + 64HW = 3HW + 1728 + 64HW = 67HW + 1728. For H=W=224: 67×224² + 1728 ≈ 3.4M elements × 4 bytes = 13.6MB. Filters (1728 elements = 7KB) negligible vs activations. Typical workspace for conv algorithms (im2col, FFT): O(C_in × k² × H × W) for im2col (explodes input), but frameworks optimize. Production: cuDNN uses various algorithms (im2col, FFT, Winograd) - different memory/speed tradeoffs. Auto-select based on input size. Trade-off: Im2col fast but high memory (5-10× input), FFT low memory but slower for small kernels. Winograd optimal for 3×3 kernels (2.25× faster, same memory).",
            "Hard",
            200
        ),

        # GRAPH ALGORITHMS & OPTIMIZATION (Questions 15-20)
        create_question(
            "Q15: For a computation graph with N nodes (operations), what is the time complexity of backpropagation?",
            [
                "O(N²) - must visit all pairs",
                "O(N) - visit each node once in reverse topological order",
                "O(N log N) - tree traversal",
                "O(E) - depends on edges, not nodes"
            ],
            1,
            "Senior Explanation: Backprop is reverse-mode autodiff: Traverse computation graph in reverse topological order (outputs to inputs), compute gradients via chain rule at each node. Each node visited once: O(N). For DAG with E edges, also O(E) work propagating gradients along edges. Typically E ≈ N (each op has 1-3 inputs), so O(N). Complexity per node: Depends on operation (matmul backward O(n²), relu backward O(n)). Total: O(N × average_op_cost). Production: PyTorch autograd builds computation graph during forward, traverses in backward. Graph size: O(model ops × batch size) but independent of training iterations. Trade-off: Autograd overhead ~20-50% vs manual gradients (graph building + traversal) but worth it for flexibility.",
            "Hard",
            200
        ),
        create_question(
            "Q16: For deadlock detection in distributed training (circular wait on gradient synchronization), what algorithm is used?",
            [
                "DFS to detect cycles in resource allocation graph",
                "Timeout-based detection - if synchronization takes >threshold, assume deadlock",
                "Banker's algorithm for deadlock avoidance",
                "No algorithm - deadlocks impossible in data parallel training"
            ],
            1,
            "Senior Explanation: Distributed training deadlocks: Rare but possible (e.g., mismatched collective calls - rank 0 calls allreduce, rank 1 calls broadcast → hang). Detection: Timeout-based - if collective operation doesn't complete in reasonable time (e.g., 60s), assume deadlock/failure. Framework: NCCL watchdog threads, allreduce with timeout parameter. Option A (cycle detection) theoretically sound but impractical (no global resource graph in distributed system). Option C (Banker's) for deadlock avoidance, not detection. Production: Set collective timeout (e.g., 30s for small models, 300s for large), log rank states for debugging. Common causes: Rank mismatch (different code paths), network partition, hardware failure. Trade-off: Short timeout false positives (slow network), long timeout delays failure detection.",
            "Medium",
            180
        ),
        create_question(
            "Q17: Adam optimizer state (momentum + variance) for model with P parameters has space complexity?",
            [
                "O(P) - just one state vector",
                "O(2P) - two state vectors (momentum m_t and variance v_t)",
                "O(3P) - states + gradients",
                "O(log P) - compressed state"
            ],
            1,
            "Senior Explanation: Adam maintains: m_t (first moment), v_t (second moment) for each parameter. Total state: 2P. For 7B model: 7B × 2 × 4 bytes (fp32) = 56GB. SGD with momentum: 1P (only momentum). Adafactor: ~sqrt(P) via factorized second moment (memory-efficient variant). Production: Adam's 2P state dominates memory for large models. 7B model: weights 14GB (fp16) + gradients 14GB + Adam state 56GB = 84GB. Optimization: Use 8-bit Adam (state in int8) reduces to 14GB state, total ~42GB (fits on A100 80GB). Trade-off: Adam faster convergence but 2× memory vs SGD. For huge models (100B+), use memory-efficient optimizers (Adafactor, 8-bit Adam, or SGD).",
            "Medium",
            180
        ),
        create_question(
            "Q18: For finding learning rate (LR range test - train with exponentially increasing LR), what is the time complexity?",
            [
                "O(1) - constant time test",
                "O(K × N) where K = number of LR steps, N = dataset size (must train for K iterations)",
                "O(log K × N) - binary search",
                "O(K) - independent of dataset"
            ],
            1,
            "Senior Explanation: LR range test: Start with lr=1e-7, train for 100-1000 steps, multiply lr by constant (e.g., 1.1) each step. Observe loss vs LR curve, pick LR where loss decreases fastest. Time: K steps × time_per_step. For K=500 steps, batch=64, dataset=1M: Process 500×64 = 32K samples (3% of dataset). Time ~5-10 min for ResNet-50. Not full epoch - just enough to see LR effect. Option C wrong - not binary search (need full curve, not single optimal LR). Production: LR finder in PyTorch Lightning, fastai. Finds good LR in minutes vs hours of grid search. Trade-off: Small overhead (few minutes) for significant benefit (optimal LR → 2-5× faster training).",
            "Medium",
            180
        ),
        create_question(
            "Q19: For model quantization (convert fp32 → int8), what is the time complexity for calibration-based quantization on dataset of size N?",
            [
                "O(1) - quantization is instant",
                "O(N) - single forward pass through dataset to collect activation ranges",
                "O(N²) - must compare all samples",
                "O(N log N) - sorting-based calibration"
            ],
            1,
            "Senior Explanation: Post-training quantization (PTQ): Run calibration set through model (forward only, no backward), collect min/max of activations per layer, compute scale/zero-point, quantize weights. One forward pass: O(N) where N = calibration samples (typically 100-1000). For N=1000, ResNet-50: ~1-2 minutes. Quantization-aware training (QAT): Full training with fake quantization - O(training iterations). Option A wrong - quantization instant but calibration needed. Production: PTQ fast (minutes), suitable for deployment. QAT slow (hours-days) but better quality. Trade-off: PTQ sufficient for 8-bit, QAT needed for 4-bit or high-accuracy requirements. Calibration: 1000 samples enough to estimate activation ranges (more gives diminishing returns).",
            "Medium",
            180
        ),
        create_question(
            "Q20: For graph optimization (fusing Conv+BatchNorm+ReLU into single op), what is the complexity of finding fusable patterns in graph with N nodes?",
            [
                "O(N) - linear scan for patterns",
                "O(N²) - check all pairs",
                "O(N × P) where P = number of fusion patterns - match each pattern against graph",
                "O(N!) - NP-hard problem"
            ],
            2,
            "Senior Explanation: Graph fusion: Pattern matching in computation graph. For each node, check if it matches start of fusion patterns (e.g., Conv node followed by BN followed by ReLU). For N nodes, P patterns: O(N × P) in worst case. Typical P ~10-20 patterns (Conv-BN, Conv-BN-ReLU, MatMul-Add, etc.). For N=1000 nodes, P=20: 20K checks (~1ms). Each check: Local graph traversal (O(pattern size), typically 2-5 nodes). Total: O(N × P × pattern_size) ≈ O(N) for fixed P and pattern sizes. Production: TorchScript, TensorRT, ONNX optimizers fuse operations. Speedup: Fused Conv-BN-ReLU ~2-3× faster than separate ops (fewer kernel launches, memory transfers). Trade-off: Fusion optimization time negligible (<1s) vs runtime savings (2-3× inference speedup).",
            "Hard",
            200
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_algorithms()
    db.add_bulk_questions("Senior Algorithms - Complexity & Memory for ML", questions)
    print(f"✓ Successfully added {len(questions)} senior Algorithms questions!")
    print(f"✓ Category: Senior Algorithms - Complexity & Memory for ML")
