"""
Senior AI Engineer Interview Questions - Batch 1: NumPy Advanced Optimization
Topics: Strides, Broadcasting, Memory Layout, View vs Copy, Vectorization
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_numpy():
    """20 Senior-Level NumPy Questions"""
    questions = [
        # STRIDES & MEMORY LAYOUT (Questions 1-6)
        create_question(
            "Q1: You need to transpose a large 10GB NumPy array (shape: 50000x50000 float32) in a production pipeline with minimal memory overhead. Which approach is most memory-efficient?",
            [
                "Use np.transpose() - it's the standard approach and handles memory automatically",
                "Leverage np.ndarray.T which returns a view by manipulating strides without data copy",
                "Use np.copy() followed by transpose to ensure data locality for better cache performance",
                "Convert to a list of lists, transpose manually, then convert back to ensure no memory leaks"
            ],
            1,
            "Senior Explanation: np.ndarray.T returns a VIEW by swapping the strides tuple, requiring ZERO additional memory (no data copy). For a 10GB array, this is instant and uses ~0 bytes extra. Option A (np.transpose()) also returns a view but is less explicit. Option C (copy + transpose) would require 20GB total (original + transposed copy), wasting memory and taking O(n²) time. Option D is a 'junior trap' - converting to lists would consume massive memory due to Python object overhead (~100GB+) and destroy all NumPy optimizations. In production with VRAM constraints, view operations are critical for efficiency. Trade-off: Views share memory with original array, so modifying the view affects the original.",
            "Hard",
            180
        ),
        create_question(
            "Q2: Given two arrays: A with shape (1000, 1) and B with shape (1, 1000), what is the shape of C = A + B, and what is the computational complexity?",
            [
                "Shape: (1000, 1000), Complexity: O(n) due to NumPy's optimized broadcasting",
                "Shape: (1000, 1000), Complexity: O(n²) due to outer product-style broadcasting",
                "Shape: Error - incompatible shapes for addition",
                "Shape: (1000, 1), Complexity: O(n) - NumPy automatically reduces B to match A's shape"
            ],
            1,
            "Senior Explanation: Broadcasting expands A (1000×1) and B (1×1000) to both become (1000×1000) virtually, then performs element-wise addition. This creates a 1M element output requiring O(n²) operations where n=1000. Memory: 1000 + 1000 + 1,000,000 elements (A + B + C) ≈ 4MB for float32. Option A is the 'junior trap' - broadcasting is optimized but still requires O(n²) work for n² outputs. FLOPs: 1 million additions. Production impact: Accidentally broadcasting large arrays (e.g., 10000×1 and 1×10000 = 100M elements) can cause OOM errors. Always verify output shapes before operations in production pipelines.",
            "Hard",
            150
        ),
        create_question(
            "Q3: You're processing a 100GB dataset with NumPy stride tricks to create a sliding window view (window size: 1000, step: 1). What is the memory footprint of the resulting view array?",
            [
                "~100GB - NumPy creates a full copy for each window position",
                "~0 bytes additional - stride_tricks creates a view without copying data",
                "~1000× original size due to overlapping windows",
                "~50GB - NumPy optimizes by copying only unique window data"
            ],
            1,
            "Senior Explanation: np.lib.stride_tricks.as_strided creates a VIEW by manipulating the strides and shape metadata without copying the underlying data buffer. Memory footprint is effectively 0 bytes additional (only metadata: ~48 bytes for new array object). Option A is the 'junior trap' - beginners assume copies are made. For 100GB data with 99M windows (100B elements - 1000 + 1), a naive copy approach would need ~100TB of memory (impossible). Stride tricks enable memory-efficient windowing for time-series, convolutions, and rolling statistics. Trade-off: Views can cause unexpected behavior if original array is modified. Critical for production ML pipelines processing large sequential sensor/log data.",
            "Hard",
            200
        ),
        create_question(
            "Q4: What happens to NumPy array strides when you perform arr[::2, ::3] on a C-contiguous 2D array with strides (24, 8) and dtype float64?",
            [
                "New strides: (48, 24) - each stride is multiplied by the step size",
                "New strides: (12, 4) - strides are divided by step size for efficiency",
                "Array becomes non-contiguous and strides are reset to default",
                "New strides: (24, 8) - strides remain unchanged, only shape changes"
            ],
            0,
            "Senior Explanation: Slicing with step N multiplies the corresponding stride by N. Original strides (24, 8) → new strides (24×2, 8×3) = (48, 24). This correctly represents jumping 2 rows (48 bytes = 6 float64s) and 3 columns (24 bytes = 3 float64s). The array remains a VIEW (no copy). Option B is the 'junior trap' - dividing makes no physical sense for memory addressing. Understanding strides is crucial for: (1) debugging unexpected view behavior, (2) optimizing memory access patterns for cache locality, (3) avoiding hidden copies. Production impact: Poor stride patterns cause cache misses and 10-100× slowdowns in tight loops. Modern CPUs have 64-byte cache lines; aligned, sequential access achieves ~50 GB/s vs ~500 MB/s for random access.",
            "Hard",
            180
        ),
        create_question(
            "Q5: For a Fortran-contiguous array (shape: 10000×5000, float32), which operation is FASTER: summing along axis=0 or axis=1?",
            [
                "axis=0 (column-wise sum) - it's the default and optimized",
                "axis=1 (row-wise sum) - Fortran layout stores rows contiguously",
                "Both are equal - NumPy auto-optimizes based on layout",
                "axis=0 (column-wise sum) - Fortran layout stores columns contiguously, enabling sequential memory access"
            ],
            3,
            "Senior Explanation: Fortran (column-major) layout stores each column sequentially in memory. Summing along axis=0 (across rows within each column) accesses memory sequentially, fitting in CPU cache lines (64 bytes = 16 float32s). Performance: ~40-50 GB/s (memory bandwidth limited). Summing along axis=1 (across columns) jumps between columns, causing cache misses. Performance: ~2-5 GB/s (10-20× slower). Option B is the 'junior trap' - confusing row/column storage. Production example: Feature standardization (mean/std per feature) on ML training data (samples × features) stored in F-order is 15-30× faster. Trade-off: Choose memory layout at data loading based on dominant access pattern. Converting C↔F requires full data copy (O(n) time).",
            "Hard",
            200
        ),
        create_question(
            "Q6: You create a view with arr_view = arr[1000:2000]. Then arr is deleted. What happens to arr_view?",
            [
                "arr_view becomes invalid and raises an error when accessed",
                "arr_view continues to work - Python keeps the underlying data alive via reference counting",
                "arr_view is automatically converted to a copy to prevent dangling references",
                "Behavior is undefined - can cause segmentation faults"
            ],
            1,
            "Senior Explanation: NumPy views hold a reference to the underlying data buffer, not the original array object. Python's reference counting keeps the buffer alive as long as ANY view references it. When arr is deleted, its refcount decreases, but arr_view's reference prevents deallocation. Memory is freed only when ALL views are deleted. Option A/D are 'junior traps' - NumPy handles this safely. Option C is wrong - no automatic copying. Production implication: Memory leaks can occur if views persist in long-running services. Example: Creating 1000 views from a 10GB array and deleting the original still uses 10GB RAM. Best practice: Use arr.copy() when views won't be needed long-term, or manage view lifetimes explicitly in production pipelines.",
            "Medium",
            180
        ),

        # BROADCASTING (Questions 7-11)
        create_question(
            "Q7: When broadcasting arrays A (100, 200, 1) and B (1, 1, 300), what is the peak memory consumption during C = A * B if float32 is used?",
            [
                "~240 KB - only input arrays are stored",
                "~23 MB - output array (100, 200, 300) after broadcasting",
                "~46 MB - NumPy temporarily expands both arrays before multiplication",
                "~70 MB - input arrays + expanded arrays + output"
            ],
            1,
            "Senior Explanation: Broadcasting is VIRTUAL - arrays are not physically expanded. Memory needed: A (100×200×1×4 = 80KB) + B (1×1×300×4 = 1.2KB) + C (100×200×300×4 = 24MB) ≈ 24.08MB. NumPy iterates efficiently using stride manipulation. Option C/D are 'junior traps' assuming physical expansion - that would require 24MB each for expanded A and B. FLOPs: 6 million multiplications. Production impact: Understanding this prevents OOM errors. Example: Broadcasting (1000, 1, 5000) × (1, 1000, 1) creates 5GB output with only 20MB inputs. Trade-off: Broadcasting saves memory but can hide expensive O(n³) operations. Always check output shape before broadcasting in production.",
            "Hard",
            180
        ),
        create_question(
            "Q8: You want to add a 1D array (shape: 10000,) to each row of a 2D array (shape: 5000×10000). Which is the most efficient?",
            [
                "arr_2d + arr_1d - direct addition, NumPy handles broadcasting",
                "arr_2d + arr_1d.reshape(1, 10000) - explicit broadcasting dimension",
                "arr_2d + arr_1d[np.newaxis, :] - clearer intent with newaxis",
                "np.add(arr_2d, arr_1d, out=arr_2d) - in-place addition to save memory"
            ],
            0,
            "Senior Explanation: Option A is optimal - NumPy's broadcasting rules automatically align arr_1d (10000,) with arr_2d's second dimension (5000, 10000) by prepending a virtual dimension. All options A/B/C produce identical performance (~200-400ms for 200MB data) because they generate the same low-level code. Option B/C are explicit but add cognitive overhead without benefit. Option D is IN-PLACE but only helpful if modifying arr_2d is acceptable (often not in production). 'Junior trap': overthinking broadcasting - trust NumPy's rules. Production pattern: Feature scaling in ML preprocessing - subtracting mean per feature from (samples, features) array. Memory: inputs 200MB + output 200MB = 400MB total. FLOPs: 50M additions.",
            "Medium",
            150
        ),
        create_question(
            "Q9: Which broadcasting operation will FAIL with a shape mismatch error?",
            [
                "A (256, 1, 64) + B (1, 128, 64)",
                "A (256, 128, 1) + B (1, 128, 64)",
                "A (256, 128, 64) + B (1, 1, 64)",
                "A (256, 128, 64) + B (256, 64)"
            ],
            3,
            "Senior Explanation: Broadcasting aligns arrays from the RIGHTMOST dimension, extending left with 1s. Option D: A (256, 128, 64) vs B (256, 64) → align as A (256, 128, 64) and B (256, 1, 64) - MISMATCH at dimension 1 (128 vs 1 is ok, but 128 vs 64 would fail). Wait, let me reconsider: B (256, 64) has ndim=2, A has ndim=3. Aligning right: A is (256, 128, 64), B becomes (1, 256, 64) prepending 1. Now dimension check: dim-0: 256 vs 1 ✓ (broadcast), dim-1: 128 vs 256 ✗ (incompatible). Actually, B (256, 64) aligned right to 3D: (1, 256, 64). Comparing: 256 vs 1 ✓, 128 vs 256 ✗. This FAILS. Options A/B/C all have compatible dimensions (1s broadcast). 'Junior trap': Not understanding alignment-from-right rule. Production: Always check ndim and shapes before broadcasting to avoid runtime errors.",
            "Hard",
            180
        ),
        create_question(
            "Q10: You need to compute pairwise distances between 10000 points (each 128-dim). Using broadcasting, what is the memory requirement for the distance matrix?",
            [
                "~50 MB - stores only the distances",
                "~400 MB - distance matrix (10000, 10000) of float32",
                "~800 MB - NumPy creates intermediate expanded arrays",
                "~10 MB - sparse representation since many distances are zero"
            ],
            1,
            "Senior Explanation: Distance matrix has shape (10000, 10000). For float32: 10000² × 4 bytes = 400MB. Broadcasting approach: X (10000, 128), compute ||X[i] - X[j]||² using (X[:, np.newaxis, :] - X[np.newaxis, :, :])² shapes (10000, 1, 128) and (1, 10000, 128) broadcasting to (10000, 10000, 128), then sum over axis=2. Intermediate array: 10000×10000×128×4 = 51.2GB - HUGE! Option C understates this. Better approach: Use scipy.spatial.distance.cdist which optimizes memory via chunking. 'Junior trap': Naive broadcasting without considering intermediate memory. Production: For large-scale similarity search, use approximate methods (FAISS, Annoy) or compute in chunks. Memory explosion is a common cause of OOM in clustering/kNN algorithms.",
            "Hard",
            200
        ),
        create_question(
            "Q11: In neural network batch processing, you have weights W (512, 256) and batch input X (64, 512). For Y = X @ W, what is the FLOPs count?",
            [
                "~8 million FLOPs - one multiply-add per output element",
                "~16 million FLOPs - matrix multiplication is O(n³)",
                "~4 million FLOPs - NumPy optimizes using BLAS",
                "~32 million FLOPs - accounts for both forward and backward pass"
            ],
            0,
            "Senior Explanation: Matrix multiplication C = A @ B where A is (m, k) and B is (k, n) requires m×k×n multiply-add operations (2 FLOPs each if counted separately, but typically counted as 1 FLOP for multiply-add). Here: (64, 512) @ (512, 256) = 64×512×256 = 8,388,608 ≈ 8M FLOPs. Option B confuses O(n³) complexity for square matrices. Option C is 'junior trap' - BLAS optimizes throughput but doesn't reduce FLOPs. Option D includes backward pass (not asked). Production: Modern GPUs achieve 10-100 TFLOPs (trillion FLOPs/sec). This operation: ~8M FLOPs ÷ 20 TFLOPs = 0.4 microseconds (compute-bound). Actual time ~50-100 microseconds due to memory transfer overhead. On batch size 64×512×4 bytes = 128KB input, this is memory-bandwidth limited on GPUs.",
            "Hard",
            180
        ),

        # VIEW VS COPY (Questions 12-15)
        create_question(
            "Q12: Which of these operations returns a VIEW (not a copy)?",
            [
                "arr[arr > 0] - boolean indexing",
                "arr[[1, 3, 5]] - fancy indexing with list",
                "arr[1:10:2] - slicing with step",
                "arr.reshape(-1) when arr is not C-contiguous"
            ],
            2,
            "Senior Explanation: Basic slicing (arr[start:stop:step]) ALWAYS returns a view, regardless of step size. Option C returns a view. Option A (boolean indexing) returns a COPY because selected elements are non-contiguous. Option B (fancy indexing with arrays/lists) returns a COPY. Option D: reshape returns a view ONLY if possible without data copy (i.e., if new shape is compatible with existing strides); for non-contiguous arrays, reshape often requires a copy. 'Junior trap': Assuming all indexing operations return views. Production: Use np.shares_memory(arr, result) to check. Memory impact: On a 10GB array, copying for boolean indexing (selecting 50%) creates 5GB extra. For repeated filtering, consider np.where() with preallocated output buffers.",
            "Hard",
            180
        ),
        create_question(
            "Q13: You perform arr_copy = arr.copy(). For a 5GB array, what is the ACTUAL memory overhead considering copy-on-write optimizations?",
            [
                "~0 bytes - modern NumPy uses copy-on-write to delay copying",
                "~5 GB - copy() creates an immediate full copy",
                "~2.5 GB - partial copy optimization",
                "Depends on whether arr is modified after copying"
            ],
            1,
            "Senior Explanation: NumPy's copy() creates an IMMEDIATE full copy of the data buffer. For a 5GB array, this allocates another 5GB instantly, totaling 10GB. NumPy does NOT implement copy-on-write (unlike some languages). Option A is a 'junior trap' - confusing with Pandas (which experimented with CoW in 2.0+) or other systems. Option D is wrong - memory is allocated immediately, regardless of future modifications. Production impact: In ML training pipelines, accidental copies (e.g., arr + 0 instead of arr += 0) double memory usage, causing OOM on GPU (typical 16-40GB VRAM). Best practice: Use views when possible; profile with memory_profiler to catch hidden copies. Trade-off: Copies are safe (no shared state) but expensive; views are fast but require careful lifetime management.",
            "Hard",
            180
        ),
        create_question(
            "Q14: After creating arr_view = arr.ravel(), you modify arr_view[0] = 999. Under what condition does arr[0, 0] also become 999?",
            [
                "Always - ravel() always returns a view",
                "Only if arr is C-contiguous - ravel() returns view for contiguous arrays, copy otherwise",
                "Never - ravel() always returns a copy to prevent side effects",
                "Only if you set arr.flags.writeable = True before raveling"
            ],
            1,
            "Senior Explanation: ravel() returns a VIEW if the array is contiguous (C or Fortran order), allowing the flattened view to share memory with the original. For non-contiguous arrays (e.g., after slicing with steps), ravel() must return a COPY. Use arr.flags['C_CONTIGUOUS'] or arr.flags['F_CONTIGUOUS'] to check. Option A is 'junior trap' - assuming always view. flatten() ALWAYS returns a copy. Production debugging: Unexpected mutations can occur when ravel() returns a view. Use flatten() for safety (guaranteed copy) or ravel() for performance (potential view). Memory: On 1GB array, ravel() view = 0 bytes, flatten() = 1GB. Trade-off: Safety vs memory efficiency.",
            "Medium",
            180
        ),
        create_question(
            "Q15: You want to ensure an operation creates a copy, not a view. Which is the MOST reliable way?",
            [
                "result = arr[:] - full slicing always copies",
                "result = np.array(arr) - array constructor creates a copy",
                "result = arr.copy() - explicit copy method",
                "result = arr + 0 - arithmetic operations trigger copying"
            ],
            2,
            "Senior Explanation: arr.copy() is the MOST explicit and reliable way to create a copy, clearly signaling intent. Option A (arr[:]) creates a VIEW (all basic slicing returns views). Option B (np.array(arr)) creates a copy ONLY if arr is already a NumPy array, but it's less explicit. Option D (arr + 0) creates a copy (arithmetic creates new arrays) but is obscure and may confuse code reviewers. 'Junior trap': Thinking arr[:] copies. Production: Explicit is better than implicit (PEP 20). Using copy() prevents bugs and improves code readability. Performance: For 1GB array, copy() takes ~50-100ms (memory bandwidth limited). Only copy when necessary - use views when safe.",
            "Medium",
            150
        ),

        # VECTORIZATION & OPTIMIZATION (Questions 16-20)
        create_question(
            "Q16: You need to compute rolling mean over a 1 billion element array with window=1000. Which NumPy approach is most efficient?",
            [
                "Use np.convolve(arr, np.ones(1000)/1000, mode='valid') for optimized convolution",
                "Use as_strided to create windowed view, then np.mean(axis=1) for zero-copy efficiency",
                "Use cumsum trick: (cumsum[i] - cumsum[i-1000])/1000 for O(n) time and O(n) space",
                "Use pandas.Series(arr).rolling(1000).mean() - better optimized for rolling operations"
            ],
            2,
            "Senior Explanation: The cumsum trick is optimal: compute cumsum (O(n)), then (cumsum[i] - cumsum[i-1000])/1000. Time: O(n), Space: O(n) for cumsum array. For 1B elements (8GB float64): cumsum array = 8GB, result = 8GB, total = 24GB. Option A (convolve) is O(n log n) via FFT - slower. Option B seems elegant but np.mean() on 999M windows still requires iterating 999B elements (same time complexity, more complex code). Option D adds Python overhead. Performance: cumsum approach ~2-3s on modern CPU vs 10-15s for other methods. Production: Used in streaming feature engineering for time-series (sensor data, financial ticks). Trade-off: Requires O(n) extra space; for memory-constrained systems, consider chunked processing.",
            "Hard",
            220
        ),
        create_question(
            "Q17: For element-wise operations on a 100M element array, which achieves HIGHEST throughput on modern CPUs?",
            [
                "Python for-loop with list comprehension",
                "NumPy vectorized operation (arr * 2 + 5)",
                "np.vectorize() wrapper around Python function",
                "Numba @njit decorated function"
            ],
            1,
            "Senior Explanation: NumPy vectorized operations use optimized C/SIMD instructions achieving ~20-50 GB/s (memory bandwidth limited). For 100M float32 (400MB): ~10-20ms. Option A (Python loop): ~100-500× slower (pure Python overhead, ~10-50s). Option C (np.vectorize): still calls Python function per element - nearly as slow as loops (it's syntactic sugar, NOT performance optimization). Option D (Numba): compiles to machine code, achieves similar performance to NumPy for simple ops, but adds compilation overhead (~100-500ms first call). 'Junior trap': Using np.vectorize() for performance. Production: NumPy vectorization is optimal for standard operations; use Numba for complex custom logic. FLOPs: 100M multiplies + 100M adds = 200M FLOPs ≈ 10-20ms at 10-20 GFLOPs (CPU limited).",
            "Medium",
            180
        ),
        create_question(
            "Q18: You're applying np.exp() to a 50M element array. What is the primary performance bottleneck?",
            [
                "Memory bandwidth - reading/writing 400MB of data",
                "CPU compute - exponential is computationally expensive (50-100 CPU cycles per element)",
                "Cache misses - random memory access patterns",
                "Python interpreter overhead"
            ],
            1,
            "Senior Explanation: Exponential function (exp) requires ~50-100 CPU cycles per element (lookup tables + polynomial approximation). For 50M elements: 2.5-5B CPU cycles ≈ 1-2s at 2-3 GHz CPU. Memory bandwidth (400MB read + 400MB write = 800MB at ~50 GB/s) ≈ 16ms - much faster. Compute-bound. Option A is 'junior trap' - true for simple ops like addition (1-2 CPU cycles). Option C is wrong - sequential array access has excellent cache locality. Option D is wrong - NumPy operations are pure C, no interpreter involvement. Production: On GPUs, exp() achieves ~1-5 TFLOPs for transcendental functions, making this ~5-25ms on a V100. Trade-off: For approximate exp() (e.g., for softmax), use fast approximations (Schraudolph's method) for 5-10× speedup with <1% error.",
            "Hard",
            200
        ),
        create_question(
            "Q19: You need to normalize a 2D array (10000, 5000) by subtracting row means. Which is most efficient?",
            [
                "arr - arr.mean(axis=1).reshape(-1, 1)",
                "arr - arr.mean(axis=1)[:, np.newaxis]",
                "(arr.T - arr.mean(axis=1)).T",
                "np.subtract(arr, arr.mean(axis=1, keepdims=True), out=arr)"
            ],
            3,
            "Senior Explanation: Option D uses IN-PLACE operation (out=arr) avoiding temporary array allocation. For 10000×5000 float32 (200MB): saves 200MB. All options A/B/C create a temporary 200MB array (arr - means), then assign back. Performance: out=arr saves ~50-100ms memory allocation + garbage collection overhead. Options A/B are equivalent (both broadcast correctly). Option C transposes twice - adds 2× transpose overhead (~20-40ms each) for no benefit. 'Junior trap': Not using out parameter for in-place ops. Production: In ML preprocessing (feature normalization) on large batches, in-place ops reduce memory pressure and improve throughput by 10-20%. Trade-off: In-place ops modify original data - ensure this is acceptable or work on a copy first.",
            "Hard",
            180
        ),
        create_question(
            "Q20: For a 3D tensor (batch=256, height=224, width=224, channels=3), you need to normalize per channel. What's the optimal approach?",
            [
                "Reshape to (256*224*224, 3), normalize, reshape back",
                "Use broadcasting: arr - mean.reshape(1, 1, 1, 3)",
                "Loop over channels: for i in range(3): arr[:,:,:,i] -= mean[i]",
                "Transpose to (3, 256, 224, 224), normalize, transpose back"
            ],
            1,
            "Senior Explanation: Broadcasting (option B) is optimal - no data copying, pure vectorized operation. Mean shape (3,) broadcasts to (1, 1, 1, 3) then to (256, 224, 224, 3). Memory: input 57.8MB (256×224×224×3×4 bytes) + output 57.8MB = 115MB. Time: ~50-100ms. Option A reshapes (fast, view operation) but adds cognitive complexity - same performance. Option C is 'junior trap' - Python loop over 3 iterations, each processing 19.3MB (slower due to interpreter overhead). Option D transposes (requires data copy, ~50-100ms each direction) - adds 100-200ms overhead. Production: ImageNet preprocessing normalizes per RGB channel using this pattern. FLOPs: 38.6M subtractions. GPU acceleration: ~1-5ms on V100 (memory bandwidth limited).",
            "Medium",
            180
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_numpy()
    db.add_questions("Senior NumPy - Advanced Optimization", questions)
    print(f"✓ Successfully added {len(questions)} senior NumPy questions!")
    print(f"✓ Category: Senior NumPy - Advanced Optimization")
