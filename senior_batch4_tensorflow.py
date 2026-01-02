"""
Senior AI Engineer Interview Questions - Batch 4: TensorFlow Production ML
Topics: tf.function & AutoGraph, Distributed Strategies, Custom Training Loops, TF-Serving, XLA
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_tensorflow():
    """20 Senior-Level TensorFlow Questions"""
    questions = [
        # TF.FUNCTION & AUTOGRAPH (Questions 1-4)
        create_question(
            "Q1: You decorate a function with @tf.function. On first call with shape (32, 128) input, it traces the graph. Second call with (64, 128) input causes what behavior?",
            [
                "Reuses existing graph - tf.function handles dynamic shapes automatically",
                "Retraces graph - different input shape triggers new concrete function compilation",
                "Raises error - shape mismatch with traced graph",
                "Partially retraces - only affected ops are recompiled"
            ],
            1,
            "Senior Explanation: tf.function creates concrete function per unique input signature (dtypes + shapes). Different shapes → retrace. First call (32, 128): trace + execute (~100-500ms). Second call (64, 128): retrace + execute (~100-500ms). Third call (32, 128): reuse first trace (~1-5ms). Excessive retracing causes performance degradation. Option A 'junior trap' - dynamic shapes require special handling (use None in signature or input_signature with TensorSpec). Production issue: Passing variable-length batches causes retrace every call, losing tf.function benefit. Fix: Use input_signature=[@tf.TensorSpec(shape=[None, 128], dtype=tf.float32)] to accept any batch size. Trade-off: None dimensions reduce optimization opportunities. Benchmark: Retracing overhead for large models ~500ms-2s; reuse ~1-10ms = 100-1000× speedup.",
            "Hard",
            200
        ),
        create_question(
            "Q2: In a @tf.function, you use Python print('Loss:', loss). What happens during execution?",
            [
                "Prints loss value every execution - tf.function preserves Python print",
                "Prints only during tracing (first call) - Python code runs only at trace time",
                "Converted to tf.print() automatically by AutoGraph",
                "Raises error - Python side effects not allowed in tf.function"
            ],
            1,
            "Senior Explanation: Python code in @tf.function runs ONLY during tracing (graph construction), not execution. print() executes once at trace time. To print during every execution, use tf.print(). Option A 'junior trap' - confusing eager vs graph execution. Option C wrong - AutoGraph converts control flow (if, while, for), not print(). Production debugging: Use tf.print('Loss:', loss) or print loss.numpy() outside @tf.function. Common bug: Expecting Python logging/debugging to work inside @tf.function. Code pattern: @tf.function; def train_step(): tf.print('Step loss:', loss). Trade-off: tf.print() slower than Python print (~10× overhead) but necessary for graph execution. Retrace check: Add print('TRACING') in function - if it prints every call, you're retracing excessively.",
            "Medium",
            180
        ),
        create_question(
            "Q3: You have a @tf.function with a Python list that grows each call: self.losses.append(loss). What issue occurs?",
            [
                "Memory leak - list grows unbounded across calls",
                "Graph captures list state at trace time - subsequent appends have no effect on graph execution",
                "AutoGraph converts list to TensorArray automatically",
                "Performance degrades as list grows - each append triggers retrace"
            ],
            1,
            "Senior Explanation: Python data structures (list, dict) are captured at trace time as constants. self.losses.append(loss) during tracing appends to the Python list, but the GRAPH uses the list's value at trace time. Future executions don't update the list within the graph (though the Python list in eager mode still updates - causing confusion). Option A 'junior trap' - list does grow in Python, but graph doesn't reflect it. Option D wrong - appends don't trigger retrace unless changing input signature. Production fix: Use tf.Variable or TensorArray for mutable state. Code: self.losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True); losses = losses.write(step, loss). Trade-off: TensorArray has append overhead but works correctly in graphs. Common bug: Metrics accumulated in Python lists inside @tf.function don't update correctly.",
            "Hard",
            200
        ),
        create_question(
            "Q4: When should you use experimental_relax_shapes=True in @tf.function?",
            [
                "Always - it improves performance by relaxing constraints",
                "When input shapes vary across calls - reduces retracing by allowing compatible shapes to reuse graphs",
                "Never - it's deprecated and causes errors",
                "Only for inference - not compatible with training"
            ],
            1,
            "Senior Explanation: experimental_relax_shapes=True allows shape dimensions to vary within certain bounds without retracing. Example: Traced with (32, 128), can reuse for (64, 128) if enabled. TF checks if new shape compatible with existing graph ops. Reduces retracing for variable batch sizes. Option A wrong - can reduce optimization (less shape-specific fusion). Option C wrong - not deprecated (as of TF 2.x). Production use: Variable-length sequences in NLP (batch_size varies), data pipelines with different batch sizes. Trade-off: Less retracing but potentially slower execution (fewer optimizations). Benchmark: With relax_shapes, 10 different batch sizes: 1 trace vs 10 traces (10× faster startup). Alternative: Use explicit input_signature with None dimensions. Compatibility: Works for training and inference.",
            "Medium",
            180
        ),

        # DISTRIBUTED STRATEGIES (Questions 5-9)
        create_question(
            "Q5: You're using tf.distribute.MirroredStrategy for multi-GPU training on 4 GPUs. How are gradients synchronized?",
            [
                "Asynchronously - each GPU updates independently for speed",
                "Synchronously using all-reduce (NCCL) - gradients averaged across GPUs before applying",
                "Parameter server - one GPU collects gradients, broadcasts updated weights",
                "Hierarchical - gradients aggregated in pairs, then merged"
            ],
            1,
            "Senior Explanation: MirroredStrategy uses SYNCHRONOUS all-reduce (via NCCL on GPUs) to average gradients across replicas. Each GPU computes gradients on its batch, all-reduce sums them, then divides by num_replicas. All GPUs apply identical updates (weights stay synchronized). Similar to PyTorch DDP. For 4 GPUs with 1B params (4GB gradients): all-reduce transfers ~4GB per GPU via ring-allreduce (~10-20ms on NVLink). Option A wrong - async training uses ParameterServerStrategy. Option C describes parameter server (different strategy). Production: MirroredStrategy for single-machine multi-GPU (2-8 GPUs). Achieves ~3.5-3.8× speedup on 4 GPUs (near-linear). Trade-off: Synchronous training slower than async but better convergence (no stale gradients). For multi-machine, use MultiWorkerMirroredStrategy.",
            "Hard",
            200
        ),
        create_question(
            "Q6: In distributed training with MultiWorkerMirroredStrategy across 4 machines (8 GPUs each = 32 total GPUs), what is the effective batch size if per-GPU batch is 16?",
            [
                "16 - same as per-GPU batch",
                "128 - 16 × 8 GPUs per machine",
                "512 - 16 × 32 total GPUs (global batch size)",
                "Depends on gradient accumulation steps"
            ],
            2,
            "Senior Explanation: Effective (global) batch size = per_replica_batch × num_replicas = 16 × 32 = 512. Each GPU processes 16 samples, gradients aggregated across all 32 GPUs, then optimizer steps. This is SYNCHRONOUS data parallelism. Option A 'junior trap' - per-GPU batch, not global. Option B counts only one machine. Option D - gradient accumulation would multiply further (not mentioned here). Production: Large batch training (512-4096) for Transformer models. Requires learning rate scaling: lr_new = lr_base × sqrt(global_batch / base_batch) or linear scaling. Convergence: Large batches can degrade generalization (sharp minima) - use warmup + learning rate schedules. Memory: Per GPU still only 16 samples, so VRAM usage same as single GPU. Trade-off: 32× throughput but may need hyperparameter tuning.",
            "Medium",
            180
        ),
        create_question(
            "Q7: You're using ParameterServerStrategy with 4 workers and 2 parameter servers. How do gradient updates work?",
            [
                "Synchronous - all workers send gradients to PS, PS updates, broadcasts weights",
                "Asynchronous - each worker independently pulls weights, computes gradients, pushes to PS, continues without waiting",
                "Hybrid - synchronous within PS, asynchronous across workers",
                "All-reduce - workers communicate directly without PS"
            ],
            1,
            "Senior Explanation: ParameterServerStrategy uses ASYNCHRONOUS updates. Worker workflow: (1) Pull latest weights from PS, (2) Compute gradients on local batch, (3) Push gradients to PS, (4) Immediately pull new weights and continue (no waiting for other workers). PS receives gradients from workers asynchronously, applies updates immediately. Benefit: High GPU utilization (no waiting for stragglers). Drawback: Stale gradients (worker may train on old weights), convergence issues. Option A describes synchronous PS. Option C not standard. Option D describes MirroredStrategy. Production: Used for large-scale training with many workers (100s) where synchronization overhead too high. Async training 30-50% faster but needs careful tuning (lower learning rate). Trade-off: Speed vs stability. Modern preference: Synchronous strategies (better convergence) with techniques to handle stragglers (gradient compression, backup workers).",
            "Hard",
            220
        ),
        create_question(
            "Q8: For fault tolerance in MultiWorkerMirroredStrategy, you enable checkpointing. If one worker fails, what happens?",
            [
                "Training stops - all workers must succeed",
                "Failed worker automatically restarts from last checkpoint, rejoins training",
                "Other workers continue - failed worker's data skipped",
                "Training reverts to single-worker mode"
            ],
            1,
            "Senior Explanation: With proper checkpointing (tf.train.CheckpointManager) and failure handling, failed workers can restart from last checkpoint and rejoin. TF's fault tolerance detects failure, pauses training, waits for worker recovery. Recovered worker loads checkpoint, synchronizes with cluster, resumes. Requires: (1) Persistent checkpoint storage (shared filesystem, GCS), (2) Cluster manager (Kubernetes) to restart failed pods. Option A 'junior trap' - without fault tolerance, yes. Option C wrong - distributed training needs all workers (synchronous). Production setup: On GKE, use preemptible VMs (80% cost savings), automatic restart on failure. Average recovery time ~2-5 minutes. Trade-off: Checkpoint frequency (every N steps) - too frequent slows training (I/O overhead), too rare loses more progress on failure. Typical: Checkpoint every 1000-5000 steps (~10-30 min intervals).",
            "Hard",
            200
        ),
        create_question(
            "Q9: You're training on TPU v4 pods (256 chips) using TPUStrategy. What is the primary communication mechanism for gradient synchronization?",
            [
                "NCCL - same as GPU training",
                "Custom TPU interconnect with 2D torus topology - much faster than PCIe/NVLink",
                "Parameter servers - each TPU chip communicates with central servers",
                "MPI - standard distributed computing protocol"
            ],
            1,
            "Senior Explanation: TPU pods use custom high-bandwidth interconnect (ICI - Inter-Chip Interconnect) with 2D/3D torus topology. TPU v4: ~4.8 TBps bisection bandwidth vs NVLink 3.0 ~600 GBps = 8× faster. Enables near-linear scaling to 100s-1000s of chips. All-reduce on TPU: Uses topology-aware algorithms optimized for torus (different from ring-allreduce on GPUs). Option A wrong - NCCL is NVIDIA-specific. Option C wrong - TPUs use all-reduce, not PS. Production: Training largest models (PaLM 540B, GPT-4) on TPU pods with thousands of chips. Scaling efficiency: ~90%+ on 1024 chips vs ~70-80% on GPUs (communication overhead). Trade-off: TPUs have better scaling but less flexible than GPUs (optimized for dense matrix ops, Transformers). Cost: TPU pods expensive but higher throughput per dollar for large-scale training.",
            "Hard",
            220
        ),

        # CUSTOM TRAINING LOOPS (Questions 10-13)
        create_question(
            "Q10: In a custom training loop, you call loss.backward() equivalent (tf.GradientTape). Where should the tape context be?",
            [
                "Persistent tape created once in __init__, reused across steps",
                "New tape created each training step - tape records operations within context, then computes gradients",
                "Global tape - one for entire training session",
                "Tape only needed for validation, not training"
            ],
            1,
            "Senior Explanation: GradientTape must be created fresh each training step. Pattern: with tf.GradientTape() as tape: loss = model(x); gradients = tape.gradient(loss, model.trainable_variables). Tape records operations (forward pass) in its context, then computes gradients via reverse-mode AD. After gradient() call, tape is destroyed (unless persistent=True). Option A wrong - persistent tapes have overhead and cause memory leaks if not deleted. Option C/D wrong. Production code: @tf.function; def train_step(x, y): with tf.GradientTape() as tape: predictions = model(x); loss = loss_fn(y, predictions); gradients = tape.gradient(loss, model.trainable_variables); optimizer.apply_gradients(zip(gradients, model.trainable_variables)). Trade-off: Non-persistent tape minimal overhead; persistent tape allows multiple gradient() calls but needs manual del tape.",
            "Medium",
            180
        ),
        create_question(
            "Q11: You implement a custom training loop with tf.function. You want to update a metric (accuracy) each step. Best approach?",
            [
                "Use Python variable: self.accuracy += batch_accuracy - simple accumulation",
                "Use tf.Variable: self.accuracy.assign_add(batch_accuracy) - graph-compatible mutable state",
                "Use tf.py_function to call Python code for metric update",
                "Return metric from tf.function, accumulate outside in Python"
            ],
            1,
            "Senior Explanation: tf.Variable is the correct way to maintain mutable state in tf.function. Python variables captured at trace time (constants in graph). Use self.accuracy = tf.Variable(0.0); then self.accuracy.assign_add(batch_acc) inside @tf.function. Option A 'junior trap' - Python variable won't update in graph execution. Option C (py_function) works but breaks graph optimization and runs in Python (slow). Option D works but requires returning values from tf.function (memory overhead for large metrics). Production pattern: Use Keras metrics (inherit tf.keras.metrics.Metric) which handle tf.Variable state internally. Trade-off: tf.Variable has overhead (~100 bytes + update op) but necessary for correctness. For thousands of metrics, consider batching updates or using tf.TensorArray.",
            "Medium",
            180
        ),
        create_question(
            "Q12: In a custom training loop, you process 10M samples in 10K steps. You want to log loss every 100 steps. Inside @tf.function train_step, how should you log?",
            [
                "if step % 100 == 0: log(loss) - standard Python conditional",
                "Use tf.cond(tf.equal(step % 100, 0), lambda: log(loss), lambda: None) - graph-compatible conditional",
                "Log every step inside @tf.function, filter outside in Python",
                "Use @tf.function with autograph=False to preserve Python control flow"
            ],
            2,
            "Senior Explanation: Python conditionals (if) in @tf.function are traced once, becoming static in the graph (doesn't evaluate at runtime). If step=0 at trace, graph always logs (or never logs). Option B (tf.cond) works but adds complexity. BEST: Log OUTSIDE @tf.function. Pattern: @tf.function; def train_step(x, y): ...; return loss; Outside: for step in range(10000): loss = train_step(x, y); if step % 100 == 0: log(loss.numpy()). Option A 'junior trap' - Python if doesn't work as expected. Option D wrong - autograph=False disables AutoGraph conversion but doesn't make Python if dynamic. Production: Return tensors from @tf.function, handle logging/checkpointing in eager mode (outside). Trade-off: Returning loss adds minimal overhead (~4 bytes per step); logging inside graph with tf.cond adds graph complexity.",
            "Hard",
            200
        ),
        create_question(
            "Q13: You implement gradient clipping in a custom loop. Which is more efficient: clip by value or clip by norm?",
            [
                "Clip by value (tf.clip_by_value) - simpler operation",
                "Clip by norm (tf.clip_by_global_norm) - prevents gradient explosion better with less impact on optimization",
                "Both identical performance-wise",
                "Depends on model size"
            ],
            1,
            "Senior Explanation: Clip by global norm is standard for deep learning: gradients = [tape.gradient(loss, var) for var]; clipped_grads, global_norm = tf.clip_by_global_norm(gradients, clip_norm=1.0). Computes total L2 norm of all gradients, scales if exceeds threshold. Preserves gradient direction (important for optimization). Clip by value: tf.clip_by_value(grad, -1, 1) clips each element independently, changes direction. Performance: Clip by norm adds one extra pass to compute norm (~1-2ms for 1B params), but optimization benefit is huge (stable training, especially RNNs/Transformers). Option A 'junior trap' - by_value is NOT standard practice. Production: Nearly all Transformer training uses clip_by_global_norm with clip_norm=1.0. Trade-off: Tiny compute overhead for much better convergence. Gradient explosion detection: Monitor global_norm; if suddenly spikes (1000×), indicates instability.",
            "Medium",
            180
        ),

        # TF-SERVING & OPTIMIZATION (Questions 14-17)
        create_question(
            "Q14: You export a model for TF Serving using tf.saved_model.save(). What is the inference latency overhead of SavedModel vs in-process Python?",
            [
                "~50-100ms - SavedModel loading overhead per request",
                "~1-5ms - minor serialization overhead for gRPC communication",
                "~100-500ms - model needs to reload each request",
                "Negligible (<0.1ms) - SavedModel compiled to same graph as in-process"
            ],
            1,
            "Senior Explanation: TF Serving loads SavedModel once at startup (~1-10s), then serves requests from memory. Per-request overhead: gRPC serialization/deserialization of inputs/outputs (~0.5-2ms) + any model-specific overhead. For a 100ms model inference: SavedModel ~101-102ms vs in-process ~100ms (1-2% overhead). Option A 'junior trap' - confusing loading time with per-request overhead. Option C wrong - model loaded once. Production: TF Serving achieves ~1000-10000 QPS for small models (10ms latency), ~10-100 QPS for large models (100ms latency) on single GPU. Batching improves throughput: Batch 32 requests → ~3× higher QPS. Trade-off: Small latency overhead for massive scalability (horizontal scaling, versioning, monitoring). REST API has ~2-5× higher latency than gRPC due to JSON overhead.",
            "Hard",
            200
        ),
        create_question(
            "Q15: For a production model serving 1000 QPS with p99 latency requirement of 50ms, which TF Serving optimization is MOST effective?",
            [
                "Enable batching with max_batch_size=32, batch_timeout_micros=5000 - amortizes fixed costs",
                "Use multiple model versions for A/B testing",
                "Increase num_load_threads for faster model loading",
                "Enable model warmup to preload weights"
            ],
            0,
            "Senior Explanation: Batching is the #1 optimization for throughput. Example: Model latency 10ms single, 20ms batch-32 → single=100 QPS, batched=1600 QPS (16× improvement). max_batch_size=32, batch_timeout_micros=5000 means: Wait up to 5ms to accumulate 32 requests, then process together. Trade-off: Adds up to 5ms latency (batch timeout) but increases throughput massively. For 1000 QPS requirement: Without batching, need ~10 GPUs (100 QPS each); with batching ~1-2 GPUs (1000-2000 QPS). p99 latency: Model latency (20ms) + batch timeout (5ms) + queuing (~10-20ms) ≈ 35-45ms (meets 50ms SLA). Option B/C/D improve other aspects, not throughput/latency. Production: Always enable batching for high-throughput serving. Cost savings: 5-10× fewer GPUs. Monitoring: Track batch_size distribution (ensure batches filling up).",
            "Hard",
            220
        ),
        create_question(
            "Q16: You have a SavedModel with multiple signatures (prediction, preprocessing, postprocessing). Which signature is invoked by default in TF Serving?",
            [
                "All signatures executed in sequence",
                "The signature named 'serving_default' - TF Serving convention",
                "First signature alphabetically",
                "Must specify signature in each request - no default"
            ],
            1,
            "Senior Explanation: TF Serving uses 'serving_default' signature by default if no signature specified in request. When saving: tf.saved_model.save(model, path, signatures={'serving_default': model_fn, 'preprocessing': preprocess_fn}). In REST API: POST /v1/models/mymodel:predict (uses serving_default). To specify: POST /v1/models/mymodel/versions/1:predict with signature_name='preprocessing'. Option A wrong - one signature per request. Option C/D wrong. Production pattern: serving_default for main inference, additional signatures for debugging (intermediate outputs) or multi-stage pipelines. Trade-off: Multiple signatures increase model size (different graphs) but improve flexibility. Typical SavedModel: 1-3 signatures. Large models: Keep single signature to minimize size.",
            "Medium",
            180
        ),
        create_question(
            "Q17: For a TF Serving model receiving variable-length sequences, how should you handle padding for batching?",
            [
                "Pad all sequences to max_length (e.g., 512) before sending - ensures uniform shape",
                "Send variable-length sequences - TF Serving automatically pads to longest in batch",
                "Disable batching for variable-length inputs",
                "Use ragged tensors in SavedModel signature"
            ],
            0,
            "Senior Explanation: TF Serving requires fixed shapes for batching (all inputs in batch must have same shape). For variable-length sequences: CLIENT must pad to fixed length before sending (e.g., pad to 512 tokens). Model should handle padding (e.g., attention mask). Option B 'junior trap' - TF Serving does NOT auto-pad (raises shape mismatch error). Option C defeats batching benefit. Option D - ragged tensors supported but complicate client code (must send ragged representation). Production pattern: Pad to max_length on client, send attention_mask to indicate real vs padding tokens. Trade-off: Over-padding (all to max_length=512 even if max in batch is 100) wastes compute (~5× FLOPs for 100 vs 512). Advanced: Bucketing - multiple model endpoints with different max_lengths (128, 256, 512), route based on sequence length. Cost: 3× models but 2-5× better GPU utilization.",
            "Hard",
            200
        ),

        # XLA COMPILATION (Questions 18-20)
        create_question(
            "Q18: You enable XLA compilation with @tf.function(jit_compile=True). What is the PRIMARY performance benefit?",
            [
                "Reduces Python overhead - compiles Python to C++",
                "Fuses operations (e.g., bias_add + relu) into single kernels, reduces memory traffic and kernel launch overhead",
                "Enables automatic multi-GPU distribution",
                "Compresses model weights for faster loading"
            ],
            1,
            "Senior Explanation: XLA (Accelerated Linear Algebra) performs whole-program optimization, fusing multiple ops into optimized kernels. Example: x = relu(matmul(A, B) + bias) → normally 3 kernels (matmul, add, relu) with 3 memory reads/writes. XLA fuses to 1 kernel with 1 memory write. For Transformer layer (~100 ops): XLA reduces to ~20 fused kernels. Benefit: ~10-30% speedup for compute-bound models via reduced memory traffic (memory bandwidth is often bottleneck). Option A wrong - XLA is graph-level compiler, not Python. Option C/D wrong. Production: XLA especially effective for TPUs (built for XLA) and for models with many small ops (Transformers). Trade-off: Compilation overhead (~5-30s first run, then cached) - only beneficial for training/repeated inference. Benchmark: ResNet-50 training with XLA: ~20% faster. BERT training: ~30% faster.",
            "Hard",
            200
        ),
        create_question(
            "Q19: When should you NOT use XLA compilation?",
            [
                "For models with dynamic control flow (tf.cond, tf.while_loop with data-dependent conditions)",
                "For small models - XLA overhead dominates",
                "For inference - XLA only benefits training",
                "Never - XLA always improves performance"
            ],
            0,
            "Senior Explanation: XLA struggles with dynamic control flow where condition depends on tensor values (data-dependent). Example: tf.while_loop with condition on computed values. XLA must unroll or use conservative bounds (inefficient). Option A is primary limitation. Option B has some truth - for tiny models (<1M params) or very short sequences, XLA compilation overhead (~100-500ms) may exceed runtime savings (if runtime <100ms). But not the MAIN reason. Option C wrong - XLA benefits both. Option D wrong. Production: Use XLA for standard architectures (ResNet, Transformer) without complex dynamic behavior. Avoid for RNNs with variable-length loops, dynamic networks (NAS), or models with heavy Python logic. Trade-off: XLA trades compilation time for runtime performance. For constantly-changing model shapes (e.g., research experiments), compilation overhead may outweigh benefits.",
            "Hard",
            200
        ),
        create_question(
            "Q20: You enable mixed precision (policy=mixed_float16) and XLA. What precision are matmul operations computed in?",
            [
                "FP16 - matches policy precision",
                "FP32 - XLA always uses full precision for accuracy",
                "TF32 on Ampere GPUs - automatic hardware precision",
                "FP16 for forward pass, FP32 for backward pass"
            ],
            2,
            "Senior Explanation: On NVIDIA Ampere+ GPUs (A100, RTX 30xx), TensorFlow automatically uses TF32 (TensorFloat-32) for FP32 matmuls by default. TF32: 19-bit precision (vs FP32 23-bit mantissa), 8-bit exponent (same as FP32), runs on Tensor Cores at ~8× FP32 speed. With mixed_float16 policy + XLA on Ampere: Inputs/outputs FP16, computation TF32 (for ops that support it). No accuracy loss vs FP32, ~50% of BF16/FP16 performance. Option A - true for Tensor Core matmuls (if inputs FP16). Option B wrong. Option D wrong - both passes use same precision policy. Production: On A100, default TF32 gives ~3-5× speedup vs FP32 with zero code changes. Disable with tf.config.experimental.enable_tensor_float_32_execution(False) if exact FP32 needed. Trade-off: Tiny precision loss (rarely matters) for huge speedup. Combine with mixed_float16 for maximum performance (~10× vs FP32).",
            "Hard",
            220
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_tensorflow()
    db.add_questions("Senior TensorFlow - Production ML", questions)
    print(f"✓ Successfully added {len(questions)} senior TensorFlow questions!")
    print(f"✓ Category: Senior TensorFlow - Production ML")
