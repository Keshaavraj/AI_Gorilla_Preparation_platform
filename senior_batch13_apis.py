"""
Senior AI Engineer Interview Questions - Batch 13: REST APIs & System Design for ML
Topics: Throughput vs Latency, Batching, Load Balancing, Caching, Async Serving
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_apis():
    """20 Senior-Level APIs & System Design Questions"""
    questions = [
        create_question(
            "Q1: For ML model serving with SLA of p99 latency <50ms and throughput >1000 QPS, which architecture is optimal?",
            [
                "Single GPU with large batch size - maximize throughput",
                "Dynamic batching with max_batch_size=32, timeout=10ms - balances latency and throughput",
                "Multiple small batch servers with load balancing",
                "Synchronous single-request processing"
            ],
            1,
            "Senior Explanation: Dynamic batching: Accumulate requests for max 10ms or until batch size 32 reached, whichever comes first. Process batch on GPU. Benefits: (1) Small batches (1-8) during low traffic → low latency (<20ms model + 10ms wait), (2) Full batches (32) during high traffic → high throughput. For 10ms model, batch=32: 32 requests/10ms = 3200 QPS. Single request mode: 1000 QPS max (1 request/ms). Static large batches: High latency (wait for 32 requests = 100ms+ during low traffic). Production: TF-Serving, TorchServe, Triton use dynamic batching. Trade-off: Timeout controls latency-throughput tradeoff. Lower timeout (5ms) → lower latency but lower throughput. Higher (20ms) → higher throughput but may miss p99 SLA.",
            "Hard",
            220
        ),
        create_question(
            "Q2: REST API returns embeddings (768-dim vector) for text. For 1M requests/day, what's the data transfer cost at $0.12/GB?",
            [
                "~$1 - embeddings are small",
                "~$30 - 1M × 768 × 4 bytes ≈ 3GB",
                "~$300 - includes request/response overhead",
                "~$3000 - high bandwidth usage"
            ],
            1,
            "Senior Explanation: Embedding size: 768 float32 = 3KB. 1M requests × 3KB = 3GB. At $0.12/GB: $0.36. But responses include JSON overhead (~500 bytes for headers, metadata). Total: (3KB + 0.5KB) × 1M = 3.5GB ≈ $0.42/day. Option A closest. Optimization: (1) Use fp16 (halves to 1.5KB), (2) Binary protocol vs JSON (gRPC reduces overhead to ~100 bytes), (3) Compression (gzip reduces by 30-50%). Production: At scale (100M requests/day), optimizations matter: JSON=350GB ($42), gRPC+fp16=180GB ($22). Trade-off: Development simplicity (JSON REST) vs bandwidth cost (gRPC binary). For high-volume APIs, gRPC + compression essential.",
            "Medium",
            180
        ),
        create_question(
            "Q3: For caching model outputs (input hash → output), what's the expected cache hit rate for QA system with 10K unique questions and 1M total requests?",
            [
                "~1% - questions are unique",
                "~99% - 10K unique, 1M total means 990K hits",
                "~50% - depends on distribution",
                "~10% - cache too small"
            ],
            1,
            "Senior Explanation: If 1M requests for only 10K unique questions: First 10K requests miss (fill cache), next 990K hit existing cache. Hit rate: 990K/1M = 99%. Assumes uniform distribution. Real-world: Zipf distribution (few questions very frequent). Top 1K questions might account for 80% of traffic → even higher hit rate. Cache size: 10K embeddings × 3KB = 30MB (tiny). Production: FAQ/chatbot systems see 95-99% hit rates. Memory: Redis cache 30MB vs serving 1M requests on GPU (hours of compute). ROI: Massive. Trade-off: Stale cache (model updates invalidate cache) vs cost savings. TTL (time-to-live) balances freshness (24h TTL reasonable for most models).",
            "Medium",
            180
        ),
        create_question(
            "Q4: gRPC vs REST for ML serving - what's the primary latency difference?",
            [
                "~50% faster - gRPC uses HTTP/2 and protocol buffers (binary)",
                "~10× faster - gRPC fundamentally different",
                "Same latency - network dominates",
                "REST faster - simpler protocol"
            ],
            0,
            "Senior Explanation: gRPC advantages: (1) Binary protocol buffers (vs JSON parsing ~1-5ms), (2) HTTP/2 multiplexing (reuse connection), (3) Streaming support. For 1KB payload, model latency 10ms: REST ~12ms (2ms JSON overhead), gRPC ~10.5ms (0.5ms protobuf). Speedup: ~15-20%. Larger payloads: JSON overhead grows (10KB payload: +5ms vs +1ms protobuf). Production: TF-Serving supports both - gRPC for service-to-service (latency-critical), REST for external clients (easier debugging). Trade-off: gRPC 15-30% lower latency but harder to debug (binary), requires code generation. REST simpler, human-readable but slower.",
            "Medium",
            180
        ),
        create_question(
            "Q5: Async vs sync serving for I/O-bound preprocessing (fetch from DB, resize image). Which is better?",
            [
                "Sync - simpler implementation",
                "Async - enables concurrent I/O while waiting, higher throughput with same resources",
                "No difference - I/O waits are unavoidable",
                "Sync faster - no overhead"
            ],
            1,
            "Senior Explanation: Sync: One request at a time. I/O (DB fetch 10ms) blocks thread → throughput = 1000ms/10ms = 100 QPS/thread. Async: While waiting for I/O, process other requests. 1 thread handles 10 concurrent requests → throughput ~1000 QPS/thread (10× improvement). Frameworks: FastAPI (async def endpoint), TorchServe. GPU inference: Still sync (GPU blocks during inference). Pattern: Async for I/O (fetch data, resize), sync for GPU (inference). Production: Hybrid - async endpoint, sync model call. For pure inference (no I/O), sync sufficient. Trade-off: Async complexity (event loop, await/async keywords) for I/O-bound throughput gains. CPU-bound (model inference): Async doesn't help (GIL in Python).",
            "Hard",
            200
        ),
        create_question(
            "Q6: Load balancing 3 model servers (A: RTX 3090, B: RTX 4090, C: A100). Round-robin vs weighted routing?",
            [
                "Round-robin - fair distribution",
                "Weighted by throughput - A100 gets 3×, RTX 4090 gets 2×, RTX 3090 gets 1× traffic based on relative performance",
                "Random - simplest",
                "Send all to A100 - fastest GPU"
            ],
            1,
            "Senior Explanation: Throughput capacity: RTX 3090 (~50 QPS), RTX 4090 (~100 QPS), A100 (~150 QPS). Round-robin sends 33% to each → bottleneck at RTX 3090 (overloaded), A100 underutilized. Weighted routing: Distribute proportional to capacity - 3090 gets 50/300, 4090 gets 100/300, A100 gets 150/300. All GPUs utilized equally. Total throughput: 300 QPS vs round-robin ~150 QPS (limited by slowest). Implementation: Nginx weighted upstream, AWS ALB target groups with weights. Production: Monitor GPU utilization, adjust weights. Trade-off: Weighted balancing maximizes throughput but requires capacity knowledge. Dynamic weights (based on response time) adapt to changing load.",
            "Hard",
            220
        ),
        create_question(
            "Q7: For A/B testing (90% traffic model_v1, 10% model_v2), how to route deterministically (same user always gets same model)?",
            [
                "Random 90/10 split - simple",
                "Hash user_id, route based on hash % 100 < 10 → model_v2, else model_v1 - consistent routing",
                "Time-based routing",
                "Manual assignment"
            ],
            1,
            "Senior Explanation: Consistent hashing: hash(user_id) mod 100. If result <10 → model_v2 (10%), else model_v1 (90%). Same user_id always hashes to same value → deterministic. Benefits: (1) User experience consistent (no model switching), (2) Valid A/B test (independent user groups). Random routing: User may see different models across sessions → inconsistent experience. Production: Feature flags (LaunchDarkly), API gateways (Kong) support hash-based routing. Code: if hash(user_id) % 100 < 10: model = load('v2') else: model = load('v1'). Trade-off: Deterministic routing essential for valid experiments but requires user identifier.",
            "Medium",
            180
        ),
        create_question(
            "Q8: Rate limiting for ML API - 1000 requests/min per user. Which algorithm?",
            [
                "Counter reset every minute - simple but bursty (1000 requests in first second)",
                "Token bucket - smooth rate limiting, allows bursts up to bucket size",
                "Fixed window - same as counter",
                "No limiting - trust users"
            ],
            1,
            "Senior Explanation: Token bucket: Bucket holds 1000 tokens, refills at 1000/60 = 16.67 tokens/sec. Each request consumes 1 token. Allows bursts (1000 requests instantly if bucket full) but prevents sustained overload. Counter reset: Allows 1000 requests at 12:00:59, then 1000 at 12:01:00 → 2000 in 1 second (burst). Token bucket prevents: After initial 1000, limited to ~17/sec. Production: Redis for distributed rate limiting (INCR commands). Frameworks: FastAPI slowapi, Kong rate-limiting plugin. Trade-off: Token bucket smooths traffic (prevents thundering herd) vs fixed window (simpler but bursty). Typical: bucket_size=1000, refill_rate=16.67/sec.",
            "Hard",
            200
        ),
        create_question(
            "Q9: Model served via API with average latency 50ms, p99 latency 500ms (10× worse). Likely cause?",
            [
                "Slow network occasionally",
                "Cold start / model loading for occasional requests (cache miss, container scaling)",
                "Random hardware failures",
                "User error"
            ],
            1,
            "Senior Explanation: 10× latency spike suggests cold start: (1) New container launched (model load 3-5 seconds first request), (2) Cache miss for large inputs, (3) Garbage collection pause (JVM, Python GC). For 99% requests (50ms) - warm cache/container. 1% (500ms) - cold container or GC pause. Solutions: (1) Warm-up requests (pre-load model), (2) Minimum replicas >0 (avoid scale-to-zero), (3) Model caching in memory. Production: Kubernetes HPA (horizontal pod autoscaler) scales gradually to avoid cold starts. AWS Lambda: Provisioned concurrency keeps containers warm. Trade-off: Keep extra capacity warm (cost) vs accept occasional cold start (latency spike).",
            "Hard",
            220
        ),
        create_question(
            "Q10: Embedding similarity search for 1M vectors (768-dim). Exact vs approximate search?",
            [
                "Exact search - guarantees correctness",
                "Approximate (FAISS, Annoy) - 10-100× faster, 95%+ recall sufficient for most applications",
                "Exact search faster with GPU",
                "No difference"
            ],
            1,
            "Senior Explanation: Exact search: Compare query to all 1M vectors. Time: 1M × 768 dot products = ~10-50ms (optimized). Approximate (FAISS IVFPQ): Index vectors into clusters. Search nearest clusters only (~1000 vectors). Time: ~0.5-2ms (20-100× faster). Recall: 95-99% (top-10 results, 9-10 are correct). For semantic search, 95% recall acceptable (users don't notice missing 5%). Exact needed: Legal, medical (must find all matches). Production: FAISS GPU for billion-scale search (~1-5ms for 1B vectors). Trade-off: Exact O(N) linear scan vs approximate O(log N) or O(sqrt(N)) with 5% recall loss. At scale (1B+ vectors), approximate essential.",
            "Hard",
            220
        ),
        create_question(
            "Q11: For inference, containerized model (Docker) vs serverless (AWS Lambda). When to use which?",
            [
                "Always containerized - more control",
                "Serverless for sporadic traffic (<100 requests/hour), containers for sustained load (>1000 QPS)",
                "Always serverless - auto-scaling",
                "No difference"
            ],
            1,
            "Senior Explanation: Serverless (Lambda): Pay per request, auto-scales, 15-min max runtime. Suited for: Sporadic traffic (batch processing, nightly jobs), variable load. Cons: Cold start (5-10s), limited GPU (no official GPU Lambda), 10GB max memory. Containers (ECS, Kubernetes): Persistent, GPU support, <1s startup (warm). Suited for: Real-time serving, high QPS, GPU inference. Cost comparison: 100 req/hour - Lambda $5/month, container $50/month (always running). 10,000 QPS - Lambda $5000/month (excessive invocations), container $200/month. Production: Hybrid - Lambda for preprocessing, containers for model serving. Trade-off: Serverless simplicity + auto-scale vs containers performance + cost at scale.",
            "Medium",
            180
        ),
        create_question(
            "Q12: Stateless vs stateful serving for conversational AI (chatbot with context)?",
            [
                "Stateless - each request independent, easier to scale",
                "Stateful - server maintains conversation context in memory, but limits scaling (sticky sessions required)",
                "Stateless with external state store (Redis) - best of both",
                "Stateful always better"
            ],
            2,
            "Senior Explanation: Stateless + external state: API receives conversation_id, fetches context from Redis, processes, updates Redis. Benefits: (1) Any server handles any request (easy load balancing), (2) Horizontal scaling (add servers), (3) Fault tolerance (server crash doesn't lose state). Stateful in-memory: Fast (no Redis lookup ~1ms) but requires sticky sessions (user pinned to server). Server restart = lost conversations. Production: Redis cache for conversation state. TTL = 1 hour (auto-expire inactive conversations). For 10K concurrent conversations × 10KB context = 100MB in Redis. Trade-off: Network latency to Redis (1-2ms) vs stateful complexity. At scale, external state essential for reliability.",
            "Hard",
            220
        ),
        create_question(
            "Q13: For batch inference (process 1M images overnight), synchronous API vs message queue (Kafka/RabbitMQ)?",
            [
                "Synchronous API - simpler",
                "Message queue - decouple producers (upload images) from consumers (inference workers), enables retry, monitoring",
                "No difference - both work",
                "API faster"
            ],
            1,
            "Senior Explanation: Message queue pattern: (1) Client publishes 1M messages (image IDs) to queue, (2) Workers pull messages, process (inference), publish results to output queue. Benefits: (1) Decoupling (clients don't wait for processing), (2) Retry (failed messages requeued), (3) Scaling (add workers dynamically), (4) Monitoring (queue depth shows backlog). Sync API: Client sends 1M requests, waits for responses. Connection timeouts, no retry mechanism. Production: Kafka for high-throughput batch processing (1M messages = ~10 min to publish, workers process in parallel). SQS for AWS serverless. Trade-off: Queue infrastructure complexity vs reliability and scalability for batch workloads.",
            "Hard",
            200
        ),
        create_question(
            "Q14: API versioning for ML models (v1, v2 with breaking changes). Best practice?",
            [
                "Replace v1 with v2 immediately",
                "Run both versions - route via URL path (/v1/predict, /v2/predict) or header (Accept-Version: v2)",
                "Force all users to upgrade",
                "No versioning needed"
            ],
            1,
            "Senior Explanation: Versioned endpoints: /api/v1/predict (old model), /api/v2/predict (new model). Clients opt-in to v2 when ready. Benefits: (1) Backward compatibility (v1 clients continue working), (2) Gradual migration (test v2 with subset), (3) Rollback (if v2 has issues, clients revert to v1). Alternative: Header-based (X-Model-Version: v2). Deprecation: Announce v1 sunset 3-6 months ahead, monitor v1 usage, shutdown when <5% traffic. Production: Stripe, AWS APIs use versioning. Common: Support 2-3 versions concurrently. Trade-off: Operational complexity (maintain multiple models) vs user experience (no forced breaking changes).",
            "Medium",
            180
        ),
        create_question(
            "Q15: Monitoring ML API - which metric is MOST important?",
            [
                "Request count - measure usage",
                "Latency distribution (p50, p95, p99) - ensure SLA compliance",
                "Model accuracy - measure quality",
                "CPU utilization"
            ],
            1,
            "Senior Explanation: Latency distribution critical for user experience. P50 (median) shows typical case, p99 shows worst-case (1% of users). SLA: 'p99 <100ms' means 99% requests <100ms. Single slow request acceptable; consistent slowness unacceptable. Metrics: P50=20ms (good), p95=50ms (ok), p99=500ms (investigate). Causes: GC pauses, cold starts, outlier inputs. Request count useful but doesn't show quality. Accuracy important but not real-time (needs ground truth labels later). Production: Prometheus + Grafana for latency histograms. Alert on p99 >SLA for 5+ min. Trade-off: Track all metrics but prioritize latency for operational health.",
            "Medium",
            180
        ),
        create_question(
            "Q16: Circuit breaker pattern for ML API calling external service (database). Purpose?",
            [
                "Security - prevent unauthorized access",
                "Prevent cascading failures - if database down, fail fast instead of piling up requests",
                "Load balancing",
                "Caching optimization"
            ],
            1,
            "Senior Explanation: Circuit breaker states: Closed (normal), Open (service down, reject immediately), Half-Open (test recovery). When database fails (timeouts), circuit opens after N failures (e.g., 5). Further requests fail immediately (no waiting for timeout). After cooldown (e.g., 30s), test with 1 request. If succeeds, close circuit; if fails, reopen. Benefits: (1) Fast failure (no resource wastage), (2) Service recovery time (reduce load on struggling service). Without circuit breaker: Threads blocked waiting for DB timeout (30s each) → 100 requests = 100 blocked threads → server crash. Production: Hystrix, resilience4j libraries. Trade-off: Fail fast (better than cascading failure) but requires fallback behavior (cached response, default value).",
            "Hard",
            220
        ),
        create_question(
            "Q17: For global ML API (users in US, EU, Asia), how to minimize latency?",
            [
                "Single US datacenter - centralized",
                "Multi-region deployment with geo-routing - route users to nearest datacenter (US, EU, Asia)",
                "CDN for API responses",
                "Increase bandwidth"
            ],
            1,
            "Senior Explanation: Geo-routing: Deploy in 3 regions (us-east, eu-west, ap-southeast). Route users to nearest (AWS Route53, Cloudflare load balancer). Latency: US user → US datacenter ~20ms, EU user → EU datacenter ~20ms. Single US datacenter: EU user → US ~100ms (cross-Atlantic), Asia user → US ~150ms (cross-Pacific). Model sync: Shared model in S3, each region downloads on update (eventual consistency ok for ML models). Production: Multi-region adds complexity (3× infrastructure) but critical for global <50ms latency SLA. Trade-off: Cost (3× servers) vs user experience (3-7× lower latency for non-US users).",
            "Hard",
            220
        ),
        create_question(
            "Q18: WebSocket vs HTTP for real-time inference (streaming transcription)?",
            [
                "HTTP better - simpler protocol",
                "WebSocket - persistent bidirectional connection enables streaming (client sends audio chunks, server streams transcription)",
                "No difference",
                "HTTP/2 streaming equivalent"
            ],
            1,
            "Senior Explanation: WebSocket: Persistent connection, bidirectional. Client streams audio (1 chunk/100ms), server streams transcription as it's generated (low latency). HTTP: Request-response cycle. For streaming, must use polling (client requests every 100ms) or long-polling (wasteful). HTTP/2 server push helps but less natural than WebSocket. Use case: Live transcription, real-time translation. Production: WebSocket for <100ms latency streaming. Frameworks: FastAPI supports WebSocket, TorchServe for batch. Trade-off: WebSocket complexity (connection management, reconnection) vs HTTP simplicity. For non-streaming (single request/response), HTTP sufficient.",
            "Medium",
            180
        ),
        create_question(
            "Q19: Idempotency for ML inference API - user retries request with same input. How to handle?",
            [
                "Process every request independently",
                "Use request ID - cache result for 1 hour, return cached result if same request_id seen",
                "Reject duplicates",
                "Ignore - idempotency not needed for inference"
            ],
            1,
            "Senior Explanation: Idempotent API: Same request (same request_id) returns same response, even if called multiple times. Implementation: Cache request_id → response in Redis. On request: Check cache, if hit return cached, if miss compute and cache. Benefits: (1) Network retry safe (client retries on timeout, doesn't cause duplicate processing), (2) Cost savings (don't recompute). For deterministic inference (no randomness), natural idempotency (same input = same output). For stochastic (sampling), cache essential. Production: Generate request_id on client (UUID), server uses as cache key. TTL=1 hour (balance storage vs retry window). Trade-off: Cache storage (1M requests × 3KB = 3GB) vs cost of duplicate inference.",
            "Hard",
            200
        ),
        create_question(
            "Q20: For cost optimization, spot instances (70% cheaper) vs on-demand for ML serving?",
            [
                "Always spot - cheapest",
                "Hybrid - on-demand for baseline capacity, spot for overflow traffic (auto-scaling)",
                "Always on-demand - reliability",
                "Reserved instances only"
            ],
            1,
            "Senior Explanation: Spot instances can be terminated with 2-min notice (when price spikes or capacity needed). Risk for real-time serving. Hybrid approach: Maintain 2 on-demand instances (baseline for 500 QPS), add spot instances for traffic >500 QPS (burst to 2000 QPS). Spot termination: Graceful shutdown (stop accepting new requests, finish pending, deregister from load balancer). Benefits: 70% savings on burst traffic (ephemeral). Production: Kubernetes with spot node pools + on-demand node pools. Cluster autoscaler prioritizes spot. Trade-off: 70% cost savings vs 2-min notice (acceptable for stateless serving). For training (resumable), spot very cost-effective. For latency-critical serving, use majority on-demand.",
            "Hard",
            220
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_apis()
    db.add_bulk_questions("Senior APIs - System Design for ML", questions)
    print(f"✓ Successfully added {len(questions)} senior API & System Design questions!")
    print(f"✓ Category: Senior APIs - System Design for ML")
