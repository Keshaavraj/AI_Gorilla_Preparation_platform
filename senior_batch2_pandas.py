"""
Senior AI Engineer Interview Questions - Batch 2: Pandas Production Optimization
Topics: Memory Management, Vectorization, Chunking, Query/Eval, Merge Optimization
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_pandas():
    """20 Senior-Level Pandas Questions"""
    questions = [
        # MEMORY MANAGEMENT & DTYPE OPTIMIZATION (Questions 1-5)
        create_question(
            "Q1: You have a Pandas DataFrame with 100M rows and 50 columns (mix of int64, float64, object). What is the FIRST step to reduce memory footprint in production?",
            [
                "Convert int64 columns to int32/int16 based on value range; use category dtype for low-cardinality objects",
                "Drop all rows with missing values to reduce DataFrame size",
                "Use df.memory_usage(deep=True) to analyze, then compress with pickle+gzip",
                "Switch to sparse DataFrame representation for all columns"
            ],
            0,
            "Senior Explanation: Downcasting dtypes is most impactful. int64 (8 bytes) → int16 (2 bytes) = 75% reduction. For a 1M-cardinality string column stored as 'object', converting to 'category' reduces memory from ~50 bytes/string × 100M = 5GB to ~4 bytes/row (category code) + 1M × 50 bytes (category data) = 0.45GB (~91% reduction). Option B 'junior trap' destroys data. Option C only helps disk storage, not in-memory ops. Option D (sparse) only helps if >90% values are zeros/NaNs. Production impact: EC2 r5.2xlarge (64GB RAM) can process 100M rows instead of 20M after optimization. Code: df.select_dtypes(['int64']).apply(pd.to_numeric, downcast='integer'). Memory profiling: Use df.memory_usage(deep=True).sum() / 1e9 for GB estimate.",
            "Hard",
            180
        ),
        create_question(
            "Q2: A DataFrame has a high-cardinality string column 'user_id' (50M unique values, 100M rows). What dtype optimization reduces memory most?",
            [
                "Convert to 'category' dtype - categories reduce memory for repeated values",
                "Keep as 'object' but use string interning with sys.intern()",
                "Convert to hash codes using df['user_id'].apply(hash), drop original",
                "Use pd.StringDtype() - more memory-efficient than object"
            ],
            2,
            "Senior Explanation: With 50M unique values (50% unique), 'category' is INEFFICIENT. Category memory: 50M categories × ~30 bytes = 1.5GB + 100M × 8 bytes (codes) = 2.3GB. Original 'object': 100M × ~30 bytes = 3GB. Savings: only 23%. Better: Hash codes produce int64 (8 bytes) = 100M × 8 bytes = 0.8GB (73% reduction). Option A 'junior trap' - works for LOW cardinality (<1% unique). Option B doesn't work for Pandas Series. Option D (StringDtype) ≈ same memory as object. Trade-off: Hashing loses original values (can't reverse) and has collision risk (~1 in 2^64). Production use: Anonymizing user IDs for ML where only aggregates matter. Alternative: Store mapping separately if reversibility needed.",
            "Hard",
            200
        ),
        create_question(
            "Q3: You're loading a 50GB CSV (500M rows). Which Pandas strategy is most memory-efficient for computing aggregated statistics?",
            [
                "pd.read_csv('file.csv') with dtype specification - Pandas handles large files efficiently",
                "Use chunksize: for chunk in pd.read_csv('file.csv', chunksize=1e6); accumulate stats incrementally",
                "pd.read_csv('file.csv', low_memory=False) to avoid dtype warnings",
                "Use Dask: dask.dataframe.read_csv('file.csv').compute() for distributed processing"
            ],
            1,
            "Senior Explanation: Chunking processes file in fixed increments (e.g., 1M rows = ~100MB chunks), keeping peak memory constant (~100MB vs 50GB). For aggregations (sum, mean, count), maintain running totals across chunks - O(1) extra space. Option A 'junior trap' loads 50GB into RAM (OOM). Option C same as A. Option D (Dask) is overkill for single-machine stats - adds ~100ms scheduling overhead. Dask shines for multi-machine clusters. Production pattern: Daily log processing on m5.xlarge (16GB RAM). Code: chunks = pd.read_csv(..., chunksize=1e6); total = sum(chunk['col'].sum() for chunk in chunks). Trade-off: Chunking slower (sequential I/O) but enables processing datasets 100× larger than RAM. Throughput: ~100-200 MB/s for CSV parsing.",
            "Hard",
            200
        ),
        create_question(
            "Q4: For a 100M row DataFrame with dtype object columns containing numbers as strings, what's the impact of converting to numeric?",
            [
                "Minimal impact - Pandas automatically optimizes object dtype",
                "Memory reduction of ~50% and 10-100× faster arithmetic operations",
                "Slower operations due to conversion overhead",
                "Only beneficial for columns with <1000 unique values"
            ],
            1,
            "Senior Explanation: Object dtype stores Python objects (huge overhead). String '12345' as object: ~50 bytes (PyObject overhead + string data). As int64: 8 bytes (85% reduction). For 100M rows: object = 5GB, int64 = 0.8GB. Arithmetic: Vectorized int64 operations use SIMD (~20 GB/s throughput), while object dtype invokes Python's __add__ per element (~0.2 GB/s) = 100× slower. Option A 'junior trap' - no such auto-optimization. Conversion: pd.to_numeric(df['col'], errors='coerce') handles mixed types. Production: CSV files often load numbers as strings; conversion is critical for performance. Trade-off: Conversion time ~5-10s for 100M rows, but operations become 100× faster. For repeated operations, always convert numeric object columns.",
            "Hard",
            180
        ),
        create_question(
            "Q5: You need to store a sparse binary matrix (100M rows × 10K columns, 99.9% zeros) as DataFrame. Best approach?",
            [
                "Regular DataFrame with int8 dtype",
                "Sparse DataFrame with fill_value=0",
                "Store as dict of arrays for non-zero columns",
                "Use scipy.sparse.csr_matrix instead of Pandas"
            ],
            3,
            "Senior Explanation: Sparse DataFrame overhead is high for very sparse data (99.9%). Dense int8: 100M × 10K × 1 byte = 1TB - impossible. Pandas Sparse: stores only non-zero values + indices. For 0.1% non-zero: ~1B values × (8 bytes value + 8 bytes index) = 16GB. scipy.sparse.csr_matrix (Compressed Sparse Row): stores ~1B values + ~1B column indices + ~100M row pointers = ~5-6GB (3× better). Option A 'junior trap' - assumes feasible. Option C (dict) works but less efficient than CSR. Production: Recommender systems (user-item matrices), NLP (document-term matrices) use scipy.sparse. Trade-off: scipy.sparse lacks Pandas operations; convert to Pandas only for final analysis. Most ML libraries (sklearn) accept sparse matrices directly.",
            "Hard",
            200
        ),

        # VECTORIZATION (Questions 6-9)
        create_question(
            "Q6: You need to apply custom transformation to a 10M row DataFrame column. Which approach is fastest?",
            [
                "df['new_col'] = df['col'].apply(lambda x: custom_function(x))",
                "df['new_col'] = df['col'].map(custom_function)",
                "Vectorize with NumPy: df['new_col'] = custom_function_vectorized(df['col'].values)",
                "Use df.eval('new_col = custom_function(col)') for optimized evaluation"
            ],
            2,
            "Senior Explanation: Vectorized NumPy operations avoid Python loops, using compiled C/SIMD. Speed hierarchy: Vectorized NumPy (~50-200 MB/s) > apply (~5 MB/s) > Python loops (~0.5 MB/s). For 10M rows: Vectorized (~1-2s) vs apply (~20-30s) = 15-30× faster. Option A/B 'junior trap' - apply/map invoke Python function per row (interpreter overhead). Option D only works for simple expressions, not custom functions. Production pattern: If custom_function involves math (exp, log, trig), rewrite using NumPy ufuncs. Example: df['log_ratio'] = np.log(df['A'].values / df['B'].values) instead of apply(lambda row: math.log(row['A']/row['B'])). Trade-off: Vectorization requires eliminating conditionals (use np.where, np.select).",
            "Hard",
            180
        ),
        create_question(
            "Q7: For element-wise string operation on 50M rows (e.g., str.lower()), what's the performance bottleneck?",
            [
                "Memory bandwidth - reading/writing string data",
                "Python interpreter overhead - each string is a Python object",
                "CPU compute - string operations are expensive",
                "GIL (Global Interpreter Lock) contention"
            ],
            1,
            "Senior Explanation: Pandas string operations (str.lower(), str.replace()) iterate in Python, invoking Python's string methods per element. Each operation has PyObject overhead (~50-100ns per call). For 50M strings: ~2.5-5s just in overhead. Actual work (lower case conversion) is fast (~10-20ns per char). Option A is wrong - memory I/O is fast for sequential access. Option C underestimates - string ops are simple. Option D (GIL) only matters for multi-threading (Pandas str ops are single-threaded). Performance: ~5-10 MB/s for str ops vs ~50-200 MB/s for numeric. 'Junior trap': Expecting vectorized performance for string ops. Production: For huge datasets, consider: (1) Cython/Numba for custom string ops, (2) PyArrow strings (faster), (3) regex with compiled patterns. Trade-off: PyArrow strings 2-5× faster but less compatible.",
            "Hard",
            200
        ),
        create_question(
            "Q8: You need to compute df['C'] = df['A'] / df['B'] where B may contain zeros. Most efficient safe approach?",
            [
                "df['C'] = df['A'] / df['B'].replace(0, np.nan)",
                "df['C'] = df.apply(lambda row: row['A'] / row['B'] if row['B'] != 0 else np.nan, axis=1)",
                "df['C'] = np.where(df['B'] != 0, df['A'] / df['B'], np.nan)",
                "df['C'] = df['A'].divide(df['B'], fill_value=np.nan)"
            ],
            2,
            "Senior Explanation: np.where() is vectorized and efficient: evaluates condition (df['B'] != 0) as boolean array, then selects values from df['A']/df['B'] or np.nan accordingly - all in vectorized NumPy. For 10M rows: ~20-50ms. Option A replaces zeros first (~10ms) then divides (~10ms) - similar speed but less explicit about intent. Option B 'junior trap' - apply with axis=1 is SLOW (~10-30s), iterates Python functions per row. Option D uses Pandas divide() which handles division by zero, but fill_value is for missing values, not division by zero (it still raises warning/inf). Production: np.where() is the standard pattern for conditional vectorized operations. Alternative: Use df.eval('C = where(B != 0, A / B, nan)') for similar performance with cleaner syntax.",
            "Medium",
            180
        ),
        create_question(
            "Q9: You're parsing a 'timestamp' column (100M rows, string format) to datetime. Most efficient approach?",
            [
                "pd.to_datetime(df['timestamp']) - Pandas infers format automatically",
                "pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S') - explicit format",
                "df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))",
                "pd.to_datetime(df['timestamp'], infer_datetime_format=True) for faster inference"
            ],
            1,
            "Senior Explanation: Explicit format parameter enables Pandas to use optimized C-level strptime parsing without per-row inference. Speed: explicit format (~10-20s for 100M rows) vs auto-inference (~60-120s) = 5-10× faster. Option A 'junior trap' - infers format per unique pattern, trying multiple formats. Option C catastrophic (~500-1000s) - pure Python loop. Option D (infer_datetime_format=True) infers once then applies, faster than A but slower than explicit. Memory: all approaches ~800MB for datetime64[ns]. Production: Streaming log pipelines (Apache logs with known format) achieve ~5-10M rows/sec/core. Trade-off: Explicit format requires knowing format upfront; for mixed formats use errors='coerce' to handle gracefully. Benchmark on 100M rows: explicit=15s, infer_once=45s, auto=120s, apply=800s.",
            "Medium",
            150
        ),

        # QUERY & EVAL OPTIMIZATION (Questions 10-12)
        create_question(
            "Q10: You're filtering a 100M row DataFrame with multiple conditions. Which minimizes memory and time?",
            [
                "df_filtered = df[(df['A'] > 10) & (df['B'] == 'X') & (df['C'] < 100)]",
                "Use df.query(\"A > 10 and B == 'X' and C < 100\") for optimized filtering",
                "Filter sequentially: df = df[df['A']>10]; df = df[df['B']=='X']; df = df[df['C']<100]",
                "Convert to NumPy: mask = (df['A'].values > 10) & ...; df_filtered = df.iloc[mask]"
            ],
            1,
            "Senior Explanation: df.query() uses numexpr under the hood - multi-threaded vectorized evaluation in a single pass, reusing CPU cache efficiently. For 100M rows: query (~2-3s using 4 cores) vs boolean indexing (~5-6s, single-threaded). Memory: query evaluates expression without creating intermediate boolean arrays for each condition (saves 100M × 3 × 1 byte = 300MB). Option A 'junior trap' creates 3 separate 100MB boolean arrays. Option C worst - creates 3 intermediate DataFrames (potentially hundreds of GB). Option D equivalent to A. Production: query() shines with 5+ conditions on m5.2xlarge (8 cores). Trade-off: query() requires string syntax (less IDE autocomplete); complex conditions may need df.eval() or fallback. Numexpr achieves ~2-3× speedup via multi-threading + optimized expression trees.",
            "Hard",
            180
        ),
        create_question(
            "Q11: For complex column computations on 50M rows, when is df.eval() FASTER than direct assignment?",
            [
                "Always - eval() is always faster due to numexpr optimization",
                "For expressions with 3+ operations (e.g., 'A + B * C - D / E') - avoids intermediate arrays",
                "Never - direct assignment is always faster",
                "Only when columns are already in memory (not loaded from disk)"
            ],
            1,
            "Senior Explanation: df.eval() uses numexpr which evaluates multi-operation expressions in a single pass without creating intermediate arrays. Example: df.eval('X = A + B * C') computes in one pass vs df['X'] = df['A'] + df['B'] * df['C'] creates intermediate (B * C), then (A + intermediate). For 50M rows float64: saves 400MB per intermediate. Speedup: ~30-50% for 3+ operations. Option A 'junior trap' - eval() has parsing overhead (~1-5ms); for simple single-op expressions (df['X'] = df['A'] + 1), direct assignment is equivalent or faster. Option C ignores numexpr benefits. Production: In feature engineering with complex derived features, eval() reduces memory pressure and improves cache efficiency. Trade-off: String syntax less maintainable; use for hot paths only. Benchmark: 'A+B+C+D+E' on 50M rows: eval()=200ms, direct=350ms.",
            "Hard",
            200
        ),
        create_question(
            "Q12: What is the PRIMARY advantage of df.query() over boolean indexing for filtering?",
            [
                "Faster syntax - less typing required",
                "Multi-threaded execution via numexpr + reduced memory for intermediate boolean arrays",
                "Better handling of missing values",
                "Automatic dtype optimization during filtering"
            ],
            1,
            "Senior Explanation: df.query() leverages numexpr for: (1) Multi-threading (uses all CPU cores for expression evaluation), (2) Memory efficiency (evaluates expression tree without allocating full intermediate boolean arrays for each subexpression). For df.query('A > 10 & B < 20 & C == 5') on 100M rows with 8 cores: query ~2s vs boolean indexing ~6s. Memory: boolean indexing allocates 3 × 100M bytes = 300MB for intermediate masks; query uses ~0 extra (evaluates on-the-fly). Option A 'junior trap' - syntax is nice but not the main benefit. Option C/D are wrong - no such optimizations. Production: On high-core-count instances (c5.9xlarge with 36 cores), query() scales nearly linearly for complex filters. Trade-off: Requires numexpr dependency; single-threaded environments see less benefit.",
            "Hard",
            180
        ),

        # MERGE & JOIN OPTIMIZATION (Questions 13-16)
        create_question(
            "Q13: In production time-series pipeline, merge operations on 20M rows take 10 minutes. Most effective optimization?",
            [
                "Use df.merge(..., sort=False) to disable sorting",
                "Set join keys as indexes using set_index before merge; use df1.join(df2)",
                "Increase RAM allocated to Pandas using pd.options.compute.memory_limit",
                "Switch to outer join instead of inner join"
            ],
            1,
            "Senior Explanation: Setting join keys as indexes (df.set_index('key')) enables hash-based or sorted-index joins, reducing complexity from O(n×m) to O(n+m) or O(n log n). For 20M × 20M merge: naive nested loop = 400T comparisons (infeasible), hash join = 40M operations (~100× faster). Option A helps marginally (saves O(n log n) sort at end) but doesn't address core bottleneck. Option C 'junior trap' - no such Pandas option. Option D (outer join) is SLOWER (more data). Benchmark: 20M row merge on indexed keys: ~5-10s vs non-indexed: ~10-15 minutes. Production: In feature engineering joining user events with metadata, pre-indexing reference tables (users, products) at load time saves hours daily. Trade-off: set_index requires O(n) time and extra memory for index, but pays off after 2-3 merges.",
            "Hard",
            200
        ),
        create_question(
            "Q14: You're merging two DataFrames: left (100M rows) and right (10K rows) on 'key'. What merge strategy is optimal?",
            [
                "df_left.merge(df_right, on='key', how='left')",
                "df_right.merge(df_left, on='key', how='right')",
                "Pre-sort both by 'key', then use merge with sort=True",
                "Convert right to dict, use df_left['key'].map(right_dict)"
            ],
            3,
            "Senior Explanation: For extremely skewed sizes (100M vs 10K), converting smaller DataFrame to dict eliminates merge overhead. Create dict: right_dict = df_right.set_index('key')['value'].to_dict() (~1ms for 10K rows). Map: df_left['new_col'] = df_left['key'].map(right_dict) (~5-10s for 100M rows). Total: ~10s. Standard merge: ~30-60s (builds hash table for both sides, more overhead). Option A/B 'junior trap' - equivalent performance for standard merge. Option C (sorting) doesn't help for hash-based merge. Production: Feature enrichment where broadcasting small lookup tables (country codes, product categories) to large event streams. Trade-off: map() only transfers one column; for multiple columns, merge is cleaner. Memory: dict ~10K × 100 bytes = 1MB vs merge overhead ~hundreds of MB.",
            "Hard",
            200
        ),
        create_question(
            "Q15: You need to join two 100GB CSV files (left: 500M rows, right: 100M rows) on 'user_id'. Memory: 16GB. Most viable approach?",
            [
                "Load both into Pandas with dtype optimization and merge",
                "Use chunking: load right fully, iterate left in chunks, merge each chunk",
                "Use Dask: dd.read_csv for both, then merge().compute()",
                "Pre-sort both files externally by 'user_id', use sorted merge via chunking both sides"
            ],
            3,
            "Senior Explanation: External sorting (Unix sort or GNU sort) handles arbitrarily large files using disk-backed merge-sort (O(n log n) time, O(1) RAM). Once both sorted by join key, iterate through both simultaneously with small chunks (e.g., 100MB), performing merge on sorted chunks - total memory ~200MB + output buffer. Time: ~2-3 hours for 200GB. Option A 'junior trap' - requires ~200GB RAM (impossible on 16GB). Option B loads 100M row right table = ~10-20GB (exceeds 16GB) and inefficient (must compare each left chunk against full right). Option C (Dask) correct conceptually but requires multi-machine cluster OR significant disk spilling (slower). Production: Financial services join massive transaction logs with user data using sorted merge on m5.xlarge. Trade-off: External sort is I/O bound (~1-2 hours), but enables arbitrarily large joins with constant memory.",
            "Hard",
            240
        ),
        create_question(
            "Q16: For merging on multi-column keys ['col1', 'col2'], what's the performance impact vs single-column key?",
            [
                "Negligible - Pandas automatically optimizes multi-column keys",
                "~2× slower due to computing composite hash for each row pair",
                "~10× slower - must compare each column sequentially",
                "Faster - multi-column keys provide better hash distribution"
            ],
            1,
            "Senior Explanation: Multi-column keys require computing composite hash (hash(col1, col2)) per row. Hash computation: ~10-20ns per column → ~20-40ns total vs ~10-20ns for single column = ~2× slower. For 10M × 10M merge: single-key ~15s, multi-key ~30s. Option A 'junior trap' - ignores hash overhead. Option C overstates - modern hash tables are efficient. Option D is wrong - hash distribution depends on data, not key count. Production: Minimize key columns when possible. If joining on ['user_id', 'timestamp'], consider: (1) pre-compute composite key: df['key'] = df['user_id'] + '_' + df['timestamp'].astype(str) (trades memory for speed), or (2) accept 2× overhead if cleaner. Trade-off: Single composite key uses more memory (strings) but faster hashing; multi-column cleaner but slower. Benchmark on 10M rows: single=5s, double=10s, triple=15s.",
            "Medium",
            180
        ),

        # CHUNKING & STREAMING (Questions 17-20)
        create_question(
            "Q17: You update a 50M row DataFrame with 100K new rows hourly. Most efficient approach?",
            [
                "df = pd.concat([df, new_rows], ignore_index=True) - standard concatenation",
                "df = df.append(new_rows, ignore_index=True) - append is optimized for adding rows",
                "Maintain list: rows_list.append(new_rows); rebuild df = pd.concat(rows_list) every 24 hours",
                "Use df.loc[len(df):len(df)+len(new_rows)-1] = new_rows.values"
            ],
            2,
            "Senior Explanation: Repeated concat/append causes quadratic behavior - each operation copies entire DataFrame. For 24 hourly updates: (50M + 50.1M + ... + 52.4M) copies = massive memory churn and hours of CPU. Option C: Accumulate new_rows in list (negligible memory), concat ONCE daily: 24 × 100K row chunks (2.4M total) in one O(n) operation (~5-10s). Option A/B 'junior trap' - works but O(n×m) per update: 24 updates × 50M row copies = ~1.2B row operations vs 52.4M with batch concat (20-50× slower). Option D doesn't resize DataFrame - raises error. Production: Streaming pipelines (user profile updates from Kafka) buffer micro-batches in memory/disk, bulk-insert hourly/daily. Trade-off: List accumulation delays data availability up to 24 hours; balance latency vs compute cost.",
            "Hard",
            200
        ),
        create_question(
            "Q18: You're reading a 10GB parquet file with 50 columns but only need 5. Optimal approach?",
            [
                "pd.read_parquet('file.parquet')[['col1', 'col2', 'col3', 'col4', 'col5']]",
                "pd.read_parquet('file.parquet', columns=['col1', 'col2', 'col3', 'col4', 'col5'])",
                "Load full DataFrame, use del df['unwanted_col'] to remove unneeded columns",
                "Use chunksize parameter to load 5 columns incrementally"
            ],
            1,
            "Senior Explanation: Parquet is columnar - specifying columns parameter reads ONLY those columns from disk, skipping 45/50 columns entirely. I/O reduction: ~10GB → ~1GB (90% less disk read). Memory: ~1GB vs 10GB. Read time: ~2-3s vs ~15-20s (5-10× faster). Option A 'junior trap' - reads all 50 columns (10GB I/O), loads into memory (10GB), THEN selects 5 - wasteful. Option C same as A with extra memory fragmentation. Option D - chunksize doesn't exist for read_parquet; parquet already supports efficient partial reads. Production: ML pipelines selecting features from wide feature stores (e.g., 1000-column user profile tables) - column pruning reduces S3 data transfer costs by 90%+ and fits jobs on smaller EC2 instances. Trade-off: None - always use column pruning with columnar formats (Parquet, ORC). For CSV (row-based), must read all data then select.",
            "Medium",
            150
        ),
        create_question(
            "Q19: You need to deduplicate a 100M row DataFrame by 'user_id' (keep last) with 20GB RAM available. DataFrame size: 25GB. Most memory-efficient?",
            [
                "df.drop_duplicates(subset='user_id', keep='last')",
                "df.sort_values('user_id').drop_duplicates(subset='user_id', keep='last')",
                "Use groupby: df.groupby('user_id').tail(1) to keep last row per group",
                "Chunk-based: process in chunks, maintain dict of last-seen rows, combine"
            ],
            3,
            "Senior Explanation: With 25GB DataFrame and 20GB RAM, loading full DataFrame causes OOM. Chunking: Process chunks (e.g., 5GB each), maintain dict {user_id: last_row_data} (~2-4GB for unique users). For each row in chunk: if user_id in dict, update else insert. After all chunks: convert dict to DataFrame. Peak memory: 5GB (chunk) + 4GB (dict) = 9GB (fits in 20GB). Option A/B 'junior trap' - require loading entire 25GB - OOM. Option C (groupby.tail) also needs full DataFrame. Time: All O(n), but chunking adds dict overhead (~2× slower, acceptable). Production: De-duplicating event streams (clickstream, IoT) on m5.xlarge. Trade-off: Chunking slower (~2-3× vs in-memory) but enables processing datasets larger than RAM, critical for cost-optimized instances.",
            "Hard",
            220
        ),
        create_question(
            "Q20: For a 50-column DataFrame (10M rows), you groupby 'category' (1000 unique) and aggregate 45 numeric columns. What reduces computation most?",
            [
                "df.groupby('category').agg('mean') - Pandas optimizes multi-column aggregation",
                "df.groupby('category', sort=False).agg('mean') to skip sorting groups",
                "Pre-sort by 'category', then groupby('category', sort=False).agg('mean') for cache-friendly access",
                "Parallelize using df.groupby('category').parallel_apply() with joblib"
            ],
            2,
            "Senior Explanation: Pre-sorting by group key (df.sort_values('category')) makes groupby cache-friendly - all rows for a group are contiguous in memory, enabling sequential CPU cache access. Combined with sort=False (skip re-sorting), achieves ~30-50% speedup. For 10M rows × 45 cols: unsorted groupby (~15-20s) vs sorted (~8-12s). Option A baseline good but sorts groups. Option B 'junior trap' - saves sorting at end (~0.5s) but doesn't address core bottleneck. Option D doesn't exist (no parallel_apply for groupby; that's Dask). Production: Feature engineering for ML (e.g., user aggregates from event logs) - sort_values at ETL ingestion enables fast repeated groupby ops. Memory: Sorting requires O(n log n) time upfront but pays off after 3-4 groupby operations. Note: Pandas 1.x+ uses hash-based groupby reducing sorted benefit, but sorted still helps for very large groups.",
            "Hard",
            200
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_pandas()
    db.add_questions("Senior Pandas - Production Optimization", questions)
    print(f"✓ Successfully added {len(questions)} senior Pandas questions!")
    print(f"✓ Category: Senior Pandas - Production Optimization")
