"""
Senior AI Engineer Interview Questions - Batch 9: Tokenization & Text Processing
Topics: BPE, WordPiece, SentencePiece, Vocabulary Management, Multilingual Tokenization
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_tokenization():
    """20 Senior-Level Tokenization Questions"""
    questions = [
        # BYTE-PAIR ENCODING (BPE) (Questions 1-6)
        create_question(
            "Q1: BPE (Byte-Pair Encoding) builds vocabulary by iteratively merging most frequent character pairs. For vocabulary size 50K trained on 10GB text, what is the typical training time?",
            [
                "~1-5 minutes - BPE is very fast",
                "~30-60 minutes - depends on merge iterations",
                "~3-6 hours - requires multiple passes over data",
                "~1-2 days - comparable to model training"
            ],
            1,
            "Senior Explanation: BPE training: (1) Count all character pair frequencies in corpus, (2) Merge most frequent pair, (3) Recount frequencies, (4) Repeat until vocab size reached. For 50K vocab from 10GB: ~30-60 minutes on modern CPU (depends on implementation efficiency). Each merge iteration: O(n) scan over data. Total iterations: ~50K - 256 (initial bytes) ≈ 50K merges. Option A underestimates (simple counting is fast but 50K iterations adds up). Option C/D overestimate - BPE doesn't require model training. Production: Tokenizer training done once, then reused. SentencePiece (optimized BPE) achieves ~10-20 minutes for 50K vocab. Trade-off: Larger vocab (100K) → longer training (~2-3 hours) but better compression.",
            "Medium",
            180
        ),
        create_question(
            "Q2: GPT-2 uses BPE with 50,257 vocab size. Why this specific odd number?",
            [
                "Random - no special significance",
                "256 bytes + ~50K learned merges + 1 special token (e.g., <|endoftext|>)",
                "Prime number for hash table efficiency",
                "Aligned to GPU memory boundaries"
            ],
            1,
            "Senior Explanation: BPE starts with 256 base tokens (all bytes 0-255), then learns ~50K merge operations. GPT-2: 256 bytes + 50,000 merges + 1 special token <|endoftext|> = 50,257. Special tokens: <|endoftext|> marks document boundaries. Option A wrong - carefully chosen. Option C/D irrelevant to tokenization. Production: Vocab size trade-off - larger vocab (100K) → shorter sequences (fewer tokens per text) but larger embedding matrix. GPT-2: 50,257 × 768 (embedding dim) = 38.6M params for embeddings alone. Increasing to 100K → 76.8M params (+38M). Trade-off: Compression vs model size.",
            "Hard",
            200
        ),
        create_question(
            "Q3: For multilingual BPE (e.g., mBERT vocabulary), what issue arises with character-based languages (Chinese, Japanese)?",
            [
                "Characters take multiple bytes in UTF-8 - each character becomes 3-4 tokens, wasting sequence length",
                "BPE doesn't support non-ASCII characters",
                "Requires separate vocabulary per language",
                "No issue - BPE handles all languages equally"
            ],
            0,
            "Senior Explanation: Chinese characters in UTF-8 typically use 3 bytes. Naive byte-level BPE: Each character → 3 tokens (huge waste). Example: '你好世界' (hello world, 4 chars) → 12 tokens vs English 'hello world' → 2-3 tokens. This imbalance hurts multilingual models - Chinese uses 4-6× more sequence length than English for same content. Solution: (1) Character-level BPE for Chinese (mBERT approach), (2) Hybrid tokenization (separate for CJK languages), (3) Larger vocab to learn Chinese character combinations. Option B wrong - BPE handles UTF-8. Production: mBERT uses ~110K vocab with mixed approach. XLM-R (multilingual RoBERTa) uses 250K vocab to better handle diverse languages. Trade-off: Larger vocab for multilingual fairness vs model size.",
            "Hard",
            220
        ),
        create_question(
            "Q4: In BPE, what happens when encountering an unknown word during inference?",
            [
                "Raises error - BPE requires fixed vocabulary",
                "Falls back to character-level tokenization - decomposes into known subwords or bytes",
                "Uses <UNK> token for entire word",
                "Skips the unknown word"
            ],
            1,
            "Senior Explanation: BPE is **open-vocabulary** - any word can be represented by decomposing into subword units, down to bytes if necessary. Example: Unknown word 'Anthropomorphization' → ['Ant', 'hrop', 'oморph', 'ization'] (learned merges) or worst case ['A', 'n', 't', 'h', ...] (byte-level). No <UNK> token needed (unlike word-level tokenization). Option A wrong - BPE's key advantage is handling unseen words. Production: This makes BPE ideal for domain adaptation - medical/legal jargon becomes subwords, not <UNK>. Trade-off: Unknown words → longer token sequences (inefficient) but no information loss.",
            "Medium",
            180
        ),
        create_question(
            "Q5: BPE tokenization of 'hello' vs 'Hello' (capital H) - what typically happens?",
            [
                "Identical tokenization - BPE is case-insensitive",
                "Different tokenization - 'H' might not merge with 'ello' like 'h' does, resulting in different subwords",
                "Always uses lowercase normalization preprocessing",
                "Special handling for capital letters"
            ],
            1,
            "Senior Explanation: BPE is case-sensitive. If training data has 'hello' frequently but 'Hello' rarely, BPE learns merge 'h' + 'ello' → 'hello' but may not learn 'H' + 'ello'. Result: 'hello' → 1 token ['hello'], 'Hello' → 2-3 tokens ['H', 'ello'] or ['He', 'llo']. Impact: Inconsistent tokenization for same word. Solutions: (1) Lowercase normalization (loses case information), (2) Larger vocab to learn both, (3) Case-aware training (explicitly include capitals). Option C - some models do this (BERT uncased) but not inherent to BPE. Production: GPT models case-sensitive (preserve capitals), BERT has cased/uncased versions. Trade-off: Case sensitivity preserves info (proper nouns) but increases vocab size.",
            "Hard",
            200
        ),
        create_question(
            "Q6: For BPE compression efficiency, what is the typical compression ratio (tokens per word) for English?",
            [
                "~0.5 tokens/word - BPE very efficient",
                "~1.3-1.5 tokens/word - typical for 50K vocab",
                "~3-4 tokens/word - heavy fragmentation",
                "Exactly 1.0 - one token per word by design"
            ],
            1,
            "Senior Explanation: English with 50K BPE vocab: Average ~1.3-1.5 tokens per word. Common words ('the', 'is', 'in'): 1 token. Rarer words ('international'): 2-3 tokens (['inter', 'national'] or ['intern', 'ational']). Very rare words: 4-6 tokens. Overall: 100 words → ~130-150 tokens. With larger vocab (100K): ~1.1-1.2 tokens/word (better compression). Option A too optimistic. Option C too pessimistic (character-level would be ~5 tokens/word). Production: For GPT-3 context (2048 tokens), ~1500-1700 words of English text fit. Trade-off: Larger vocab → better compression but larger embedding matrix. Benchmark: LLaMA 32K vocab: ~1.2 tokens/word. GPT-2 50K: ~1.4 tokens/word.",
            "Medium",
            180
        ),

        # WORDPIECE VS SENTENCEPIECE (Questions 7-10)
        create_question(
            "Q7: WordPiece (used by BERT) differs from BPE in the merge criterion. What does WordPiece optimize?",
            [
                "Frequency - merges most frequent pairs like BPE",
                "Likelihood - chooses merge that maximizes language model likelihood on training data",
                "Entropy - minimizes entropy of token distribution",
                "Length - prefers merges creating longer tokens"
            ],
            1,
            "Senior Explanation: BPE: Greedy frequency-based (merge most frequent pair). WordPiece: Likelihood-based (merge that increases LM likelihood most). Computation: For each candidate merge, compute LM probability improvement. More principled than BPE (optimizes for language modeling) but slower training. For 50K vocab: WordPiece ~2-3× slower than BPE (likelihood computation expensive). Quality: Marginal improvement (~0.5-1% better downstream tasks) for significant training cost. Option A describes BPE. Production: BERT uses WordPiece, GPT uses BPE (faster training). Trade-off: Training time vs slightly better vocabulary. Modern trend: BPE preferred for speed, WordPiece legacy from BERT era.",
            "Hard",
            200
        ),
        create_question(
            "Q8: SentencePiece vs BPE/WordPiece - what is the key architectural difference?",
            [
                "SentencePiece is faster - optimized C++ implementation",
                "SentencePiece treats input as raw byte stream (no pre-tokenization), includes whitespace in vocabulary",
                "SentencePiece supports only BPE algorithm",
                "SentencePiece requires pre-trained language model"
            ],
            1,
            "Senior Explanation: Traditional BPE/WordPiece: Assumes pre-tokenized text (split by spaces), then apply subword tokenization. SentencePiece: Treats entire input as raw text, learns to segment including whitespace. Represents space as '_' (U+2581). Benefits: (1) Language-agnostic (works for Chinese/Japanese without word boundaries), (2) Reversible (can perfectly reconstruct original text including spaces), (3) No preprocessing needed. Example: 'hello world' → SentencePiece: ['▁hello', '▁world'], BPE: ['hello', 'world'] (assumes space-split). Option A true but not key difference. Production: T5, mT5, XLM-R use SentencePiece for multilingual support. Trade-off: SentencePiece adds whitespace tokens to vocab (slight overhead) but gains reversibility and universality.",
            "Hard",
            220
        ),
        create_question(
            "Q9: For a multilingual model (100 languages), what vocabulary size is typically needed?",
            [
                "~50K - same as monolingual",
                "~100K-250K - need to cover diverse scripts and morphology",
                "~500K+ - one vocab per language",
                "~10-20K - aggressive compression"
            ],
            1,
            "Senior Explanation: Multilingual vocab must cover: Latin, Cyrillic, Arabic, CJK (Chinese/Japanese/Korean), Devanagari, etc. Each script needs ~5-10K tokens minimum. 100 languages → ~100-250K vocab. mBERT: 110K vocab (limited). XLM-R: 250K vocab (better multilingual coverage). mT5: 250K (SentencePiece). Too small vocab (<50K): Over-segments non-Latin text (Chinese characters become 5-10 tokens). Too large (>500K): Embedding matrix huge (250K × 768 = 192M params). Option A 'junior trap' - assumes monolingual suffices. Production: Larger vocab essential for multilingual fairness. Trade-off: Model size (embeddings) vs per-language efficiency. Benchmark: With 250K vocab, Chinese/English have similar tokens/character (~1.2-1.5× compression).",
            "Medium",
            180
        ),
        create_question(
            "Q10: SentencePiece supports both BPE and Unigram algorithms. What is Unigram language model tokenization?",
            [
                "Same as BPE - different name",
                "Starts with large vocabulary, iteratively removes tokens that minimize LM loss",
                "Uses single characters only",
                "Learns one token per unique word"
            ],
            1,
            "Senior Explanation: Unigram LM: (1) Start with very large vocab (e.g., all substrings), (2) Train unigram language model (each token has probability), (3) Iteratively remove tokens that least degrade LM likelihood, (4) Stop at target vocab size (e.g., 50K). Contrast BPE: Bottom-up (start small, add merges). Unigram: Top-down (start large, prune). Tokenization: For input, find segmentation that maximizes LM likelihood (Viterbi algorithm). Quality: Comparable to BPE, sometimes slightly better for morphologically rich languages. Speed: Training slower (iterative pruning + LM), inference slower (Viterbi vs greedy BPE). Production: T5, ALBERT use Unigram. Trade-off: Training complexity for potentially better tokenization.",
            "Hard",
            200
        ),

        # VOCABULARY MANAGEMENT & OOV (Questions 11-14)
        create_question(
            "Q11: You're deploying a model trained with 50K vocab to a new domain (medical). Vocabulary mismatch causes many unknown subwords. Best approach?",
            [
                "Retrain tokenizer from scratch on medical data - new 50K vocab",
                "Extend vocabulary with domain-specific tokens - add 10K medical terms to existing 50K",
                "Use existing tokenizer as-is - BPE handles unknown words via decomposition",
                "Fine-tune model with character-level tokenization"
            ],
            1,
            "Senior Explanation: Extending vocabulary: (1) Train BPE on medical corpus to learn 10K medical tokens, (2) Merge with base 50K vocab → 60K vocab, (3) Initialize new token embeddings (random or from subword composition), (4) Fine-tune model with extended vocab. Benefits: Preserves base vocab (general knowledge) while adding domain-specific compression. Option A loses general vocabulary. Option C works but inefficient (medical terms become 5-10 tokens). Cost: 10K new embeddings × 768 dim = 7.7M params. Fine-tuning: ~few hours to learn new embeddings. Production: Common for domain adaptation (legal, medical, code). Trade-off: Vocab extension adds parameters but improves domain efficiency. Alternative: Adapter-based approach (keep vocab fixed, adapt representations).",
            "Hard",
            220
        ),
        create_question(
            "Q12: For code tokenization (Python, Java, etc.), what challenge does standard BPE face?",
            [
                "Code is too short - BPE needs long texts",
                "Indentation and whitespace are semantically important - byte-level BPE loses structure",
                "Programming keywords are always in vocabulary",
                "Code has no unknown tokens"
            ],
            1,
            "Senior Explanation: Code structure: Indentation (tabs/spaces) conveys meaning (Python blocks). Standard BPE: Treats whitespace inconsistently (multiple spaces might merge into single token or split). Solution: (1) Preserve whitespace tokens explicitly (don't merge), (2) Use AST-aware tokenization (parse code, tokenize syntax nodes), (3) Character-level for whitespace, BPE for identifiers. Example: '    def foo():' → should preserve 4 spaces, not merge to arbitrary token. Codex/CodeGen: Use BPE with special whitespace handling. Option C wrong - code has rare variable names. Production: GitHub Copilot uses custom tokenizer preserving code structure. Trade-off: Standard BPE simpler but loses code semantics. Code-specific tokenizer better for code generation.",
            "Medium",
            180
        ),
        create_question(
            "Q13: What is the memory overhead of tokenizer vocabulary (50K tokens) in production serving?",
            [
                "~1-2 MB - just vocabulary strings",
                "~10-50 MB - includes merge rules, prefix trees for fast lookup",
                "~500 MB - comparable to small model",
                "Negligible (<100 KB)"
            ],
            1,
            "Senior Explanation: Tokenizer components: (1) Vocabulary strings (50K × ~15 bytes avg) ≈ 750KB, (2) Merge rules for BPE (50K pairs) ≈ 1MB, (3) Trie/prefix tree for fast token lookup ≈ 5-20MB (depends on implementation), (4) Regex patterns, special tokens ≈ 1MB. Total: ~10-50MB. Transformers library (Hugging Face): Tokenizers ~20-30MB loaded. Option A underestimates (just strings). Option C vastly overestimates. Production: Tokenizer memory negligible vs model (7B model = 14GB). But for edge deployment (mobile), 30MB tokenizer + 100MB quantized model = significant. Trade-off: Smaller vocab (10K) → ~5MB tokenizer but worse compression. Optimized tokenizers (C++ SentencePiece) ~10-15MB.",
            "Medium",
            180
        ),
        create_question(
            "Q14: For streaming tokenization (processing input as it arrives), what is the main challenge?",
            [
                "Tokenization is too slow for real-time",
                "BPE requires complete input to choose optimal segmentation - streaming must tokenize greedily (may be suboptimal)",
                "Streaming not supported by tokenizers",
                "Memory exhaustion from buffering"
            ],
            1,
            "Senior Explanation: BPE tokenization: Greedy left-to-right (apply longest matching merge). For complete input, this is deterministic and correct. For streaming: As characters arrive, tokenize immediately with current knowledge. Issue: Later characters might suggest different tokenization. Example: Stream 'inter...' → tokenize as ['in', 'ter'], later '...national' arrives → optimal would be ['inter', 'national']. Generally not a problem (greedy is usually optimal), but edge cases exist. Unigram LM: Worse for streaming (needs whole sequence for Viterbi). Production: Most streaming applications (chatbots, live transcription) use BPE greedily - works fine. Trade-off: Streaming latency vs optimal tokenization. Buffering: Can buffer ~10-20 characters to improve without much latency.",
            "Hard",
            200
        ),

        # MULTILINGUAL & DETOKENIZATION (Questions 15-20)
        create_question(
            "Q15: For detokenization (converting tokens back to text), BPE vs SentencePiece - which is lossless?",
            [
                "BPE - designed for reversibility",
                "SentencePiece - includes whitespace in vocabulary, enabling perfect reconstruction",
                "Both equally lossless",
                "Neither - tokenization always loses information"
            ],
            1,
            "Senior Explanation: BPE: Assumes pre-tokenized input (space-split). Detokenization: Concatenate tokens, add spaces between words. Problem: Multiple spaces, tabs, newlines collapsed to single space. Example: 'hello  world' (2 spaces) → tokens → 'hello world' (1 space). Information loss. SentencePiece: Treats space as '▁' token. Detokenization: Replace '▁' with space, perfectly recovers original including whitespace. Example: 'hello  world' → ['▁hello', '▁', '▁world'] → exact reconstruction. Option C wrong - BPE loses whitespace details. Production: For tasks needing exact text (code generation, data augmentation), use SentencePiece. For NLP tasks where exact spacing doesn't matter (classification), BPE sufficient. Trade-off: SentencePiece complexity for reversibility vs BPE simplicity.",
            "Hard",
            200
        ),
        create_question(
            "Q16: Special tokens (<bos>, <eos>, <pad>) are added to vocabulary. For a 50K vocab, how many special tokens are typical?",
            [
                "1-3 - minimal set (bos, eos, pad)",
                "5-10 - includes mask, unknown, separator, etc.",
                "50-100 - extensive special token set",
                "0 - not needed with modern architectures"
            ],
            1,
            "Senior Explanation: Common special tokens: <bos> (begin sequence), <eos> (end sequence), <pad> (padding), <unk> (unknown, rarely used with BPE), <mask> (BERT-style), <sep> (separator), <cls> (classification). Total: ~5-10 typical. GPT-2: <|endoftext|> (1 token). BERT: [CLS], [SEP], [PAD], [UNK], [MASK] (5 tokens). T5: 100 'extra_id' tokens for span masking (100 special tokens). Option C describes T5's extensive set (outlier). Production: Keep special tokens minimal to maximize vocab for real text. Trade-off: More special tokens → less vocab for subwords. T5's 100 extra tokens from 32K vocab means ~0.3% waste.",
            "Medium",
            180
        ),
        create_question(
            "Q17: For emoji tokenization, standard BPE (50K vocab) performs how?",
            [
                "Well - emojis in vocabulary",
                "Poorly - each emoji becomes 3-4 byte tokens (UTF-8), wasting sequence length",
                "Emojis automatically filtered during preprocessing",
                "Requires special emoji tokenizer"
            ],
            1,
            "Senior Explanation: Emojis in UTF-8: 3-4 bytes each (e.g., 😀 = 0xF0 0x9F 0x98 0x80). Byte-level BPE: Each emoji → 4 tokens (unless emoji appears frequently enough in training to learn merged representation). If BPE trains on emoji-heavy data (Twitter), common emojis might merge to 1-2 tokens. Otherwise, 4 tokens/emoji. Example: 'I love coding 😀❤️💻' → ~10-15 tokens (emojis dominate). Impact: Social media text inefficient. Solution: (1) Train on emoji-rich corpus, (2) Add common emojis as special tokens. Production: Models trained on web data (GPT-3) handle emojis reasonably (common ones are merged). Domain-specific models (formal text) waste tokens on emojis. Trade-off: Emoji coverage vs general text efficiency.",
            "Medium",
            180
        ),
        create_question(
            "Q18: For tokenization speed, what is the throughput for BPE tokenizer (50K vocab) on modern CPU?",
            [
                "~10-50 MB/s - quite slow",
                "~100-500 MB/s - moderate speed",
                "~1-5 GB/s - very fast with optimized implementations",
                "~10-50 KB/s - bottleneck for inference"
            ],
            2,
            "Senior Explanation: Optimized BPE (e.g., HuggingFace Tokenizers Rust-based, SentencePiece C++): ~1-5 GB/s on modern CPU (single-threaded). For 1 million tokens: ~10-50 MB of text → ~10-50 ms tokenization. Pure Python BPE: ~50-200 MB/s (20-100× slower). Option A/D underestimate modern implementations. Production: Tokenization rarely bottleneck - model inference (10-100ms for generation) dominates. For batch preprocessing (offline), parallelize across CPUs: 32 cores × 2 GB/s = 64 GB/s (process 1TB in ~15 seconds). Trade-off: Rust/C++ tokenizer fast but harder to customize vs Python slow but flexible. Benchmark: Tokenize 1GB text in ~200-1000ms (optimized) vs ~10-20s (Python).",
            "Hard",
            200
        ),
        create_question(
            "Q19: Subword regularization (used in training) randomly samples different tokenizations. What is the purpose?",
            [
                "Speed up tokenization by using randomness",
                "Data augmentation - same text gets different token sequences, improving robustness",
                "Reduce vocabulary size",
                "Handle out-of-vocabulary words"
            ],
            1,
            "Senior Explanation: Subword regularization: During training, instead of deterministic tokenization, sample from multiple valid segmentations. Example: 'international' → could be ['inter', 'national'], ['intern', 'ational'], ['in', 'ter', 'national']. Each valid under BPE/Unigram. Benefits: (1) Data augmentation (same sentence seen with different tokenizations → regularization), (2) Model learns to be robust to tokenization variations. Used in: XLM-R, mBART (improves multilingual performance ~1-2%). Implementation: Unigram LM naturally supports (sample from distribution), BPE requires modification (dropout on merges). Production: Only used during TRAINING (inference uses deterministic tokenization for consistency). Trade-off: Training slower (~10-20% overhead) but better robustness.",
            "Hard",
            220
        ),
        create_question(
            "Q20: For cross-lingual transfer (train on English, test on French), vocabulary choice matters. Best approach?",
            [
                "English-only vocabulary - simpler",
                "French-only vocabulary - target language focused",
                "Multilingual vocabulary trained on English + French - enables shared representations",
                "Separate vocabularies with translation"
            ],
            2,
            "Senior Explanation: Multilingual vocab: Train BPE/SentencePiece on combined English + French corpus. Shared subwords: Many Romance language words share Latin roots ('international' in English, 'international' in French → same tokens). Enables zero-shot transfer (model learns from English, applies to French via shared vocabulary). With separate vocabs: 'international' → different tokens in each language, no transfer. Benchmark: Multilingual vocab improves zero-shot transfer by 10-30% absolute accuracy. Example: XLM-R (100 languages, 250K vocab) achieves strong cross-lingual transfer. Option A/B 'junior trap' - limits transfer. Production: For multilingual deployment, always use multilingual vocab. Trade-off: Vocab size (250K for 100 langs) vs monolingual efficiency (50K per language but no transfer).",
            "Hard",
            220
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_tokenization()
    db.add_bulk_questions("Senior Tokenization - BPE, WordPiece, SentencePiece", questions)
    print(f"✓ Successfully added {len(questions)} senior Tokenization questions!")
    print(f"✓ Category: Senior Tokenization - BPE, WordPiece, SentencePiece")
