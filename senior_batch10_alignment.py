"""
Senior AI Engineer Interview Questions - Batch 10: LLM Alignment Techniques
Topics: RLHF, DPO, PPO, Reward Modeling, Safety Alignment
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_alignment():
    """20 Senior-Level Alignment Questions"""
    questions = [
        # RLHF (REINFORCEMENT LEARNING FROM HUMAN FEEDBACK) (Questions 1-7)
        create_question(
            "Q1: RLHF has 3 stages: (1) SFT (supervised fine-tuning), (2) reward model training, (3) RL optimization. For a 7B model, what is the approximate compute cost ratio between stages?",
            [
                "1:1:1 - all stages equal compute",
                "1:0.5:3 - RL dominates compute (70-80% of total cost)",
                "3:1:0.5 - SFT most expensive",
                "1:2:1 - reward model training dominates"
            ],
            1,
            "Senior Explanation: Compute breakdown (relative): SFT ~15-20% (single-pass supervised learning on demonstrations, few epochs). Reward model ~5-10% (small model 1-6B, binary classification training). RL (PPO) ~70-80% (requires 4 models in memory - policy, reference, reward, value; many rollout iterations). For InstructGPT (GPT-3 175B): SFT ~1K GPU-hours, reward model ~500 GPU-hours, PPO ~5-10K GPU-hours. Option A 'junior trap'. Production: Most RLHF cost is RL stage. Optimization: (1) Smaller reward/value models, (2) Fewer RL iterations, (3) LoRA for policy (reduces memory). Trade-off: RL stage crucial for alignment quality, can't skip.",
            "Hard",
            240
        ),
        create_question(
            "Q2: In RLHF reward modeling, you train a model to predict human preference between two outputs. What architecture is typically used?",
            [
                "Separate encoder for each output, compare embeddings",
                "Single sequence: [prompt, output_A, SEP, output_B], predict A>B or B>A",
                "Run model twice (once per output), compare final scores",
                "Siamese network with shared weights"
            ],
            1,
            "Senior Explanation: Reward model: Concatenate prompt, output_A, separator, output_B into single sequence. Feed to model (typically smaller than policy, e.g., 6B for 13B policy), output scalar score. Training: Pairs (A, B) with human label 'A better' or 'B better'. Loss: Cross-entropy on preference. Why single sequence: Enables comparison in context (attending between outputs), more parameter-efficient than separate encoders. Option C (run twice) loses cross-attention. Production: InstructGPT uses 6B reward model for 175B policy. Memory: 6B model ~12GB (fp16), allows batch size ~8-16 for comparisons. Trade-off: Smaller reward model faster and cheaper but may miss subtle quality differences.",
            "Hard",
            220
        ),
        create_question(
            "Q3: During PPO training for RLHF, how many models are simultaneously in memory?",
            [
                "1 - just the policy model being trained",
                "2 - policy and reward model",
                "4 - policy, reference policy, reward model, value model",
                "8 - multiple checkpoints for stability"
            ],
            2,
            "Senior Explanation: PPO requires: (1) **Policy model** (being trained, updated gradients), (2) **Reference policy** (frozen copy of initial policy, for KL penalty), (3) **Reward model** (scores outputs), (4) **Value model** (estimates future rewards for advantage computation). Memory for 7B policy: Policy (14GB trainable) + reference (14GB frozen) + reward 3B (6GB) + value 3B (6GB) = ~40GB just models. With gradients, optimizer states (Adam), activations: ~80-120GB total. Requires 2-4× A100 (80GB). Option A/B vastly underestimate. Production: Major RLHF bottleneck is multi-model memory. Optimizations: (1) Share reference/value models (marginal), (2) 8-bit reference model (halve memory), (3) Offload reference to CPU. Trade-off: Memory limits batch size (small batches →noisy gradients).",
            "Hard",
            240
        ),
        create_question(
            "Q4: In PPO for RLHF, what is the KL divergence penalty term for?",
            [
                "Regularization to prevent overfitting",
                "Prevents policy from diverging too far from reference (initial) policy - maintains language fluency and prevents reward hacking",
                "Ensures policy matches human distribution",
                "Speeds up convergence"
            ],
            1,
            "Senior Explanation: KL penalty: KL(π_θ || π_ref) where π_θ is current policy, π_ref is reference (pre-RL policy). Without KL penalty: Policy optimizes reward aggressively, potentially: (1) Reward hacking (exploiting reward model flaws), (2) Mode collapse (generates same high-reward response), (3) Losing language fluency (incoherent text that scores high). With KL penalty (β typically 0.01-0.1): Policy stays 'close' to reference, preserving original model's capabilities while improving alignment. Example: If reference says 'The answer is X' (fluent), policy won't change to '!@#X' (nonsense) even if reward model mistakenly scores it high. Production: KL penalty critical for stable RLHF. Too high β → policy doesn't improve. Too low → reward hacking. Trade-off: Alignment improvement vs preserving base model capabilities.",
            "Hard",
            220
        ),
        create_question(
            "Q5: For RLHF reward model, how much human preference data is typically needed?",
            [
                "~1K-10K comparisons - minimal data",
                "~50K-100K comparisons - moderate dataset",
                "~1M-10M comparisons - large dataset",
                "~100M+ comparisons - comparable to pre-training"
            ],
            1,
            "Senior Explanation: Typical RLHF: ~50K-100K human preference comparisons. InstructGPT: ~50K comparisons (100K outputs compared pairwise). Anthropic (Claude): ~100K+ comparisons. Each comparison: Human ranks 2-4 model outputs for same prompt. Data collection cost: ~$0.50-$2 per comparison (labeler time), so 100K comparisons = $50K-$200K. Option A too small (underfits). Option C/D overkill (expensive, diminishing returns). Production: Reward model generalizes well from 50K comparisons due to transfer learning from pre-trained LLM. Trade-off: More data improves reward model but quadratic cost increase. Active learning: Focus on hard comparisons (close outputs) to maximize data efficiency.",
            "Medium",
            180
        ),
        create_question(
            "Q6: RLHF reward model training uses pairwise comparisons. What loss function?",
            [
                "MSE between predicted and actual scores",
                "Cross-entropy on binary preference (A > B or B > A)",
                "Ranking loss (e.g., Bradley-Terry model) - max log probability of observed ranking",
                "Contrastive loss"
            ],
            2,
            "Senior Explanation: Bradley-Terry model: P(A > B) = σ(r(A) - r(B)) where r(x) is reward model's score. Loss: -log P(observed ranking). For label 'A > B': loss = -log σ(r(A) - r(B)) = log(1 + exp(r(B) - r(A))). Encourages r(A) > r(B). Option B (binary cross-entropy) equivalent formulation. Ranking loss more general (handles ties, multiple outputs). Production: OpenAI uses Bradley-Terry. Anthropic uses similar. Extensions: Plackett-Luce for ranking >2 outputs. Trade-off: Pairwise comparisons simpler for humans than absolute scores, but requires more data (N outputs → O(N²) pairs).",
            "Hard",
            220
        ),
        create_question(
            "Q7: After RLHF, the model sometimes exhibits 'reward hacking'. What is this?",
            [
                "Model learns to predict rewards instead of generating good text",
                "Model exploits reward model flaws - generates outputs that score high reward but are actually low quality (e.g., overly verbose, sycophantic)",
                "Reward model overfits to training data",
                "Policy model diverges from reference too much"
            ],
            1,
            "Senior Explanation: Reward hacking: Policy finds shortcut to maximize reward without improving actual quality. Examples: (1) **Sycophancy**: Always agreeing with user, even for false statements (reward model might prefer agreement). (2) **Verbosity**: Longer responses score higher (quantity vs quality). (3) **Keyword stuffing**: Including phrases reward model associates with quality. Prevention: (1) Better reward model (more data, adversarial training), (2) KL penalty (limits divergence), (3) Multi-objective rewards (length penalty, diversity). Option A confuses with mode collapse. Production: All RLHF models exhibit some reward hacking. Iterative process: Deploy, identify hacks, retrain reward model, repeat. Trade-off: Perfect reward modeling impossible (human preferences complex), some hacking inevitable.",
            "Hard",
            220
        ),

        # DPO (DIRECT PREFERENCE OPTIMIZATION) (Questions 8-14)
        create_question(
            "Q8: DPO (Direct Preference Optimization) vs RLHF - what is the key difference?",
            [
                "DPO is faster - simpler algorithm",
                "DPO eliminates reward model and RL - optimizes policy directly from preference data via reparameterized objective",
                "DPO requires less data",
                "DPO is only for small models"
            ],
            1,
            "Senior Explanation: RLHF: Train reward model (stage 2), then RL with PPO (stage 3). DPO: Derives closed-form solution to RLHF objective, bypassing reward model and RL. DPO loss: -log σ(β log(π_θ(y_w)/π_ref(y_w)) - β log(π_θ(y_l)/π_ref(y_l))) where y_w = preferred output, y_l = rejected output, β = KL penalty weight. Directly optimizes policy from preference pairs. Benefits: (1) No reward model (saves ~6GB memory), (2) No RL (simpler, faster, more stable), (3) Single stage (vs 2 stages in RLHF). Performance: Comparable or better than PPO in many tasks. Production: Zephyr-7B uses DPO (outperforms RLHF models). Trade-off: DPO assumes implicit reward model structure, may be less flexible than explicit reward model for complex objectives.",
            "Hard",
            240
        ),
        create_question(
            "Q9: For DPO training, what models are needed in memory?",
            [
                "1 - just policy model",
                "2 - policy and reference policy",
                "3 - policy, reference, and critic",
                "4 - same as PPO"
            ],
            1,
            "Senior Explanation: DPO requires: (1) **Policy model** (being trained), (2) **Reference policy** (frozen, for KL term in DPO loss). Total: ~28GB for 7B model (14GB × 2). Compare PPO: ~40GB (4 models). Memory savings: ~30%. No reward model needed (DPO loss directly uses preference pairs). No value model needed (no advantage estimation). Batch size: Larger batches feasible vs PPO (e.g., batch=16 DPO vs batch=4-8 PPO on same VRAM). Production: DPO enables RLHF-style alignment on smaller hardware. 7B DPO possible on single A100 (80GB) with reasonable batch size. Trade-off: Simpler (2 models) but less modular (can't separate reward modeling from policy optimization).",
            "Hard",
            200
        ),
        create_question(
            "Q10: DPO training time compared to full RLHF (SFT + reward model + PPO) for 7B model?",
            [
                "~10× faster - DPO very efficient",
                "~2-3× faster - eliminates reward training and RL stages, but still needs comparable iterations",
                "Same speed - different algorithms, same compute",
                "Slower - DPO is more complex"
            ],
            1,
            "Senior Explanation: RLHF: SFT (10 hours) + reward model (5 hours) + PPO (50-100 hours) = ~65-115 hours (single A100). DPO: SFT (10 hours) + DPO (20-30 hours) = ~30-40 hours. Speedup: ~2-3×. DPO faster because: (1) No reward model training, (2) Supervised-style training (stable, fewer iterations than RL). Both need similar data and SFT. Production: DPO preferred for rapid iteration (experiments in days vs weeks). Quality: DPO comparable to PPO for instruction-following, sometimes better for summarization. Trade-off: PPO more flexible for complex reward shaping (e.g., multi-objective: helpfulness + safety + factuality). DPO simpler but tied to pairwise preferences.",
            "Medium",
            180
        ),
        create_question(
            "Q11: In DPO, the β hyperparameter controls what?",
            [
                "Learning rate",
                "Strength of KL penalty - how much policy can deviate from reference",
                "Batch size",
                "Number of training iterations"
            ],
            1,
            "Senior Explanation: β in DPO loss: Same role as KL penalty in RLHF. Higher β → policy stays closer to reference (more conservative), lower β → policy can deviate more (aggressive optimization). Typical β: 0.1-0.5 for DPO (vs 0.01-0.1 for PPO, different scales). Tuning: Low β → risk of reward hacking, high β → minimal improvement from reference. Optimal β: Depends on quality of preference data and reference model. Production: β=0.1 good default for 7B models. Larger models (13B+) can use lower β (0.05) safely. Trade-off: β controls exploration-exploitation. Grid search over [0.05, 0.1, 0.2, 0.5] common.",
            "Medium",
            180
        ),
        create_question(
            "Q12: Can DPO be combined with LoRA for efficient training?",
            [
                "No - DPO requires full model fine-tuning",
                "Yes - apply LoRA to policy model, keep reference model frozen in full precision, significantly reduces memory",
                "Yes but performance degrades significantly",
                "Only for models <7B"
            ],
            1,
            "Senior Explanation: DPO + LoRA: (1) Policy model with LoRA adapters (trainable params ~8M for r=16), (2) Reference model frozen full precision (14GB for 7B). Memory: Policy base (14GB) + LoRA params (16MB) + optimizer (32MB) + reference (14GB) = ~28GB total (vs ~40GB full DPO). Further optimization: Reference model in 8-bit (7GB), total ~21GB. Enables DPO on smaller GPUs. Quality: LoRA DPO achieves ~95-98% of full DPO performance. Production: QLoRA + DPO enables alignment on consumer GPUs (RTX 4090 24GB). Example: Zephyr-7B-beta uses LoRA+DPO. Trade-off: Slight quality loss (~1-2%) for massive memory savings (2×).",
            "Hard",
            220
        ),
        create_question(
            "Q13: DPO requires preference pairs (chosen vs rejected). How much data typically needed?",
            [
                "~1K-5K pairs - very sample efficient",
                "~10K-60K pairs - moderate dataset comparable to RLHF",
                "~500K-1M pairs - large scale",
                "~10M+ pairs - needs huge dataset"
            ],
            1,
            "Senior Explanation: DPO: ~10K-60K preference pairs typical. Zephyr-7B: ~60K pairs from UltraFeedback dataset. StableLM: ~20K pairs. Comparable to RLHF reward model data (50K-100K). DPO doesn't require more data than RLHF - same preference data, different usage (direct policy optimization vs reward modeling). Data quality > quantity: High-quality 20K pairs outperform noisy 100K pairs. Collection: Same as RLHF (humans rank outputs, ~$0.50-$2 per comparison). Production: 60K pairs = $30K-$120K labeling cost. Alternative: Use AI feedback (Constitutional AI approach) to generate synthetic preferences (cheaper but potentially biased). Trade-off: Human data expensive but higher quality, synthetic data cheap but may reinforce model biases.",
            "Medium",
            180
        ),
        create_question(
            "Q14: DPO vs PPO - which is more stable during training?",
            [
                "PPO - more mature and tested",
                "DPO - supervised-style training is inherently more stable than RL",
                "Both equally stable",
                "Neither - alignment training always unstable"
            ],
            1,
            "Senior Explanation: DPO training: Supervised-style (gradient descent on preference loss), deterministic, stable. PPO training: RL algorithm (policy gradients, value estimation), stochastic, sensitive to hyperparameters. DPO advantages: (1) No clipping hyperparameters (PPO's ε=0.2), (2) No advantage estimation (source of variance), (3) Direct gradient signal from preferences. PPO issues: (1) Exploration-exploitation tradeoff, (2) Reward model errors compound, (3) Value function approximation errors. Production: DPO preferred for stability - fewer failed runs, less hyperparameter tuning. PPO requires expert tuning (learning rates, clip ranges, KL penalties). Trade-off: DPO simpler but less flexible for complex reward functions (multi-objective, sparse rewards).",
            "Medium",
            180
        ),

        # PPO FOR LLMS (Questions 15-17)
        create_question(
            "Q15: In PPO for LLMs, what is the typical PPO clip range (ε)?",
            [
                "ε = 0.01-0.05 - very conservative",
                "ε = 0.1-0.3 - standard range for LLMs",
                "ε = 0.5-1.0 - aggressive clipping",
                "No clipping used for LLMs"
            ],
            1,
            "Senior Explanation: PPO clip range ε: Limits policy ratio r = π_new(a|s) / π_old(a|s) to [1-ε, 1+ε]. For LLMs: ε = 0.1-0.3 typical. InstructGPT likely uses ε ≈ 0.2. Smaller ε: More conservative updates (stable but slow). Larger ε: Aggressive updates (faster but risky - potential collapse). LLMs use slightly larger ε than standard RL (Atari: ε=0.1-0.2) because: (1) Continuous text space (vs discrete actions), (2) KL penalty provides additional stability. Production: ε=0.2 good default. Monitor KL divergence during training - if KL spikes, reduce ε. Trade-off: Exploration speed vs stability. Grid search [0.1, 0.2, 0.3] with early stopping on KL violations.",
            "Hard",
            200
        ),
        create_question(
            "Q16: PPO for LLMs uses GAE (Generalized Advantage Estimation). What is the advantage function?",
            [
                "Difference between current and reference policy",
                "Difference between actual return and value function estimate: A(s,a) = Q(s,a) - V(s) - measures how much better action a is than average",
                "Reward model score",
                "KL divergence term"
            ],
            1,
            "Senior Explanation: Advantage A(s,a) = Q(s,a) - V(s): How much better is action a than baseline (value function). Positive advantage → action better than expected → increase probability. Negative → worse → decrease probability. GAE: A^GAE = Σ(γλ)^t δ_t where δ_t = r_t + γV(s_{t+1}) - V(s_t) (TD error). Parameters: γ (discount, typically 1.0 for text), λ (GAE lambda, 0.95). Reduces variance in advantage estimates vs raw returns. Production: Value model (separate network or shared with policy) estimates V(s). Quality: Good value function critical for PPO convergence. Poor value estimates → high variance → slow/unstable training. Trade-off: Shared value-policy network saves memory but couples learning. Separate value network more stable.",
            "Hard",
            220
        ),
        create_question(
            "Q17: For PPO on LLMs, how many rollout steps per update?",
            [
                "1-10 steps - short rollouts",
                "50-200 steps - moderate rollouts",
                "Entire episode (full response generation, ~100-1000 tokens)",
                "Variable length"
            ],
            2,
            "Senior Explanation: LLM generation is episodic - one response = one episode. Rollout = generate complete response (e.g., 256 tokens for QA, 1024 for summarization). Collect reward at end (from reward model scoring full response). Unlike Atari RL (step-by-step rewards), LLM RLHF: Sparse reward (only at episode end). Implications: (1) Credit assignment harder (which tokens caused good reward?), (2) Value function estimates entire response value, (3) High variance (single reward for hundreds of actions). Production: PPO collects batch of rollouts (e.g., 64 responses), computes advantages, updates policy. Batch size limited by memory (4 models + rollout buffers). Trade-off: Larger rollout batch → lower variance but more memory.",
            "Hard",
            200
        ),

        # REWARD MODELING & SAFETY (Questions 18-20)
        create_question(
            "Q18: For safety alignment (preventing harmful outputs), what approach is most effective?",
            [
                "Filter training data - remove harmful content",
                "Multi-objective RLHF - separate reward models for helpfulness and harmlessness, optimize both",
                "Post-hoc filtering - detect harmful outputs and block",
                "Prompt engineering - instruct model to be safe"
            ],
            1,
            "Senior Explanation: Multi-objective RLHF (Anthropic's Constitutional AI): Train separate reward models: (1) Helpfulness RM (useful responses), (2) Harmlessness RM (safe, non-toxic). Combined objective: R = α × R_helpful + β × R_harmless. Balance α, β to prioritize safety. During RL, policy optimizes both - can't maximize helpfulness by sacrificing safety. Option A insufficient (model learns from context during pre-training). Option C reactive, not proactive. Option D weak (easily circumvented by jailbreaks). Production: Claude uses Constitutional AI (multi-objective). GPT-4 likely similar. Trade-off: Helpfulness vs safety (reducing β improves helpfulness but increases risk). Typical β > α (prioritize safety).",
            "Hard",
            220
        ),
        create_question(
            "Q19: Constitutional AI (Anthropic) uses AI feedback instead of human feedback for safety. How?",
            [
                "AI labels safety violations automatically - no humans needed",
                "AI generates critiques and revisions based on constitutional principles - uses this as synthetic preference data",
                "AI filters unsafe content during pre-training",
                "AI acts as reward model"
            ],
            1,
            "Senior Explanation: Constitutional AI: (1) Define 'constitution' (principles like 'choose less harmful response'), (2) Model generates responses, (3) AI critic (separate LLM) evaluates responses against constitution, suggests revisions, (4) Model generates revised responses, (5) Create preference pairs: (original, revised) with revised as preferred. Train on synthetic preferences. Benefits: (1) Scalable (AI feedback cheaper than human), (2) Consistent (principles explicitly defined), (3) Reduces human exposure to harmful content. Concerns: AI feedback may miss nuanced harm. Production: Anthropic uses hybrid - AI feedback for safety, human feedback for helpfulness. Trade-off: AI feedback scalable but potentially biased (inherits model's biases). Human feedback expensive but higher quality.",
            "Hard",
            240
        ),
        create_question(
            "Q20: Reward model overoptimization (Goodhart's Law): 'When a measure becomes a target, it ceases to be a good measure.' How does this manifest in RLHF?",
            [
                "Reward model accuracy decreases over time",
                "Policy exploits reward model - generates outputs that score artificially high but are low quality per true human preferences (proxy becomes target)",
                "Training becomes unstable",
                "Model forgets pre-training knowledge"
            ],
            1,
            "Senior Explanation: Goodhart's Law in RLHF: Reward model is PROXY for true human preferences (trained on limited data). During RL, policy optimizes proxy aggressively. Eventually, policy finds inputs where proxy diverges from true preferences (reward hacking). Example: Reward model trained on 50K comparisons may prefer concise answers. Policy learns to generate very short answers (scores high on proxy) but are unhelpful (low true preference). Detection: Monitor out-of-distribution (OOD) detection on policy outputs - if outputs drift from reward model's training distribution, proxy likely unreliable. Mitigation: (1) Iterative RLHF (retrain reward model on policy outputs), (2) Ensemble reward models, (3) Early stopping before overoptimization. Production: All RLHF systems exhibit some overoptimization. KL penalty helps but doesn't eliminate. Trade-off: Optimization time vs proxy quality degradation.",
            "Hard",
            240
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_alignment()
    db.add_bulk_questions("Senior Alignment - RLHF, DPO, PPO, Reward Modeling", questions)
    print(f"✓ Successfully added {len(questions)} senior Alignment questions!")
    print(f"✓ Category: Senior Alignment - RLHF, DPO, PPO, Reward Modeling")
