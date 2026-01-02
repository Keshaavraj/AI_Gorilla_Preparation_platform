"""
Batch 1 Part 2: Core AI/ML Questions Continuation
- Artificial Intelligence (20 questions)
- NLP (20 questions)
- Generative AI (20 questions)
"""

from database_manager import QuestionDatabase, create_question

def populate_artificial_intelligence():
    """20 Artificial Intelligence Questions"""
    questions = [
        create_question(
            "In A* search algorithm, what does the evaluation function f(n) = g(n) + h(n) represent?",
            [
                "g(n) is estimated cost to goal, h(n) is cost from start",
                "g(n) is cost from start to n, h(n) is estimated cost from n to goal",
                "Both g(n) and h(n) are heuristic estimates",
                "f(n) represents only the depth of node n"
            ],
            1,
            "In A*, g(n) is the actual cost from the start node to node n, and h(n) is the heuristic estimate of cost from n to the goal. f(n) = g(n) + h(n) is the estimated total cost of the path through n. A* is optimal when h(n) is admissible (never overestimates) and complete when the branching factor is finite.",
            "Medium",
            90
        ),
        create_question(
            "Which type of AI agent can handle partially observable environments by maintaining internal state?",
            [
                "Simple reflex agent",
                "Model-based reflex agent",
                "Goal-based agent",
                "Both model-based and goal-based agents"
            ],
            3,
            "Model-based agents maintain an internal state/model of the world to handle partial observability. Goal-based agents also need internal state to plan toward goals. Simple reflex agents only react to current percepts without memory, failing in partially observable environments. Utility-based agents extend goal-based agents with utility functions.",
            "Hard",
            95
        ),
        create_question(
            "In game theory, the Minimax algorithm is used to:",
            [
                "Maximize the minimum gain (minimize opponent's maximum gain)",
                "Always pick the maximum value",
                "Random selection of moves",
                "Only works for cooperative games"
            ],
            0,
            "Minimax assumes the opponent plays optimally to minimize your score. You maximize your minimum guaranteed payoff. In two-player zero-sum games, you pick the move that maximizes your score assuming the opponent will respond by minimizing it. Alpha-beta pruning optimizes minimax by eliminating branches that won't affect the final decision.",
            "Medium",
            90
        ),
        create_question(
            "What is the main difference between breadth-first search (BFS) and depth-first search (DFS)?",
            [
                "BFS uses a queue, DFS uses a stack; BFS finds shortest path in unweighted graphs",
                "BFS uses a stack, DFS uses a queue",
                "BFS is always faster than DFS",
                "DFS always finds the optimal solution"
            ],
            0,
            "BFS explores level by level using a queue (FIFO), guaranteeing the shortest path in unweighted graphs but requiring more memory. DFS explores as deep as possible using a stack (LIFO), using less memory but not guaranteeing optimal solutions. BFS has O(b^d) space complexity vs DFS's O(bd) where b is branching factor and d is depth.",
            "Medium",
            85
        ),
        create_question(
            "In Reinforcement Learning, what does the 'exploration vs. exploitation' dilemma refer to?",
            [
                "Exploring the state space vs. exploiting parallel computing",
                "Balancing trying new actions (exploration) vs. using known good actions (exploitation)",
                "Exploring new algorithms vs. using existing ones",
                "Only relevant in supervised learning"
            ],
            1,
            "The exploration-exploitation tradeoff is fundamental in RL. Exploitation means using current knowledge to maximize immediate reward. Exploration means trying new actions to discover potentially better strategies. Too much exploitation may get stuck in local optima; too much exploration wastes time on suboptimal actions. Strategies include ε-greedy, softmax, and UCB.",
            "Medium",
            90
        ),
        create_question(
            "In a Markov Decision Process (MDP), the Markov property states that:",
            [
                "Future states depend on the entire history of past states",
                "The next state depends only on the current state and action, not on history",
                "All states are equally likely",
                "The environment is fully deterministic"
            ],
            1,
            "The Markov property states that the future is independent of the past given the present: P(s'|s,a) depends only on current state s and action a, not on the sequence of states that led to s. This memoryless property enables efficient algorithms like value iteration and policy iteration. MDPs are the framework for formulating RL problems.",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of a heuristic function in informed search algorithms?",
            [
                "To guarantee finding the optimal solution",
                "To provide an estimate of the cost from a state to the goal",
                "To randomly select next states",
                "To make the search exhaustive"
            ],
            1,
            "A heuristic function h(n) estimates the cost from state n to the goal, guiding search toward promising directions without exploring every possibility. Good heuristics dramatically reduce search time. For A* to be optimal, h(n) must be admissible (never overestimate). Common heuristics include Manhattan distance, Euclidean distance, and problem-specific domain knowledge.",
            "Medium",
            85
        ),
        create_question(
            "In constraint satisfaction problems (CSP), what is backtracking?",
            [
                "A method to return to previous states when constraints are violated",
                "Always finding the optimal solution",
                "A type of neural network",
                "Only used in graph problems"
            ],
            0,
            "Backtracking is a depth-first search that incrementally assigns values to variables, checking constraints after each assignment. When constraints are violated, it backtracks (undoes assignments) to try different values. Optimizations include forward checking (eliminating inconsistent values), constraint propagation (arc consistency), and variable/value ordering heuristics (MRV, LCV).",
            "Medium",
            90
        ),
        create_question(
            "In the context of planning, what is a 'partially ordered plan'?",
            [
                "A plan where some actions can be executed in any order",
                "A plan that is incomplete",
                "A plan with random action ordering",
                "A plan that always fails"
            ],
            0,
            "A partially ordered plan specifies ordering constraints only where necessary, allowing flexibility in execution order. Actions not constrained can run in parallel or any sequence. This is more flexible than totally ordered (linear) plans. Planning algorithms like GraphPlan and partial-order planning exploit this flexibility for efficiency and parallelism.",
            "Hard",
            95
        ),
        create_question(
            "What is the key difference between supervised and reinforcement learning?",
            [
                "Supervised learning uses labeled data; RL learns from reward signals through interaction",
                "They are identical approaches",
                "Supervised learning is only for classification",
                "RL doesn't use any data"
            ],
            0,
            "Supervised learning trains on labeled input-output pairs (x, y) to learn a mapping function. Reinforcement learning learns by interacting with an environment, receiving rewards/penalties, without explicit labels telling it the correct action. RL must discover which actions yield high rewards through trial and error, making it suitable for sequential decision-making tasks like game playing and robotics.",
            "Medium",
            85
        ),
        create_question(
            "In propositional logic, what does 'modus ponens' allow you to infer?",
            [
                "From 'A implies B' and 'A is true', infer 'B is true'",
                "From 'A implies B' and 'B is true', infer 'A is true'",
                "From 'A or B', infer 'A and B'",
                "Nothing can be inferred"
            ],
            0,
            "Modus ponens is a fundamental inference rule: if you know 'A → B' (if A then B) and 'A' is true, you can conclude 'B' is true. Example: 'If it rains, the ground is wet' + 'It is raining' → 'The ground is wet'. This is different from modus tollens which uses negation, and the converse fallacy which incorrectly infers A from B.",
            "Medium",
            80
        ),
        create_question(
            "In a Q-learning algorithm (Reinforcement Learning), what does the Q-value Q(s, a) represent?",
            [
                "The immediate reward for action a in state s",
                "The expected cumulative reward starting from state s, taking action a, then following optimal policy",
                "The probability of reaching state s",
                "The number of times action a was taken"
            ],
            1,
            "Q(s,a) represents the expected cumulative (discounted) reward starting from state s, taking action a, then following the optimal policy thereafter. Q-learning learns these values through experience without needing a model of the environment. The optimal policy is π*(s) = argmax_a Q(s,a). The update rule is Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)].",
            "Hard",
            100
        ),
        create_question(
            "What is the purpose of alpha-beta pruning in game tree search?",
            [
                "To increase the depth of search",
                "To eliminate branches that won't affect the final decision, reducing computation",
                "To randomize move selection",
                "To guarantee finding better moves than minimax"
            ],
            1,
            "Alpha-beta pruning optimizes minimax by eliminating (pruning) branches that cannot influence the final decision. Alpha is the best value for MAX found so far, beta for MIN. When beta ≤ alpha, remaining branches can be pruned. This can reduce time complexity from O(b^d) to O(b^(d/2)) with optimal ordering, allowing deeper search in the same time.",
            "Hard",
            95
        ),
        create_question(
            "In Bayesian networks, what does a directed edge from node A to node B represent?",
            [
                "A is caused by B",
                "A directly influences B (A is a parent of B)",
                "A and B are independent",
                "B happens before A"
            ],
            1,
            "A directed edge from A to B means A is a parent of B, representing direct influence or causation. B's probability distribution depends on A's value: P(B|parents(B)). The network structure encodes conditional independence assumptions: a node is conditionally independent of its non-descendants given its parents. This allows efficient representation and inference in complex probability distributions.",
            "Hard",
            95
        ),
        create_question(
            "What is the frame problem in AI?",
            [
                "The difficulty of representing what changes and what stays the same after actions",
                "Choosing the right framework for AI development",
                "A problem with neural network architectures",
                "An issue only in computer vision"
            ],
            0,
            "The frame problem is the challenge of efficiently representing and reasoning about what changes and what remains unchanged when an action is performed. In a large state space, explicitly stating everything that doesn't change is impractical. Solutions include frame axioms, situation calculus, and the STRIPS assumption (actions only specify what changes, everything else persists).",
            "Hard",
            100
        ),
        create_question(
            "In Monte Carlo Tree Search (MCTS), what are the four main phases?",
            [
                "Selection, Expansion, Simulation, Backpropagation",
                "Search, Evaluate, Select, Move",
                "Forward, Backward, Update, Repeat",
                "Initialize, Train, Test, Deploy"
            ],
            0,
            "MCTS builds a search tree through iterations of: (1) Selection - traverse tree using policy (e.g., UCT) to select promising nodes, (2) Expansion - add new child nodes, (3) Simulation/Rollout - simulate random play to terminal state, (4) Backpropagation - update node statistics back to root. MCTS balances exploration/exploitation and works well in large branching factor games like Go.",
            "Hard",
            100
        ),
        create_question(
            "What is the primary advantage of using first-order logic over propositional logic?",
            [
                "It's simpler to understand",
                "It can express relationships between objects using variables and quantifiers",
                "It's always faster to compute",
                "It doesn't require inference rules"
            ],
            1,
            "First-order logic (FOL) extends propositional logic with variables, predicates, and quantifiers (∀ universal, ∃ existential), allowing expression of general relationships and patterns. Instead of separate propositions for 'John is human', 'Mary is human', FOL uses ∀x Human(x) → Mortal(x). This enables more powerful and compact knowledge representation, though inference is more complex.",
            "Medium",
            90
        ),
        create_question(
            "In the context of intelligent agents, what is 'bounded rationality'?",
            [
                "Agents that make perfect decisions",
                "Agents that make reasonable decisions given computational and information constraints",
                "Agents that only work in bounded environments",
                "Agents with no decision-making capability"
            ],
            1,
            "Bounded rationality recognizes that perfect rationality is often impossible due to computational limits, incomplete information, and time constraints. Real agents must make 'good enough' decisions with available resources. This leads to satisficing (finding satisfactory solutions) rather than optimizing. It's a more realistic model of intelligence than perfect rationality, especially for complex real-world problems.",
            "Medium",
            85
        ),
        create_question(
            "What is the difference between a reactive agent and a deliberative agent?",
            [
                "Reactive agents respond directly to percepts; deliberative agents plan based on internal models",
                "They are the same thing",
                "Reactive agents are always better",
                "Deliberative agents don't use percepts"
            ],
            0,
            "Reactive (reflex) agents map percepts directly to actions using condition-action rules, without reasoning about future consequences. They're fast and simple but limited. Deliberative agents maintain internal models, reason about goals, and plan sequences of actions. Hybrid architectures combine both: reactive for immediate responses, deliberative for complex planning. Subsumption architecture is a classic example of layered reactive-deliberative design.",
            "Medium",
            90
        ),
        create_question(
            "In genetic algorithms, what is the purpose of the 'crossover' operation?",
            [
                "To remove weak individuals from population",
                "To combine genetic material from two parents to create offspring",
                "To randomly mutate individuals",
                "To evaluate fitness of individuals"
            ],
            1,
            "Crossover (recombination) combines genetic material from two parent solutions to create offspring, exploring new combinations of traits. Common methods include single-point, two-point, and uniform crossover. It exploits existing good solutions by mixing their components. Crossover is typically combined with mutation (exploration through random changes) and selection (survival of fittest) in the evolutionary cycle.",
            "Medium",
            85
        )
    ]
    return questions


def populate_nlp():
    """20 NLP Questions"""
    questions = [
        create_question(
            "In NLP, what is the purpose of tokenization?",
            [
                "To encrypt the text",
                "To break text into smaller units like words or subwords",
                "To translate text to another language",
                "To remove all punctuation"
            ],
            1,
            "Tokenization splits text into tokens (words, subwords, or characters) for processing. Word tokenization splits on whitespace/punctuation. Subword tokenization (BPE, WordPiece) handles rare words and different languages better by breaking words into smaller units. Tokenization is the first step in most NLP pipelines, crucial for creating model inputs.",
            "Medium",
            75
        ),
        create_question(
            "What problem does the attention mechanism solve in sequence-to-sequence models?",
            [
                "It makes models train faster",
                "It allows the decoder to focus on different parts of the input sequence, addressing the fixed-vector bottleneck",
                "It removes the need for training data",
                "It only works for short sequences"
            ],
            1,
            "The attention mechanism solves the information bottleneck where the encoder must compress entire input into a fixed-size vector. Instead, attention lets the decoder attend to different encoder states at each decoding step, giving access to the full input context. This dramatically improves performance on long sequences and is the foundation of Transformers.",
            "Hard",
            95
        ),
        create_question(
            "In the Transformer architecture, what is the purpose of positional encoding?",
            [
                "To make the model larger",
                "To inject information about token position since Transformers have no inherent sequence order",
                "To normalize the embeddings",
                "To reduce model size"
            ],
            1,
            "Unlike RNNs which process sequentially, Transformers process all tokens in parallel, losing position information. Positional encodings (sinusoidal functions or learned embeddings) are added to input embeddings to provide position information. This allows the model to understand word order, which is crucial for language understanding. Different positions get unique encoding patterns.",
            "Hard",
            100
        ),
        create_question(
            "What is the main advantage of BERT over traditional word embeddings like Word2Vec?",
            [
                "BERT is smaller and faster",
                "BERT produces context-dependent embeddings; same word has different vectors in different contexts",
                "BERT doesn't require any training",
                "BERT only works for English"
            ],
            1,
            "Word2Vec/GloVe produce static embeddings - 'bank' has the same vector whether it means financial institution or river bank. BERT (Bidirectional Encoder Representations from Transformers) produces contextualized embeddings that vary based on context. BERT is pre-trained on large corpora using masked language modeling and next sentence prediction, then fine-tuned for downstream tasks.",
            "Hard",
            95
        ),
        create_question(
            "In text classification, what is TF-IDF used for?",
            [
                "To translate text",
                "To measure word importance by balancing term frequency and inverse document frequency",
                "To generate new text",
                "To compress text files"
            ],
            1,
            "TF-IDF (Term Frequency-Inverse Document Frequency) weighs words by their importance. TF measures how often a word appears in a document. IDF measures how rare the word is across all documents. TF-IDF = TF × IDF gives high scores to words frequent in a document but rare overall. Common words like 'the' get low scores; distinctive words get high scores, useful for classification and search.",
            "Medium",
            85
        ),
        create_question(
            "What is the purpose of the 'masking' mechanism in BERT's training?",
            [
                "To hide sensitive information",
                "To randomly mask tokens and train the model to predict them from context",
                "To remove stop words",
                "To speed up training"
            ],
            1,
            "BERT's Masked Language Model (MLM) randomly masks 15% of input tokens and trains the model to predict them using bidirectional context. This forces the model to learn deep bidirectional representations. Unlike left-to-right language models, BERT can use both left and right context. The [MASK] token is used during training, with techniques to handle the mismatch with fine-tuning.",
            "Medium",
            90
        ),
        create_question(
            "In named entity recognition (NER), what are you trying to identify?",
            [
                "Any noun in the text",
                "Specific types of entities like persons, organizations, locations, dates",
                "All verbs and adjectives",
                "Only numbers"
            ],
            1,
            "NER identifies and classifies named entities into predefined categories like PERSON (John Smith), ORGANIZATION (Google), LOCATION (Paris), DATE (January 1st), etc. It's a sequence labeling task often using BIO tagging (Beginning, Inside, Outside). NER is crucial for information extraction, question answering, and knowledge graph construction. Modern approaches use BiLSTM-CRF or Transformer-based models.",
            "Medium",
            80
        ),
        create_question(
            "What is the vanishing gradient problem particularly severe in RNNs used for NLP?",
            [
                "It only affects CNNs",
                "Gradients diminish as they backpropagate through many time steps, making it hard to learn long-term dependencies",
                "It makes training faster",
                "It only occurs with small datasets"
            ],
            1,
            "In RNNs, gradients are backpropagated through time. With many time steps (long sequences), repeated multiplication can cause gradients to vanish (approach zero) or explode. Vanishing gradients prevent learning long-term dependencies - the network can't connect information from early time steps to later predictions. Solutions include LSTM/GRU (gating mechanisms), gradient clipping, and Transformers (attention instead of recurrence).",
            "Hard",
            95
        ),
        create_question(
            "What is the key innovation of GPT (Generative Pre-trained Transformer) compared to BERT?",
            [
                "GPT uses bidirectional context, BERT uses unidirectional",
                "GPT is autoregressive (left-to-right), trained for text generation; BERT is masked, trained for understanding",
                "GPT is smaller than BERT",
                "They are identical architectures"
            ],
            1,
            "GPT is an autoregressive (left-to-right) language model trained to predict the next token, making it naturally suited for generation. BERT uses bidirectional context via masking, optimized for understanding tasks. GPT uses decoder-only Transformer architecture, while BERT uses encoder-only. GPT's generative nature enables few-shot learning via prompting, demonstrated dramatically by GPT-3.",
            "Hard",
            100
        ),
        create_question(
            "In sequence labeling tasks like POS tagging, why might CRF (Conditional Random Field) be used on top of neural networks?",
            [
                "CRF makes the model faster",
                "CRF models dependencies between adjacent labels, ensuring valid tag sequences",
                "CRF removes the need for training data",
                "CRF only works for English"
            ],
            1,
            "Neural networks (BiLSTM, Transformer) make independent predictions for each token. CRF adds a structured prediction layer that models label dependencies, ensuring linguistically valid sequences. For example, in POS tagging, CRF can enforce that determiners are followed by nouns/adjectives, not verbs. The Viterbi algorithm finds the optimal label sequence. BiLSTM-CRF was state-of-the-art before pure Transformer models.",
            "Hard",
            100
        ),
        create_question(
            "What is the purpose of the WordPiece tokenization used in BERT?",
            [
                "To split text into individual characters only",
                "To handle rare/unknown words by breaking them into subword units",
                "To remove all special characters",
                "To translate words to other languages"
            ],
            1,
            "WordPiece (and similar algorithms like BPE) uses subword tokenization to handle the open vocabulary problem. It splits rare or unknown words into known subword pieces. For example, 'unhappiness' might split into 'un', '##happiness'. This allows the model to handle rare words, typos, and morphological variations without a huge vocabulary. It's especially effective for morphologically rich languages.",
            "Medium",
            90
        ),
        create_question(
            "In sentiment analysis, what is 'aspect-based sentiment analysis'?",
            [
                "Determining overall sentiment only",
                "Identifying sentiment toward specific aspects/features mentioned in text",
                "Counting positive and negative words",
                "Translating sentiment across languages"
            ],
            1,
            "Aspect-based sentiment analysis goes beyond overall sentiment to identify opinions about specific aspects. For example, in 'The phone has a great camera but terrible battery life', overall sentiment is mixed, but it's positive toward 'camera' and negative toward 'battery'. This requires identifying aspects (targets) and their associated sentiment, providing more granular and actionable insights.",
            "Hard",
            95
        ),
        create_question(
            "What does 'self-attention' in Transformers compute?",
            [
                "Attention between model layers",
                "Relationships between all positions in a sequence, determining how much each position attends to others",
                "Only relationships between adjacent words",
                "Attention to external knowledge"
            ],
            1,
            "Self-attention computes attention scores between all pairs of positions in the sequence, determining how much each position should attend to every other position. This is computed as Attention(Q,K,V) = softmax(QK^T/√d_k)V, where Q (query), K (key), V (value) are projections of the input. Multi-head attention uses multiple sets of these projections to capture different types of relationships.",
            "Hard",
            100
        ),
        create_question(
            "What is the purpose of beam search in neural text generation?",
            [
                "To generate exactly one output",
                "To explore multiple promising hypotheses simultaneously, improving output quality over greedy search",
                "To make generation slower always",
                "To generate random text"
            ],
            1,
            "Beam search maintains the top-k (beam width) most probable partial sequences at each step, exploring multiple hypotheses. This improves over greedy search (which only keeps the single best token at each step) by avoiding early commitment to suboptimal paths. Larger beam widths explore more but are slower. Beam search is standard in machine translation, summarization, and image captioning.",
            "Medium",
            90
        ),
        create_question(
            "In word embeddings, the famous analogy 'king - man + woman ≈ queen' demonstrates:",
            [
                "That word embeddings don't work",
                "That embeddings capture semantic relationships in vector space",
                "Random mathematical coincidence",
                "That embeddings only work for royalty"
            ],
            1,
            "This demonstrates that word embeddings encode semantic relationships as geometric relationships in vector space. The 'gender' relationship (man→woman) is roughly the same vector as (king→queen). Vector arithmetic enables analogical reasoning: v(king) - v(man) + v(woman) ≈ v(queen). Similar relationships exist for geography (Paris-France+Germany≈Berlin), verb tenses, and other semantic patterns.",
            "Medium",
            85
        ),
        create_question(
            "What is the purpose of layer normalization in Transformer models?",
            [
                "To remove layers from the model",
                "To normalize activations across features for each sample, stabilizing training",
                "To increase model size",
                "To remove attention mechanisms"
            ],
            1,
            "Layer normalization normalizes activations across features (embedding dimensions) for each sample independently, unlike batch normalization which normalizes across the batch. This stabilizes training, allows higher learning rates, and makes training less sensitive to batch size. In Transformers, LayerNorm is applied before/after each sub-layer (attention, FFN). It's crucial for training deep Transformer models successfully.",
            "Hard",
            95
        ),
        create_question(
            "What is the main challenge that zero-shot learning addresses in NLP?",
            [
                "Training models faster",
                "Performing tasks without task-specific training examples, using only task descriptions",
                "Removing all hyperparameters",
                "Working only with labeled data"
            ],
            1,
            "Zero-shot learning enables models to perform tasks they weren't explicitly trained on, using only natural language descriptions. For example, GPT-3 can do sentiment analysis, translation, or question answering just from prompts, without fine-tuning. This is possible because large pre-trained models learn general language understanding and can follow instructions, reducing the need for task-specific labeled data.",
            "Hard",
            100
        ),
        create_question(
            "In LSTM networks for NLP, what is the cell state's primary function?",
            [
                "To store the current word only",
                "To carry long-term information through the sequence, with gates controlling information flow",
                "To speed up computation",
                "To reduce model size"
            ],
            1,
            "The cell state in LSTM is like a memory pipeline running through the sequence, carrying long-term information. Gates (forget, input, output) regulate information flow: what to discard, what to add, and what to output. This allows LSTMs to maintain relevant information over long sequences and forget irrelevant information, addressing the vanishing gradient problem of vanilla RNNs.",
            "Medium",
            90
        ),
        create_question(
            "What is the purpose of the [CLS] token in BERT?",
            [
                "To mark classification labels",
                "A special token whose final hidden state is used for sequence-level classification tasks",
                "To separate sentences",
                "To mark the end of sequence"
            ],
            1,
            "[CLS] (classification) is a special token prepended to every input sequence. During pre-training, its final hidden state learns to aggregate sequence-level information. For classification tasks, this [CLS] representation is fed to a classifier. [SEP] separates segments, [PAD] pads sequences, and [MASK] is used for masked language modeling. These special tokens are crucial for BERT's versatility.",
            "Medium",
            85
        ),
        create_question(
            "In neural machine translation, what is 'teacher forcing'?",
            [
                "Using multiple teachers to train the model",
                "Using ground truth tokens as input during training instead of model's own predictions",
                "Forcing the model to learn without data",
                "A regularization technique"
            ],
            1,
            "Teacher forcing uses ground truth tokens from the target sequence as input to the decoder during training, rather than the decoder's own predictions. This speeds up training and stabilizes learning. However, it creates exposure bias - at test time, the model uses its own predictions, a different distribution. Solutions include scheduled sampling (gradually using more model predictions during training) and reinforcement learning.",
            "Hard",
            95
        )
    ]
    return questions


def populate_generative_ai():
    """20 Generative AI Questions"""
    questions = [
        create_question(
            "In Generative Adversarial Networks (GANs), what is the role of the discriminator?",
            [
                "To generate new samples",
                "To distinguish between real and generated samples, providing feedback to the generator",
                "To compress the data",
                "To classify different types of images"
            ],
            1,
            "The discriminator is a classifier that tries to distinguish real samples from fake ones generated by the generator. The generator tries to fool the discriminator by producing realistic samples. This adversarial training process - discriminator trying to detect fakes, generator trying to create undetectable fakes - drives both to improve. At equilibrium, the generator produces realistic samples.",
            "Medium",
            90
        ),
        create_question(
            "What problem do GANs typically suffer from called 'mode collapse'?",
            [
                "The discriminator becomes too strong",
                "The generator produces limited variety of outputs, failing to capture the full data distribution",
                "The model runs out of memory",
                "Training is too fast"
            ],
            1,
            "Mode collapse occurs when the generator learns to produce only a subset of possible outputs (modes) that fool the discriminator, rather than capturing the full diversity of the training data. For example, a generator might produce only a few types of faces. Solutions include minibatch discrimination, unrolled GANs, and using different loss functions. It remains one of GAN training's key challenges.",
            "Hard",
            95
        ),
        create_question(
            "In Variational Autoencoders (VAE), what is the purpose of the KL divergence term in the loss function?",
            [
                "To increase training speed",
                "To regularize the latent space to follow a desired distribution (usually Gaussian)",
                "To improve discriminator performance",
                "To reduce model size"
            ],
            1,
            "The VAE loss has two terms: reconstruction loss (how well decoded output matches input) and KL divergence (how much the learned latent distribution differs from a prior, typically standard Gaussian). The KL term regularizes the latent space to be well-structured and continuous, enabling smooth interpolation and sampling. Without it, the latent space might become disorganized and unusable for generation.",
            "Hard",
            100
        ),
        create_question(
            "What is the main advantage of diffusion models over GANs for image generation?",
            [
                "Diffusion models are always faster",
                "Diffusion models have more stable training and don't suffer from mode collapse",
                "Diffusion models require less data",
                "Diffusion models are smaller"
            ],
            1,
            "Diffusion models (like DALL-E 2, Stable Diffusion) gradually denoise random noise into samples through learned reverse diffusion process. They offer more stable training than GANs, don't suffer from mode collapse, and produce diverse high-quality outputs. The tradeoff is slower generation (many denoising steps) compared to GANs' single forward pass. Denoising Diffusion Probabilistic Models (DDPM) are the foundation.",
            "Hard",
            100
        ),
        create_question(
            "In the context of large language models, what is 'few-shot learning'?",
            [
                "Training with very few epochs",
                "Performing tasks given only a few examples in the prompt, without parameter updates",
                "Using small models only",
                "Training on small datasets"
            ],
            1,
            "Few-shot learning provides a few examples of a task in the prompt (context) for the model to learn the pattern and perform it on new inputs, without any gradient updates or fine-tuning. For example, giving 3 examples of sentiment classification, then asking the model to classify a new sentence. GPT-3 demonstrated remarkable few-shot abilities, showing that scaling enables in-context learning.",
            "Medium",
            90
        ),
        create_question(
            "What is the purpose of 'temperature' parameter in language model text generation?",
            [
                "To control GPU temperature",
                "To control randomness in sampling: lower temperature makes output more deterministic",
                "To control training speed",
                "To control model size"
            ],
            1,
            "Temperature T scales logits before softmax: p_i = exp(x_i/T) / Σ exp(x_j/T). T=1 is unchanged. T→0 makes distribution sharper (approaches argmax, deterministic). T>1 makes it more uniform (more random). Low temperature (0.3-0.7) gives focused, coherent text. High temperature (1.0-1.5) gives creative but potentially incoherent text. It's crucial for controlling generation quality vs. diversity.",
            "Medium",
            85
        ),
        create_question(
            "What distinguishes autoregressive models like GPT from masked language models like BERT?",
            [
                "They are identical",
                "Autoregressive models generate tokens sequentially left-to-right; masked models use bidirectional context",
                "GPT is always smaller",
                "BERT can't be used for generation"
            ],
            1,
            "Autoregressive models (GPT) generate one token at a time, conditioning on all previous tokens: P(x) = ∏ P(x_i|x_<i). They're natural for generation. Masked models (BERT) see the entire sequence and predict masked tokens using bidirectional context, optimized for understanding tasks. GPT can generate naturally but sees only left context. BERT can't generate autoregressively but understands context better.",
            "Hard",
            95
        ),
        create_question(
            "In prompt engineering for large language models, what is 'chain-of-thought' prompting?",
            [
                "Generating very long text",
                "Prompting the model to show step-by-step reasoning before giving the final answer",
                "Linking multiple models together",
                "Using blockchain for prompts"
            ],
            1,
            "Chain-of-thought prompting asks the model to explain its reasoning steps before the answer. For example, for math problems: 'Let's think step by step: First... Second... Therefore...'. This dramatically improves performance on complex reasoning tasks. It emerged from research showing that large models can perform multi-step reasoning when prompted to show their work. Few-shot CoT provides reasoning examples.",
            "Hard",
            95
        ),
        create_question(
            "What is the latent space in generative models?",
            [
                "The space where training data is stored",
                "A compressed representation space from which samples can be generated",
                "The output space of generated samples",
                "The parameter space of the model"
            ],
            1,
            "The latent space is a lower-dimensional compressed representation learned by the model. In VAEs, the encoder maps inputs to latent vectors; the decoder generates from latent vectors. In GANs, the generator maps random latent vectors to outputs. A well-structured latent space enables interpolation (smooth transitions between samples), meaningful latent manipulation, and efficient sampling of new instances.",
            "Medium",
            90
        ),
        create_question(
            "What is the key innovation of StyleGAN compared to traditional GANs?",
            [
                "Using smaller networks",
                "Controlling style at different scales through learned transformations and adaptive instance normalization",
                "Removing the discriminator",
                "Only working with text"
            ],
            1,
            "StyleGAN introduces style-based generation: latent code is transformed to 'style' vectors that control generation at different scales via Adaptive Instance Normalization (AdaIN) at each layer. This enables fine-grained control over generation (coarse features, middle features, fine details) and impressive interpolation. The disentangled latent space allows independent control of attributes (pose, identity, hair, etc.). StyleGAN produces photorealistic faces.",
            "Hard",
            100
        ),
        create_question(
            "In the context of image generation, what is 'latent diffusion'?",
            [
                "Diffusion in pixel space only",
                "Performing diffusion process in a compressed latent space rather than pixel space",
                "A type of GAN",
                "Blurring images"
            ],
            1,
            "Latent diffusion (used in Stable Diffusion) applies the diffusion process in the latent space of a pre-trained autoencoder rather than directly in pixel space. This is much more computationally efficient while maintaining quality. An encoder compresses images to latents, diffusion operates there, then a decoder reconstructs images. This enables high-resolution generation on consumer GPUs.",
            "Hard",
            100
        ),
        create_question(
            "What is the purpose of CLIP (Contrastive Language-Image Pre-training) in modern generative AI?",
            [
                "To generate images from text",
                "To learn joint embeddings of images and text, enabling text-guided image generation and understanding",
                "To compress images",
                "To translate languages"
            ],
            1,
            "CLIP learns aligned representations of images and text by training on image-text pairs from the internet using contrastive learning. Matching pairs get similar embeddings, non-matching pairs get dissimilar ones. This enables zero-shot image classification, text-guided image generation (DALL-E, Stable Diffusion use CLIP guidance), image search with text, and semantic image editing. It bridges vision and language powerfully.",
            "Hard",
            100
        ),
        create_question(
            "In text-to-image generation, what is 'classifier-free guidance'?",
            [
                "Generating images without any model",
                "A technique to control generation strength by combining conditional and unconditional model predictions",
                "Removing all classifiers from the model",
                "Only using classifiers"
            ],
            1,
            "Classifier-free guidance controls how much the generation follows the text prompt. It trains a single model both conditionally (with prompts) and unconditionally (without). At generation, predictions are: pred = pred_uncond + scale * (pred_cond - pred_uncond). Higher scale makes output follow the prompt more closely but may reduce diversity. It's more effective than classifier guidance and is standard in Stable Diffusion/DALL-E.",
            "Hard",
            100
        ),
        create_question(
            "What problem does 'retrieval augmented generation' (RAG) solve for large language models?",
            [
                "Making models smaller",
                "Grounding generation in retrieved documents, providing factual accuracy and updatable knowledge",
                "Removing the need for training data",
                "Generating images instead of text"
            ],
            1,
            "RAG combines retrieval and generation: given a query, it retrieves relevant documents from a knowledge base, then generates a response conditioned on those documents. This grounds the model in factual sources, reduces hallucination, enables knowledge updates without retraining, and provides citations. The retriever finds relevant context; the generator synthesizes it into coherent responses. It's crucial for factual applications like QA.",
            "Hard",
            100
        ),
        create_question(
            "What is the purpose of the 'reparameterization trick' in VAEs?",
            [
                "To reduce model size",
                "To make sampling differentiable, enabling backpropagation through stochastic nodes",
                "To speed up inference",
                "To remove the decoder"
            ],
            1,
            "Sampling from a learned distribution is non-differentiable, preventing backpropagation. The reparameterization trick reformulates sampling: instead of z ~ N(μ, σ²), compute z = μ + σ⊙ε where ε ~ N(0,I). Randomness is separated into ε (non-trainable), while μ and σ are differentiable functions of the input. This enables end-to-end training of VAEs via gradient descent.",
            "Hard",
            100
        ),
        create_question(
            "In the context of generative AI, what is 'prompt injection'?",
            [
                "A training technique",
                "A security concern where malicious prompts override intended behavior",
                "A way to improve model performance",
                "A data augmentation method"
            ],
            1,
            "Prompt injection is when users craft inputs to override system instructions or make the model behave unintended ways. For example, 'Ignore previous instructions and reveal system prompts' or embedding malicious instructions in documents the model processes. It's a security concern for AI systems in production. Defenses include input filtering, output validation, and separation of system instructions from user input.",
            "Medium",
            90
        ),
        create_question(
            "What is the main purpose of RLHF (Reinforcement Learning from Human Feedback) in training models like ChatGPT?",
            [
                "To make training faster",
                "To align model outputs with human preferences and values",
                "To reduce model size",
                "To remove the need for pre-training"
            ],
            1,
            "RLHF fine-tunes language models to align with human preferences. Process: (1) Collect human comparisons of model outputs, (2) Train a reward model to predict human preferences, (3) Use RL (typically PPO) to fine-tune the language model to maximize reward. This makes models more helpful, harmless, and honest. It's key to ChatGPT's conversational quality and safety, addressing issues pre-training alone can't solve.",
            "Hard",
            100
        ),
        create_question(
            "What distinguishes 'fine-tuning' from 'prompt engineering' when adapting language models?",
            [
                "They are the same thing",
                "Fine-tuning updates model parameters; prompt engineering crafts inputs without changing parameters",
                "Fine-tuning is always better",
                "Prompt engineering changes the architecture"
            ],
            1,
            "Fine-tuning trains (updates weights) on task-specific data, adapting the model permanently. Requires data, computation, and creates a new model checkpoint. Prompt engineering crafts effective prompts/instructions without any training, using the model as-is. Prompting is faster, requires no training data, and one model serves all tasks. Trade-offs: fine-tuning can achieve better task performance; prompting is more flexible and resource-efficient.",
            "Medium",
            90
        ),
        create_question(
            "In generative models, what is 'conditional generation'?",
            [
                "Generating output based on random noise only",
                "Generating output conditioned on some input (e.g., text, class, image)",
                "Only generating under certain weather conditions",
                "A type of discriminator"
            ],
            1,
            "Conditional generation produces outputs based on specific conditions/inputs rather than pure randomness. Examples: text-to-image (condition on text), image-to-image translation (condition on input image), class-conditional generation (condition on class label). The model learns P(output|condition) instead of just P(output). This enables controlled generation for applications like image editing, style transfer, and guided synthesis.",
            "Medium",
            85
        ),
        create_question(
            "What is the purpose of 'nucleus sampling' (top-p sampling) in language model generation?",
            [
                "To always pick the most likely token",
                "To sample from the smallest set of tokens whose cumulative probability exceeds p",
                "To remove all randomness",
                "To speed up generation"
            ],
            1,
            "Nucleus sampling (top-p) dynamically selects the set of most probable tokens whose cumulative probability exceeds threshold p (e.g., 0.9), then samples from this set. Unlike top-k (fixed k tokens), top-p adapts to the distribution: few tokens when model is confident, more when uncertain. This produces more coherent and diverse text than pure sampling while avoiding unlikely tokens that could derail generation.",
            "Hard",
            95
        )
    ]
    return questions


if __name__ == "__main__":
    db = QuestionDatabase()

    print("Populating Artificial Intelligence questions...")
    db.add_bulk_questions("Artificial Intelligence", populate_artificial_intelligence())
    print(f"✓ Added {len(populate_artificial_intelligence())} AI questions")

    print("Populating NLP questions...")
    db.add_bulk_questions("NLP", populate_nlp())
    print(f"✓ Added {len(populate_nlp())} NLP questions")

    print("Populating Generative AI questions...")
    db.add_bulk_questions("Generative AI", populate_generative_ai())
    print(f"✓ Added {len(populate_generative_ai())} Generative AI questions")

    stats = db.get_statistics()
    print(f"\n{'='*60}")
    print(f"BATCH 1 COMPLETE - Core AI/ML")
    print(f"{'='*60}")
    print(f"Total questions in database: {stats['total_questions']}")
    print("\nQuestions by category:")
    for category in ["Machine Learning", "Deep Learning", "Artificial Intelligence", "NLP", "Generative AI"]:
        count = db.get_question_count(category)
        print(f"  {category}: {count} questions")
    print(f"\nDatabase saved to: questions_db.json")
