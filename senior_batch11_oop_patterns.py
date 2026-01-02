"""
Senior AI Engineer Interview Questions - Batch 11: OOP & Design Patterns for ML
Topics: Factory, Strategy, Observer, Pipeline, Singleton, Abstract Classes
Level: Expert/Senior (5+ years experience)
Total: 20 Questions
"""

from database_manager import QuestionDatabase, create_question

def populate_senior_oop_patterns():
    """20 Senior-Level OOP & Design Patterns Questions"""
    questions = [
        # FACTORY PATTERN (Questions 1-4)
        create_question(
            "Q1: You're building a model registry that instantiates different model architectures (ResNet, ViT, BERT) from config. Which pattern is most appropriate?",
            [
                "Singleton - ensures only one model instance",
                "Factory pattern - creates model instances based on type parameter without exposing instantiation logic",
                "Strategy pattern - different model behaviors",
                "Observer pattern - model state changes"
            ],
            1,
            "Senior Explanation: Factory pattern encapsulates object creation. ModelFactory.create(model_type='resnet50', num_classes=1000) returns appropriate model without client knowing instantiation details. Benefits: (1) Centralized model creation (easy to add new architectures), (2) Config-driven instantiation (JSON config → model), (3) Dependency injection (inject pretrained weights, custom layers). Implementation: class ModelFactory: @staticmethod def create(model_type, **kwargs): if model_type == 'resnet50': return ResNet50(**kwargs); elif model_type == 'vit': return VisionTransformer(**kwargs). Production: Hugging Face AutoModel.from_pretrained() is factory pattern. Trade-off: Factory adds abstraction layer (slight complexity) but crucial for scalable ML systems with many model types.",
            "Hard",
            200
        ),
        create_question(
            "Q2: In an ML pipeline factory, you need to create different preprocessing pipelines (image, text, audio). What's the best way to ensure type safety and extensibility?",
            [
                "Use if-else chain to check pipeline type",
                "Abstract factory with base PreprocessingPipeline class - subclasses implement process() method, factory returns appropriate subclass",
                "Dictionary mapping pipeline names to classes",
                "Separate factory for each data type"
            ],
            1,
            "Senior Explanation: Abstract Factory: Define interface (base class PreprocessingPipeline with abstract process() method). Concrete implementations (ImagePipeline, TextPipeline) inherit and implement process(). Factory returns base type but actual instance is subclass. Benefits: (1) Type checking (all pipelines conform to interface), (2) Extensibility (add VideoFipeline by inheriting base), (3) Polymorphism (client code works with base type). Code: class PreprocessingPipeline(ABC): @abstractmethod def process(self, data): pass. Class ImagePipeline(PreprocessingPipeline): def process(self, data): return normalize(resize(data)). Option A 'junior trap' - no type safety. Option C works but less structured. Production: Scikit-learn pipelines use this pattern. Trade-off: Abstraction overhead for maintainability and extensibility.",
            "Hard",
            220
        ),
        create_question(
            "Q3: For a model serving system that needs to switch between different model versions (A/B testing), which pattern combination is optimal?",
            [
                "Factory + Strategy - Factory creates models, Strategy selects which to use",
                "Singleton + Observer - Single model instance with state observation",
                "Builder + Prototype - Build and clone models",
                "Facade + Adapter - Wrap model interfaces"
            ],
            0,
            "Senior Explanation: Factory creates model instances (v1, v2). Strategy pattern selects which model to serve based on criteria (user ID hash, traffic percentage, experiment group). Implementation: class ModelSelector: def select_model(self, user_id): if hash(user_id) % 100 < 10: return factory.create('model_v2') else: return factory.create('model_v1'). Benefits: (1) Decouple model creation from selection logic, (2) Easy to change selection strategy (random, weighted, feature-based), (3) Models lifecycle managed separately from routing. Production: TF-Serving, Ray Serve use similar patterns for A/B testing. Trade-off: Two patterns increase complexity but provide flexibility for production ML systems with frequent model updates.",
            "Hard",
            220
        ),
        create_question(
            "Q4: When should you use Factory pattern over simple class instantiation in ML systems?",
            [
                "Always - Factory is always better",
                "When you have 3+ model types and need config-driven instantiation, or when instantiation logic is complex (loading weights, device placement)",
                "Never - too much abstraction for ML code",
                "Only for inference, not training"
            ],
            1,
            "Senior Explanation: Use Factory when: (1) Multiple model variants (ResNet18/34/50/101/152 - factory avoids 5 separate imports), (2) Complex instantiation (load checkpoint, move to GPU, wrap in DDP, compile with torch.compile), (3) Config-driven (YAML config → model), (4) Testing (factory can return mocks). Don't use for: Simple single-model scripts, notebooks, prototyping. Code smell: If creating model requires >5 lines of boilerplate, factor into factory. Production: Training frameworks (Lightning, Trainer classes) use factories extensively. Trade-off: 1-2 model types - direct instantiation simpler. 5+ types - factory essential for maintainability. Memory: Factory creates instances on-demand (no overhead).",
            "Medium",
            180
        ),

        # STRATEGY PATTERN (Questions 5-8)
        create_question(
            "Q5: You have multiple training strategies (standard, mixed precision, distributed). How should you structure this with Strategy pattern?",
            [
                "Separate training scripts for each strategy",
                "Define TrainingStrategy interface with train_step() method - concrete strategies (StandardTraining, MixedPrecisionTraining, DistributedTraining) implement interface",
                "Use global flags to switch behavior",
                "Inheritance chain with base Trainer class"
            ],
            1,
            "Senior Explanation: Strategy pattern: Define interface for interchangeable algorithms. Interface: class TrainingStrategy(ABC): @abstractmethod def train_step(self, model, batch): pass. Implementations: class MixedPrecisionTraining(TrainingStrategy): def train_step(self, model, batch): with autocast(): loss = model(batch); scaler.scale(loss).backward(). Usage: trainer = Trainer(strategy=MixedPrecisionTraining()) - swap strategies without changing client code. Benefits: (1) Easy to add new strategies (LoRA training, gradient accumulation), (2) Test strategies independently, (3) Runtime strategy selection based on hardware. Production: PyTorch Lightning strategies (ddp, fsdp, deepspeed) use this. Trade-off: More classes (one per strategy) but cleaner than if-else chains.",
            "Hard",
            200
        ),
        create_question(
            "Q6: For learning rate scheduling (constant, linear decay, cosine annealing), which pattern is most maintainable?",
            [
                "Single Scheduler class with mode parameter",
                "Strategy pattern - SchedulingStrategy interface with get_lr(step) method, different strategies implement scheduling logic",
                "Lambda functions passed to optimizer",
                "Hard-coded in training loop"
            ],
            1,
            "Senior Explanation: Strategy pattern for LR schedules: Interface: class LRSchedule(ABC): @abstractmethod def get_lr(self, step, base_lr): pass. Strategies: class CosineAnnealing(LRSchedule): def get_lr(self, step, base_lr): return base_lr * 0.5 * (1 + cos(pi * step / max_steps)). Usage: scheduler = CosineAnnealing(); for step in range(max_steps): lr = scheduler.get_lr(step, base_lr). Benefits: (1) Add custom schedules easily (warmup + cosine, polynomial), (2) Unit test schedules independently, (3) Config-driven selection. Option A (mode parameter) becomes bloated with many schedules. Production: Transformers library has ~15 schedule strategies. Trade-off: Strategies cleanly separate concerns but require more files.",
            "Medium",
            180
        ),
        create_question(
            "Q7: You need different data augmentation strategies for training (strong augment) vs validation (minimal augment). Best pattern?",
            [
                "Two separate augmentation functions",
                "Strategy pattern - AugmentationStrategy with apply(image) method, TrainingAugmentation and ValidationAugmentation strategies",
                "Boolean flag in single augmentation function",
                "Duplicate pipeline code"
            ],
            1,
            "Senior Explanation: Augmentation strategies: class AugmentationStrategy(ABC): @abstractmethod def apply(self, image): pass. Class TrainingAugmentation(AugmentationStrategy): def apply(self, image): return random_crop(flip(color_jitter(image))). Class ValidationAugmentation(AugmentationStrategy): def apply(self, image): return center_crop(image). Dataset: class ImageDataset: def __init__(self, ..., augmentation_strategy): self.augmentation = augmentation_strategy; def __getitem__(self, idx): return self.augmentation.apply(load_image(idx)). Benefits: (1) Easy to add test-time augmentation (TTA), (2) Reusable across datasets, (3) Composable (chain strategies). Production: Albumentations library uses similar composition. Trade-off: Strategy pattern clearer intent than boolean flags.",
            "Medium",
            180
        ),
        create_question(
            "Q8: For loss functions (CrossEntropy, Focal Loss, Label Smoothing), should you use Strategy pattern?",
            [
                "No - PyTorch/TensorFlow provide loss modules, use directly",
                "Yes - Strategy pattern allows runtime loss selection and custom losses without modifying training code",
                "Only for custom losses",
                "Use inheritance hierarchy"
            ],
            1,
            "Senior Explanation: Loss strategy useful when: (1) Experiment with multiple losses (cross-entropy vs focal for imbalanced data), (2) Custom losses (contrastive, triplet), (3) Composite losses (classification + bbox regression in detection). Implementation: class LossStrategy(ABC): @abstractmethod def compute(self, predictions, targets): pass. Class FocalLoss(LossStrategy): def compute(self, pred, target): return focal_loss(pred, target, gamma=2). Trainer: loss_fn = FocalLoss(); loss = loss_fn.compute(outputs, labels). Direct usage (option A) works for simple cases but Strategy enables: Config-driven loss selection, Easy experimentation, Reusable loss implementations. Production: Detection frameworks (MMDetection) use loss registries (factory + strategy). Trade-off: Overkill for single loss, valuable for frameworks supporting multiple tasks.",
            "Medium",
            180
        ),

        # OBSERVER PATTERN (Questions 9-12)
        create_question(
            "Q9: For logging metrics during training (TensorBoard, Weights & Biases, CSV), which pattern avoids tight coupling?",
            [
                "Hard-code logging calls in training loop",
                "Observer pattern - Trainer emits events (on_epoch_end, on_batch_end), observers (loggers) subscribe and handle events",
                "Callbacks passed as functions",
                "Global logging singleton"
            ],
            1,
            "Senior Explanation: Observer pattern: Trainer has list of observers, notifies them on events. Class TrainingObserver(ABC): @abstractmethod def on_epoch_end(self, epoch, metrics): pass. Class TensorBoardObserver(TrainingObserver): def on_epoch_end(self, epoch, metrics): writer.add_scalar('loss', metrics['loss'], epoch). Trainer: def train(self): for epoch in epochs: metrics = train_epoch(); for observer in self.observers: observer.on_epoch_end(epoch, metrics). Benefits: (1) Add/remove loggers without changing training code, (2) Multiple observers simultaneously (TensorBoard + WandB + CSV), (3) Observers can have side effects (checkpointing, early stopping). Option C (callbacks) similar but less structured. Production: PyTorch Lightning, Keras callbacks are observer pattern. Trade-off: Slight overhead (notification loops) for decoupled, extensible logging.",
            "Hard",
            220
        ),
        create_question(
            "Q10: For distributed training, you need to synchronize callbacks (checkpointing, early stopping) across workers. How does Observer pattern help?",
            [
                "Observers run independently per worker - no synchronization",
                "Observers can implement distributed-aware logic - e.g., CheckpointObserver only saves on rank 0, EarlyStoppingObserver uses allreduce for validation loss",
                "Observer pattern doesn't work in distributed setting",
                "Requires separate pattern for distributed"
            ],
            1,
            "Senior Explanation: Distributed-aware observers: Class CheckpointObserver(TrainingObserver): def on_epoch_end(self, epoch, metrics): if get_rank() == 0: save_checkpoint(model, epoch). Class EarlyStoppingObserver: def on_epoch_end(self, epoch, metrics): val_loss = allreduce(metrics['val_loss'], op=MIN); if self.should_stop(val_loss): self.trainer.stop_training(). Benefits: (1) Encapsulates distributed logic in observers (training loop stays simple), (2) Easy to test (mock distributed ops), (3) Reusable across projects. Production: PyTorch Lightning callbacks handle distributed automatically (checkpoint on rank 0, early stopping synced). Trade-off: Observers must be distributed-aware (complexity) but keeps training code clean.",
            "Hard",
            220
        ),
        create_question(
            "Q11: How many observers can attach to a single subject (Trainer) efficiently?",
            [
                "1-2 - more causes performance issues",
                "5-10 - typical production setup (TensorBoard, checkpointing, early stopping, profiling, custom metrics)",
                "100+ - observers are very lightweight",
                "Unlimited - no performance impact"
            ],
            1,
            "Senior Explanation: Typical production: 5-10 observers (TensorBoard, WandB, model checkpointing, early stopping, learning rate logging, gradient norm logging, custom validation). Each observer adds ~0.1-1ms overhead per notification (callback invocation, metric logging). For 1000 steps/epoch, 10 observers: ~10-100ms total (negligible vs training time ~minutes-hours). Option C/D overestimate - 100 observers would add clutter and maintenance burden. Option A too conservative. Production: PyTorch Lightning supports ~20+ built-in callbacks, users typically use 5-10. Memory: Each observer ~1-10KB (state variables). Trade-off: More observers → more visibility but complex coordination. Keep essential observers only.",
            "Medium",
            180
        ),
        create_question(
            "Q12: Observer pattern vs Callback functions - when to use which in ML training?",
            [
                "Always use Observer - more OOP",
                "Observer for stateful monitoring (early stopping, checkpointing), simple callbacks for stateless operations (logging single metric)",
                "Always use callbacks - simpler",
                "No difference - same pattern"
            ],
            1,
            "Senior Explanation: Observer pattern: Use for stateful behavior (EarlyStoppingObserver tracks best loss, patience counter). Callbacks: Use for stateless operations (log metric, save visualization). Observer benefits: (1) Encapsulates state (self.best_loss), (2) Multiple methods (on_train_begin, on_epoch_end, on_train_end), (3) Inheritance (BaseObserver with common logic). Callback benefits: (1) Simpler for one-off tasks, (2) Less boilerplate (no class definition). Example: def log_lr(epoch, lr): print(f'{epoch}: {lr}') - simple callback. Class EarlyStoppingObserver: maintains state, complex logic. Production: Frameworks support both - Keras callbacks (observers), PyTorch hooks (simple callbacks). Trade-off: Observers for complex, reusable logic; callbacks for quick, simple tasks.",
            "Medium",
            180
        ),

        # PIPELINE & BUILDER PATTERNS (Questions 13-16)
        create_question(
            "Q13: For data preprocessing (tokenization → padding → batching), which pattern ensures clean composition?",
            [
                "Monolithic preprocessing function",
                "Pipeline pattern - chain of transformations, each implementing transform() method",
                "Nested function calls",
                "Sequential if-else logic"
            ],
            1,
            "Senior Explanation: Pipeline pattern: Class Transformation(ABC): @abstractmethod def transform(self, data): pass. Concrete: Class Tokenizer(Transformation): def transform(self, text): return tokenize(text). Class Padder(Transformation): def transform(self, tokens): return pad(tokens, max_len). Pipeline: Class PreprocessingPipeline: def __init__(self, transformations): self.steps = transformations; def process(self, data): for step in self.steps: data = step.transform(data); return data. Usage: pipeline = PreprocessingPipeline([Tokenizer(), Padder(), Batcher()]); processed = pipeline.process(raw_text). Benefits: (1) Composable (add/remove steps), (2) Reusable (share Tokenizer across pipelines), (3) Testable (test each step independently). Production: Scikit-learn Pipeline, Hugging Face Datasets.map() use this. Trade-off: Slight overhead (loop through steps) for massive flexibility.",
            "Hard",
            200
        ),
        create_question(
            "Q14: For building complex model configurations (architecture + optimizer + scheduler + loss), which pattern is most appropriate?",
            [
                "Factory pattern - single create() method with many parameters",
                "Builder pattern - fluent interface to construct configuration step-by-step: ModelBuilder().set_architecture('resnet50').set_optimizer('adam', lr=0.001).build()",
                "Constructor with default arguments",
                "Config dictionary"
            ],
            1,
            "Senior Explanation: Builder pattern: Separates construction from representation. Class ModelConfigBuilder: def set_architecture(self, arch): self.arch = arch; return self; def set_optimizer(self, opt, **kwargs): self.opt = (opt, kwargs); return self; def build(self): return ModelConfig(self.arch, self.opt, ...). Usage: config = ModelConfigBuilder().set_architecture('resnet50').set_optimizer('adam', lr=0.001).set_scheduler('cosine').build(). Benefits: (1) Fluent interface (readable), (2) Immutable config object (safe), (3) Validation at build time, (4) Optional parameters natural (skip steps). Option A becomes unwieldy (create(arch='resnet50', opt='adam', opt_lr=0.001, ...) - 20+ parameters). Production: PyTorch Lightning Trainer uses builder-like pattern. Trade-off: Builder adds class complexity but improves API usability for complex objects.",
            "Hard",
            220
        ),
        create_question(
            "Q15: Pipeline pattern for feature engineering (imputation → scaling → encoding). How to handle fit/transform paradigm (fit on train, transform on train/val/test)?",
            [
                "Fit and transform in single method",
                "Separate fit() and transform() methods - fit on train data stores statistics, transform applies to any data using stored statistics",
                "Fit on each dataset independently",
                "No fitting needed"
            ],
            1,
            "Senior Explanation: Fit/Transform paradigm: Each pipeline step implements fit(data) and transform(data). Scaler.fit(train_data): Computes mean/std, stores in self.mean, self.std. Scaler.transform(data): Returns (data - self.mean) / self.std. Pipeline: Class Pipeline: def fit(self, data): for step in self.steps: data = step.fit_transform(data); return self; def transform(self, data): for step in self.steps: data = step.transform(data); return data. Usage: pipeline.fit(train_data); train_transformed = pipeline.transform(train_data); val_transformed = pipeline.transform(val_data). Critical: Prevent data leakage (don't fit on validation). Production: Scikit-learn standard. Trade-off: Fit/transform separation essential for correct ML (test data never seen during fit). Stateless transforms (e.g., log) don't need fit.",
            "Hard",
            220
        ),
        create_question(
            "Q16: For GPU-accelerated data pipelines (preprocessing on GPU), how should Pipeline pattern be adapted?",
            [
                "No changes needed",
                "Add device parameter - each transformation handles device placement, pipeline manages data movement (CPU → GPU → CPU)",
                "Separate CPU and GPU pipelines",
                "Always process on CPU, move final batch to GPU"
            ],
            1,
            "Senior Explanation: Device-aware pipeline: Transformations specify device preference. Class GPUTransformation(Transformation): def transform(self, data): return gpu_resize(data.to('cuda')). Pipeline: Manages data movement (minimize CPU↔GPU transfers). Class Pipeline: def process(self, data): device = 'cpu'; for step in self.steps: if step.requires_gpu and device == 'cpu': data = data.to('cuda'); device = 'gpu'; data = step.transform(data); if device == 'gpu': data = data.to('cpu'); return data. Benefits: (1) Batched GPU operations (resize 32 images at once), (2) Explicit device management (avoid silent CPU fallback), (3) Optimized transfer (minimal host-device copies). Production: NVIDIA DALI, Kornia use GPU pipelines. Trade-off: GPU preprocessing faster (10-50×) but limited by VRAM (smaller batch sizes than CPU).",
            "Hard",
            220
        ),

        # SINGLETON & ABSTRACT CLASSES (Questions 17-20)
        create_question(
            "Q17: For a model registry (global mapping of model names to classes), which pattern ensures single source of truth?",
            [
                "Global dictionary variable",
                "Singleton pattern - single ModelRegistry instance across application",
                "Module-level registry",
                "Class variables"
            ],
            1,
            "Senior Explanation: Singleton: Ensures exactly one instance. Class ModelRegistry: _instance = None; def __new__(cls): if cls._instance is None: cls._instance = super().__new__(cls); cls._instance.models = {}; return cls._instance; def register(self, name, model_class): self.models[name] = model_class. Usage: registry = ModelRegistry(); registry.register('resnet50', ResNet50); later: registry = ModelRegistry(); model_cls = registry.models['resnet50']. Benefits: (1) Global access (import once, use anywhere), (2) Lazy initialization (created when needed), (3) Thread-safe with locks. Option A (global dict) works but less structured. Production: Hugging Face transformers uses module-level registry (similar to singleton). Trade-off: Singletons can make testing harder (global state) but appropriate for registries, config managers. Alternative: Dependency injection (pass registry instance).",
            "Hard",
            200
        ),
        create_question(
            "Q18: When should you use Abstract Base Classes (ABC) in ML code?",
            [
                "Always - forces proper OOP",
                "When defining interfaces for multiple implementations (Dataset, Model, Metric) - ensures all implementations provide required methods",
                "Never - too restrictive for ML research",
                "Only in production, not research code"
            ],
            1,
            "Senior Explanation: Use ABC when: (1) Multiple implementations of same interface (Dataset: ImageDataset, TextDataset, VideoDataset all need __getitem__, __len__), (2) Contract enforcement (all Models must have forward()), (3) Documentation (ABC shows required methods). Implementation: class Dataset(ABC): @abstractmethod def __getitem__(self, idx): pass; @abstractmethod def __len__(self): pass. Subclass must implement both or TypeError. Benefits: (1) Catch errors early (forget to implement method → error at import, not runtime), (2) IDE autocomplete (knows required methods), (3) Type checking (mypy validates). Don't use: Simple scripts, notebooks, one-off experiments. Production: PyTorch Dataset, nn.Module (not ABC but similar concept) use this. Trade-off: Enforced structure (good for frameworks) vs flexibility (good for research).",
            "Medium",
            180
        ),
        create_question(
            "Q19: For a custom training loop, you want users to override specific methods (e.g., compute_loss) but keep core logic fixed. Which pattern?",
            [
                "Template Method pattern - base class defines algorithm structure (train_step), subclasses override specific steps (compute_loss)",
                "Strategy pattern - pass loss function as parameter",
                "Inheritance with all methods overridable",
                "Composition with loss injected"
            ],
            0,
            "Senior Explanation: Template Method: Base class defines skeleton, subclasses fill in specifics. Class BaseTrainer: def train_step(self, batch): # Template method; outputs = self.forward(batch); loss = self.compute_loss(outputs, batch); self.backward(loss); self.optimizer_step(); @abstractmethod def compute_loss(self, outputs, batch): pass. Subclass: Class CustomTrainer(BaseTrainer): def compute_loss(self, outputs, batch): return custom_loss(outputs, batch['targets']). Benefits: (1) Core logic protected (users can't break train_step flow), (2) Extension points clear (override compute_loss, not full train_step), (3) Code reuse (backward, optimizer_step same across trainers). Production: PyTorch Lightning uses this (users override training_step, validation_step). Trade-off: Less flexible than full override but safer (prevents common mistakes).",
            "Hard",
            220
        ),
        create_question(
            "Q20: Composition vs Inheritance for ML model architectures - which principle is better?",
            [
                "Inheritance - create ResNet18, ResNet34, ResNet50 via inheritance",
                "Composition - build models from reusable components (ResidualBlock, DownsampleBlock) - 'favor composition over inheritance'",
                "Both equally good",
                "Neither - use functional approach"
            ],
            1,
            "Senior Explanation: Composition: Models composed of building blocks. Class ResNet: def __init__(self, layers): self.stem = StemBlock(); self.layer1 = self._make_layer(ResidualBlock, layers[0]); self.layer2 = self._make_layer(ResidualBlock, layers[1]). ResNet50 = ResNet([3,4,6,3]). Inheritance issues: Deep hierarchies (BaseModel → ConvNet → ResNet → ResNet50) hard to maintain. Changes to BaseModel affect all descendants. Composition benefits: (1) Flexibility (mix and match blocks), (2) Testability (test ResidualBlock independently), (3) Reusability (ResidualBlock in ResNet, ResNeXt, DenseNet). Production: Modern architectures (EfficientNet, Vision Transformer) use composition. Trade-off: Composition slightly more boilerplate (explicit block creation) but much more flexible and maintainable. Inheritance OK for simple cases (single level).",
            "Hard",
            220
        ),
    ]

    return questions

if __name__ == "__main__":
    db = QuestionDatabase()
    questions = populate_senior_oop_patterns()
    db.add_bulk_questions("Senior OOP - Design Patterns for ML", questions)
    print(f"✓ Successfully added {len(questions)} senior OOP & Design Patterns questions!")
    print(f"✓ Category: Senior OOP - Design Patterns for ML")
