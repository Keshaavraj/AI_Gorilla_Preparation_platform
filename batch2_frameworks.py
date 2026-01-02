"""
Batch 2: Framework Questions
- TensorFlow (15 questions)
- PyTorch (15 questions)
- Scikit-learn (15 questions)
"""

from database_manager import QuestionDatabase, create_question

def populate_tensorflow():
    """15 TensorFlow Questions"""
    questions = [
        create_question(
            "In TensorFlow 2.x, what is the primary advantage of using tf.function decorator?",
            [
                "It makes code run on GPU only",
                "It converts Python functions to optimized computation graphs for better performance",
                "It enables distributed training automatically",
                "It removes the need for data preprocessing"
            ],
            1,
            "The @tf.function decorator uses AutoGraph to convert Python code into TensorFlow computation graphs, enabling optimizations like operation fusion, constant folding, and better GPU utilization. This provides significant speedups while maintaining Python's ease of use. Without it, TensorFlow runs in eager mode (convenient but slower).",
            "Medium",
            90
        ),
        create_question(
            "You're building a custom training loop in TensorFlow. What is the correct way to compute and apply gradients?",
            [
                "gradients = tape.gradient(loss, model.weights); optimizer.apply_gradients(zip(gradients, model.trainable_variables))",
                "optimizer.minimize(loss)",
                "model.fit(x, y)",
                "gradients = model.compute_gradients(loss)"
            ],
            0,
            "In custom training loops, you use tf.GradientTape to record operations, compute gradients with tape.gradient(loss, trainable_vars), then apply them with optimizer.apply_gradients(zip(grads, vars)). This gives full control over the training process. model.fit() is the high-level API that does this automatically. Note: must use model.trainable_variables, not model.weights.",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of tf.data.Dataset.prefetch() in TensorFlow?",
            [
                "To download data from the internet",
                "To load next batches while the model trains on current batch, reducing training time",
                "To increase batch size automatically",
                "To cache all data in GPU memory"
            ],
            1,
            "prefetch() overlaps data preprocessing and model execution. While the model trains on batch N, the input pipeline prepares batch N+1 in parallel (on CPU), reducing idle GPU time. Usage: dataset.prefetch(tf.data.AUTOTUNE) automatically tunes the buffer size. This is crucial for efficient GPU utilization, especially with complex preprocessing.",
            "Medium",
            85
        ),
        create_question(
            "In TensorFlow, what is the difference between model.save() and model.save_weights()?",
            [
                "They are identical",
                "save() saves the full model (architecture + weights + optimizer state); save_weights() saves only weights",
                "save_weights() is deprecated",
                "save() only works with Sequential models"
            ],
            1,
            "model.save() (SavedModel format) saves the complete model: architecture, weights, training config, and optimizer state. You can reload and continue training or deploy directly. model.save_weights() saves only the weight values, requiring you to recreate the architecture separately. Use save() for deployment, save_weights() for checkpointing during training.",
            "Medium",
            80
        ),
        create_question(
            "You need to fine-tune a pre-trained MobileNet model. How do you freeze the base layers in TensorFlow?",
            [
                "Delete the base layers",
                "Set base_model.trainable = False before compiling",
                "Use a smaller learning rate only",
                "Remove optimizer"
            ],
            1,
            "Setting layer.trainable = False freezes those layers' weights during training. For transfer learning: base = MobileNet(weights='imagenet'); base.trainable = False; then add new layers and compile. You can later unfreeze and fine-tune with a lower learning rate. This must be set BEFORE compile() to take effect.",
            "Medium",
            85
        ),
        create_question(
            "What does tf.GradientTape(persistent=True) do?",
            [
                "Makes the tape last forever",
                "Allows computing multiple gradients from the same tape (normally consumed after first gradient() call)",
                "Saves gradients to disk",
                "Prevents gradient computation"
            ],
            1,
            "By default, GradientTape is consumed after one gradient() call. persistent=True allows multiple gradient computations from the same tape, useful for computing gradients with respect to different variables or multiple losses. Remember to manually del tape when done to free resources. Most common in custom training loops with multiple optimizers.",
            "Hard",
            90
        ),
        create_question(
            "In TensorFlow Keras, what is the purpose of the validation_split parameter in model.fit()?",
            [
                "To split the model into multiple parts",
                "To automatically reserve a portion of training data for validation during training",
                "To enable cross-validation",
                "To reduce training data size"
            ],
            1,
            "validation_split=0.2 automatically takes the last 20% of training data for validation (without shuffling that portion). This is convenient but less flexible than validation_data parameter. For better control, manually split data and use validation_data=(X_val, y_val). Note: if data is shuffled before fit(), the split is taken after shuffling.",
            "Medium",
            75
        ),
        create_question(
            "What is the purpose of tf.keras.layers.BatchNormalization()?",
            [
                "To normalize the batch size",
                "To normalize layer inputs across the batch dimension, stabilizing and accelerating training",
                "To create larger batches",
                "To remove outliers from batches"
            ],
            1,
            "BatchNormalization normalizes inputs to each layer across the batch dimension (mean=0, std=1), then applies learned scale and shift parameters. This reduces internal covariate shift, allows higher learning rates, and provides regularization. Important: it behaves differently in training (uses batch statistics) vs inference (uses moving averages), controlled by the training parameter.",
            "Medium",
            85
        ),
        create_question(
            "In TensorFlow, what does model.compile(run_eagerly=True) do?",
            [
                "Makes the model run faster",
                "Disables graph compilation, running operations eagerly for easier debugging",
                "Enables distributed training",
                "Compiles the model for production"
            ],
            1,
            "run_eagerly=True disables tf.function graph compilation, executing operations immediately like NumPy. This enables easier debugging (can use print, pdb, etc.) but is much slower. Use it for debugging, then remove for production training. By default, run_eagerly=False uses compiled graphs for performance.",
            "Medium",
            80
        ),
        create_question(
            "What is the purpose of tf.keras.callbacks.ModelCheckpoint?",
            [
                "To debug the model",
                "To save model/weights at intervals, typically saving the best model based on validation metrics",
                "To stop training early",
                "To visualize training"
            ],
            1,
            "ModelCheckpoint saves the model at specified intervals. Common pattern: save_best_only=True with monitor='val_loss' saves only when validation loss improves. This prevents losing the best model if training continues past optimal point. Use with save_weights_only=True for faster checkpointing or False for full model saving.",
            "Medium",
            80
        ),
        create_question(
            "In TensorFlow, what is the difference between Keras Sequential and Functional API?",
            [
                "They are identical",
                "Sequential is for linear stacks; Functional API supports complex architectures with multiple inputs/outputs",
                "Functional API is deprecated",
                "Sequential is faster"
            ],
            1,
            "Sequential is simple for linear layer stacks: model.add(layer1); model.add(layer2). Functional API uses: x = Input(); x = layer1(x); x = layer2(x); output = layer3(x); model = Model(inputs, output). Functional API supports branching, multiple inputs/outputs, shared layers, and residual connections - essential for complex architectures like ResNet, Inception.",
            "Medium",
            85
        ),
        create_question(
            "What does tf.keras.layers.Dropout(0.5) do during inference by default?",
            [
                "Drops 50% of neurons",
                "Does nothing - dropout is automatically disabled during inference",
                "Always drops neurons regardless of mode",
                "Increases neuron count"
            ],
            1,
            "Dropout automatically turns off during inference (training=False). During training, it randomly drops neurons and scales remaining activations by 1/(1-rate) to maintain expected values. During inference, all neurons are active without scaling. This behavior is controlled by the training argument in call(). Never manually apply dropout during inference.",
            "Medium",
            75
        ),
        create_question(
            "In TensorFlow, what is the purpose of tf.data.Dataset.cache()?",
            [
                "To delete old data",
                "To cache preprocessed data in memory or disk, avoiding redundant preprocessing across epochs",
                "To compress the dataset",
                "To download data faster"
            ],
            1,
            "cache() stores preprocessed data after first epoch. Subsequent epochs read from cache instead of reprocessing. Use cache() for small datasets fitting in memory, or cache(filename) for disk caching of larger datasets. Place it AFTER expensive preprocessing but BEFORE augmentation (which should vary per epoch). Huge speedup when preprocessing is expensive.",
            "Hard",
            90
        ),
        create_question(
            "What is tf.keras.mixed_precision used for?",
            [
                "Mixing different models",
                "Using both float16 and float32 for faster training with minimal accuracy loss",
                "Training multiple tasks simultaneously",
                "Combining different optimizers"
            ],
            1,
            "Mixed precision uses float16 for most operations (faster, less memory) and float32 for numerical stability where needed. Usage: tf.keras.mixed_precision.set_global_policy('mixed_float16'). This can provide 2-3x speedup on modern GPUs with Tensor Cores. Loss scaling prevents underflow in gradients. Essential for training large models efficiently.",
            "Hard",
            95
        ),
        create_question(
            "In TensorFlow, what is the purpose of tf.keras.layers.GlobalAveragePooling2D?",
            [
                "To increase feature map size",
                "To reduce each feature map to a single value by averaging, reducing parameters and preventing overfitting",
                "To normalize across channels",
                "To apply convolution"
            ],
            1,
            "GlobalAveragePooling2D reduces each feature map (H×W) to a single value by averaging all spatial locations, outputting one value per channel. For input (batch, H, W, C), output is (batch, C). This drastically reduces parameters compared to flattening + dense layers, prevents overfitting, and maintains spatial invariance. Common as the final pooling in modern CNNs before classification.",
            "Medium",
            85
        )
    ]
    return questions


def populate_pytorch():
    """15 PyTorch Questions"""
    questions = [
        create_question(
            "In PyTorch, what is the purpose of loss.backward()?",
            [
                "To move the loss backward in time",
                "To compute gradients of loss with respect to all tensors with requires_grad=True",
                "To reverse the model architecture",
                "To undo the forward pass"
            ],
            1,
            "loss.backward() performs backpropagation, computing gradients using the computational graph. It populates the .grad attribute of all tensors that have requires_grad=True. These gradients are accumulated (added to existing .grad), so you must call optimizer.zero_grad() before each backward pass to clear old gradients. This is the core of PyTorch's autograd system.",
            "Medium",
            85
        ),
        create_question(
            "What is the difference between model.eval() and torch.no_grad() in PyTorch?",
            [
                "They are identical",
                "eval() changes layer behavior (dropout/batchnorm); no_grad() disables gradient computation",
                "eval() is for evaluation, no_grad() is for training",
                "no_grad() is deprecated"
            ],
            1,
            "model.eval() switches layers like Dropout and BatchNorm to inference mode (dropout off, batchnorm uses running stats). torch.no_grad() disables gradient computation to save memory and speed up inference. For proper inference, use BOTH: model.eval(); with torch.no_grad(): predictions = model(x). eval() affects behavior, no_grad() affects computation.",
            "Hard",
            95
        ),
        create_question(
            "In PyTorch, what does optimizer.step() do?",
            [
                "Moves forward one training step",
                "Updates model parameters using computed gradients",
                "Computes gradients",
                "Increases learning rate"
            ],
            1,
            "optimizer.step() updates model parameters based on their .grad attributes using the optimizer's update rule (SGD, Adam, etc.). Standard training loop: optimizer.zero_grad() → loss.backward() → optimizer.step(). step() applies the optimization algorithm (e.g., w = w - lr * grad for SGD). It doesn't compute gradients (that's backward()) or clear them (that's zero_grad()).",
            "Medium",
            80
        ),
        create_question(
            "What is the purpose of DataLoader in PyTorch?",
            [
                "To download datasets from the internet",
                "To batch, shuffle, and parallelize data loading during training",
                "To preprocess images only",
                "To store model weights"
            ],
            1,
            "DataLoader wraps a Dataset and provides: batching, shuffling, parallel loading (num_workers), and memory pinning (pin_memory=True for GPU). Usage: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4). It returns an iterator yielding batches. Essential for efficient training, especially with large datasets and preprocessing.",
            "Medium",
            80
        ),
        create_question(
            "In PyTorch, when should you use tensor.detach()?",
            [
                "To delete the tensor",
                "To create a tensor that shares data but doesn't track gradients, breaking the computational graph",
                "To move tensor to CPU",
                "To free GPU memory"
            ],
            1,
            "detach() creates a view of the tensor that shares storage but doesn't require gradients and isn't part of the computational graph. Use cases: (1) when you want to use a value without backpropagating through it, (2) implementing stop-gradient operations, (3) extracting values for logging without affecting training. Changes to detached tensor affect original storage.",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of torch.nn.Module.register_buffer()?",
            [
                "To create trainable parameters",
                "To register non-trainable tensors that should be saved with model state",
                "To create temporary variables",
                "To increase buffer size"
            ],
            1,
            "register_buffer() registers a tensor as part of the module state (saved/loaded with state_dict) but NOT as a trainable parameter. Use for running statistics (BatchNorm), constant tensors, or any state that should persist but not be trained. Example: self.register_buffer('running_mean', torch.zeros(num_features)). These are moved with .to(device) but excluded from parameters().",
            "Hard",
            95
        ),
        create_question(
            "In PyTorch, what does nn.CrossEntropyLoss expect as input?",
            [
                "Softmax probabilities and one-hot labels",
                "Raw logits (unnormalized scores) and class indices",
                "Binary values only",
                "Probabilities between 0 and 1"
            ],
            1,
            "CrossEntropyLoss expects: (1) raw logits (unnormalized model outputs) - NOT softmax probabilities, and (2) class indices as targets - NOT one-hot vectors. It internally applies log_softmax then negative log-likelihood. Common mistake: applying softmax before loss leads to incorrect gradients. For binary classification, use BCEWithLogitsLoss (also expects logits).",
            "Medium",
            90
        ),
        create_question(
            "What is the purpose of torch.nn.Sequential in PyTorch?",
            [
                "To process sequences like RNNs",
                "To define a linear stack of layers as a single module",
                "To enable parallel processing",
                "To create recursive networks"
            ],
            1,
            "Sequential creates a container that chains layers in order: model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)). Input flows through layers sequentially. It's convenient for simple architectures but limited - no branching, skip connections, or complex logic. For complex architectures, inherit from nn.Module and define custom forward().",
            "Medium",
            75
        ),
        create_question(
            "In PyTorch, what does requires_grad=True do?",
            [
                "Requires the tensor to be on GPU",
                "Tells PyTorch to track operations for automatic differentiation",
                "Makes the tensor immutable",
                "Enables distributed training"
            ],
            1,
            "requires_grad=True tells PyTorch to build a computational graph for this tensor, enabling gradient computation via autograd. Model parameters have this by default. For inputs, usually False. Setting it tracks all operations, allowing backward() to compute gradients. Impacts: (1) memory overhead for graph, (2) computational overhead for tracking. Use with torch.no_grad() to disable when not needed.",
            "Medium",
            85
        ),
        create_question(
            "What is the correct way to move a model and data to GPU in PyTorch?",
            [
                "model.gpu(); data.gpu()",
                "model.to('cuda'); data = data.to('cuda')",
                "model.cuda() only",
                "Automatic, no code needed"
            ],
            1,
            "Use device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device); data = data.to(device). Note: .to() is preferred over .cuda() (more flexible), and for data you must reassign (data = data.to()) as it returns a new tensor. Model and data must be on the same device. For multi-GPU, use DataParallel or DistributedDataParallel.",
            "Medium",
            80
        ),
        create_question(
            "In PyTorch, what is the purpose of torch.utils.data.Dataset?",
            [
                "To automatically download data",
                "An abstract class defining interface for datasets: __len__() and __getitem__()",
                "A pre-built dataset of images",
                "A data augmentation tool"
            ],
            1,
            "Dataset is an abstract class you inherit to create custom datasets. Must implement: __len__() returning dataset size, and __getitem__(idx) returning one sample. DataLoader then uses these to fetch batches. Example: class MyDataset(Dataset): def __getitem__(self, idx): return self.data[idx], self.labels[idx]. Separates data loading logic from training loop.",
            "Medium",
            85
        ),
        create_question(
            "What does nn.Linear(in_features, out_features) implement in PyTorch?",
            [
                "A nonlinear transformation",
                "An affine transformation: y = xW^T + b",
                "A convolutional layer",
                "A dropout layer"
            ],
            1,
            "nn.Linear implements a fully-connected (dense) layer: y = xW^T + b, where W is a weight matrix (out_features × in_features) and b is a bias vector (out_features). It's a linear/affine transformation. For input (batch, in_features), output is (batch, out_features). Learnable parameters are accessed via layer.weight and layer.bias. Nonlinearity must be added separately.",
            "Medium",
            75
        ),
        create_question(
            "In PyTorch, what is the purpose of torch.optim.lr_scheduler?",
            [
                "To increase batch size over time",
                "To adjust learning rate during training according to a schedule",
                "To schedule when training starts",
                "To control GPU memory"
            ],
            1,
            "Learning rate schedulers adjust the LR during training. Common schedulers: StepLR (decay by gamma every N epochs), ReduceLROnPlateau (reduce when metric plateaus), CosineAnnealingLR (cosine decay), OneCycleLR (cyclical for super-convergence). Usage: scheduler = StepLR(optimizer, step_size=30, gamma=0.1); call scheduler.step() each epoch. Proper LR scheduling significantly improves convergence and final performance.",
            "Hard",
            90
        ),
        create_question(
            "What is the purpose of model.parameters() in PyTorch?",
            [
                "To set model hyperparameters",
                "To return an iterator over all trainable parameters (weights and biases)",
                "To print model architecture",
                "To count parameters"
            ],
            1,
            "model.parameters() returns an iterator over all learnable parameters. Used primarily when creating optimizers: optimizer = Adam(model.parameters(), lr=0.001). For parameter count: sum(p.numel() for p in model.parameters()). To separate parameters (e.g., for different learning rates), use model.named_parameters() or access specific layers.",
            "Medium",
            75
        ),
        create_question(
            "In PyTorch, what does tensor.view() do?",
            [
                "Visualizes the tensor",
                "Reshapes the tensor to a new shape without copying data (must be contiguous)",
                "Creates a copy of the tensor",
                "Prints tensor values"
            ],
            1,
            "view() returns a reshaped tensor sharing the same underlying data (no copy). Requirements: tensor must be contiguous, new shape must have same number of elements. Use -1 for one dimension to be inferred: x.view(batch_size, -1) flattens all but first dimension. Alternative: reshape() (works on non-contiguous too, may copy). Common use: flatten before fully-connected layers.",
            "Medium",
            80
        )
    ]
    return questions


def populate_scikit_learn():
    """15 Scikit-learn Questions"""
    questions = [
        create_question(
            "In scikit-learn, what is the purpose of fit_transform() vs. fit() followed by transform()?",
            [
                "They are identical",
                "fit_transform() is more efficient for the same object; fit() + transform() allows using fitted transformer on new data",
                "fit_transform() only works on training data",
                "transform() is deprecated"
            ],
            1,
            "On training data, fit_transform() is convenient and sometimes optimized. However, the key pattern is: fit on training data (scaler.fit(X_train)), then transform both train and test (X_train_scaled = scaler.transform(X_train); X_test_scaled = scaler.transform(X_test)). Never fit on test data - this causes data leakage. fit_transform() is shorthand for fit().transform() on the same data.",
            "Hard",
            95
        ),
        create_question(
            "What does StandardScaler do in scikit-learn?",
            [
                "Scales features to [0, 1] range",
                "Standardizes features to mean=0 and std=1 by removing mean and scaling to unit variance",
                "Normalizes each sample to unit norm",
                "Applies logarithmic scaling"
            ],
            1,
            "StandardScaler standardizes features: z = (x - μ) / σ, where μ is the mean and σ is the standard deviation computed from training data. This makes features have mean=0 and variance=1. Use when algorithms assume normally distributed features or are sensitive to scale (SVM, KNN, PCA). MinMaxScaler scales to [0,1], Normalizer scales samples (not features) to unit norm.",
            "Medium",
            85
        ),
        create_question(
            "In scikit-learn's Pipeline, what is the main advantage of using it?",
            [
                "It makes code longer",
                "It chains transformers and estimator, preventing data leakage and enabling easy cross-validation",
                "It only works with neural networks",
                "It speeds up training by 10x"
            ],
            1,
            "Pipeline chains transformers and a final estimator, ensuring: (1) fit is called only on training folds in CV, preventing leakage, (2) same preprocessing applies to train/test automatically, (3) hyperparameter tuning includes preprocessing, (4) cleaner code. Example: Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('clf', SVC())]). Can be used in GridSearchCV like any estimator.",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of cross_val_score() in scikit-learn?",
            [
                "To validate only once",
                "To perform k-fold cross-validation and return scores for each fold",
                "To split data into train/test",
                "To compute only training accuracy"
            ],
            1,
            "cross_val_score(estimator, X, y, cv=5) performs k-fold CV: splits data into k folds, trains on k-1 folds, evaluates on remaining fold, repeats k times, returns k scores. This gives robust performance estimate. It handles the fitting and splitting automatically. For more control (e.g., getting predictions), use cross_val_predict() or cross_validate().",
            "Medium",
            85
        ),
        create_question(
            "In scikit-learn, what does GridSearchCV do?",
            [
                "Creates a grid of data points",
                "Exhaustively searches over specified hyperparameter values to find the best combination via cross-validation",
                "Visualizes model performance",
                "Searches for grid patterns in data"
            ],
            1,
            "GridSearchCV exhaustively tests all combinations of specified hyperparameters using cross-validation. Usage: GridSearchCV(estimator, param_grid, cv=5).fit(X, y). Access best params via .best_params_, best score via .best_score_, best model via .best_estimator_. Can be slow with many parameters. Alternative: RandomizedSearchCV samples parameter combinations, faster for large search spaces.",
            "Medium",
            90
        ),
        create_question(
            "What is the purpose of train_test_split's stratify parameter?",
            [
                "To split data randomly",
                "To ensure class distribution is preserved in both train and test sets",
                "To create more data",
                "To remove outliers"
            ],
            1,
            "stratify=y ensures the proportion of each class is the same in train and test sets as in the original data. Crucial for imbalanced datasets to ensure test set is representative. Example: if 30% positive class, both train and test will have ~30% positive. Without stratification, random split might create unrepresentative splits by chance, especially with small datasets.",
            "Medium",
            80
        ),
        create_question(
            "In scikit-learn's RandomForestClassifier, what does n_estimators represent?",
            [
                "The number of features",
                "The number of trees in the forest",
                "The maximum depth of trees",
                "The number of samples"
            ],
            1,
            "n_estimators is the number of decision trees to train. More trees generally improve performance and stability but increase computation. Typical values: 100-500. Unlike neural networks, random forests don't overfit with more trees (though they may overfit with very deep trees). Trees are trained in parallel, making it efficient. Performance plateaus after enough trees.",
            "Medium",
            75
        ),
        create_question(
            "What does the feature_importances_ attribute provide in tree-based models?",
            [
                "The feature names",
                "Importance scores indicating how much each feature contributes to predictions",
                "The correlation between features",
                "The number of features"
            ],
            1,
            "feature_importances_ gives importance scores (sum to 1.0) based on how much each feature decreases impurity (Gini or entropy) across all trees. Higher values mean more important. Use for feature selection and interpretation. Limitations: biased toward high-cardinality features, can't detect feature interactions well. Available for DecisionTree, RandomForest, GradientBoosting models. Access via model.feature_importances_.",
            "Medium",
            85
        ),
        create_question(
            "In scikit-learn, what is the difference between predict() and predict_proba()?",
            [
                "They are identical",
                "predict() returns class labels; predict_proba() returns probability estimates for each class",
                "predict_proba() is faster",
                "predict() works only for binary classification"
            ],
            1,
            "predict() returns predicted class labels (0, 1, or multi-class). predict_proba() returns probability estimates for each class (shape: n_samples × n_classes). For binary classification, column 0 is P(class=0), column 1 is P(class=1). Use predict_proba() when you need confidence scores, want to set custom thresholds, or need probabilities for calibration. Not all estimators support predict_proba().",
            "Medium",
            80
        ),
        create_question(
            "What is the purpose of OneHotEncoder in scikit-learn?",
            [
                "To encode continuous variables",
                "To convert categorical variables into binary vectors (one-hot encoding)",
                "To normalize features",
                "To reduce dimensionality"
            ],
            1,
            "OneHotEncoder converts categorical features to binary (0/1) vectors. For a feature with k categories, creates k binary columns. Example: color=['red','blue','red'] → [[1,0],[0,1],[1,0]] for red/blue. Essential for algorithms requiring numerical input (linear models, neural nets). Trees can handle categorical directly. Alternative: LabelEncoder (ordinal encoding, implies ordering).",
            "Medium",
            80
        ),
        create_question(
            "In scikit-learn, what does SVC(kernel='rbf', C=1.0, gamma='scale') mean?",
            [
                "Linear SVM with C=1",
                "RBF kernel SVM; C controls regularization, gamma controls kernel width",
                "Polynomial kernel SVM",
                "Only binary classification"
            ],
            1,
            "RBF (Radial Basis Function) kernel for non-linear classification. C is regularization: higher C means less regularization (fit training data closely, risk overfitting). gamma defines kernel width: higher gamma means more complex decision boundary (narrow influence). gamma='scale' uses 1/(n_features * X.var()). Common pattern: grid search over C and gamma. RBF is most popular kernel for non-linear SVMs.",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of fit() in a scikit-learn estimator?",
            [
                "To make the model smaller",
                "To train/learn model parameters from training data",
                "To evaluate model performance",
                "To transform data"
            ],
            1,
            "fit(X, y) is the training method: it learns model parameters from training data. For classifiers/regressors, it learns decision boundaries/functions. For transformers (scalers, PCA), it learns transformation parameters (mean/std, principal components). After fitting, model is ready for predict(). fit() modifies the estimator's internal state. Calling fit() again retrains from scratch.",
            "Medium",
            75
        ),
        create_question(
            "In scikit-learn's PCA, what does n_components=0.95 mean?",
            [
                "Select 95 components",
                "Select enough components to explain 95% of variance",
                "Remove 95% of features",
                "Use 95% of the data"
            ],
            1,
            "When n_components is a float between 0 and 1, PCA selects the minimum number of components that explain that fraction of variance. n_components=0.95 selects components explaining 95% of variance. This is data-driven dimensionality reduction. Alternative: specify exact number (n_components=10) or 'mle' for automatic selection. Access explained variance via pca.explained_variance_ratio_.",
            "Medium",
            85
        ),
        create_question(
            "What does make_pipeline() do differently from Pipeline()?",
            [
                "They are completely different",
                "make_pipeline() auto-generates step names from class names, convenient shorthand for Pipeline()",
                "make_pipeline() is faster",
                "Pipeline() is deprecated"
            ],
            1,
            "make_pipeline() is a convenience function that creates a Pipeline with auto-generated step names. make_pipeline(StandardScaler(), PCA(), SVC()) is equivalent to Pipeline([('standardscaler', StandardScaler()), ('pca', PCA()), ('svc', SVC())]). Use make_pipeline() for quick prototyping, Pipeline() when you need specific names (e.g., for GridSearchCV parameter naming).",
            "Medium",
            80
        ),
        create_question(
            "In scikit-learn, what is the purpose of ColumnTransformer?",
            [
                "To add new columns",
                "To apply different transformers to different columns/subsets of features",
                "To remove columns",
                "To rename columns"
            ],
            1,
            "ColumnTransformer applies different preprocessing to different features, essential for heterogeneous data. Example: ColumnTransformer([('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(), categorical_features)]). This scales numeric features and encodes categorical ones in one step. Works seamlessly in Pipeline. Alternative: manually transform and concatenate, but error-prone and doesn't prevent leakage in CV.",
            "Hard",
            95
        )
    ]
    return questions


if __name__ == "__main__":
    db = QuestionDatabase()

    print("Populating TensorFlow questions...")
    db.add_bulk_questions("TensorFlow", populate_tensorflow())
    print(f"✓ Added {len(populate_tensorflow())} TensorFlow questions")

    print("Populating PyTorch questions...")
    db.add_bulk_questions("PyTorch", populate_pytorch())
    print(f"✓ Added {len(populate_pytorch())} PyTorch questions")

    print("Populating Scikit-learn questions...")
    db.add_bulk_questions("Scikit-learn", populate_scikit_learn())
    print(f"✓ Added {len(populate_scikit_learn())} Scikit-learn questions")

    stats = db.get_statistics()
    print(f"\n{'='*60}")
    print(f"BATCH 2 COMPLETE - Frameworks")
    print(f"{'='*60}")
    print(f"Total questions in database: {stats['total_questions']}")
    print("\nBatch 2 questions by category:")
    for category in ["TensorFlow", "PyTorch", "Scikit-learn"]:
        count = db.get_question_count(category)
        print(f"  {category}: {count} questions")
    print(f"\nDatabase saved to: questions_db.json")
