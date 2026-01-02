"""
Batch 1: Core AI/ML Questions
- Machine Learning (20 questions)
- Deep Learning (20 questions)
- Artificial Intelligence (20 questions)
- NLP (20 questions)
- Generative AI (20 questions)
"""

from database_manager import QuestionDatabase, create_question

def populate_machine_learning():
    """20 Machine Learning Questions"""
    questions = [
        create_question(
            "You're building a credit risk model and notice that your training accuracy is 98% but test accuracy is 72%. What is the most likely issue?",
            [
                "The model is underfitting the data",
                "The model is overfitting the training data",
                "The test dataset is too small",
                "The features are not normalized"
            ],
            1,
            "Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor generalization to new data. The large gap between training (98%) and test (72%) accuracy is a classic sign of overfitting. Common solutions include regularization, reducing model complexity, or increasing training data.",
            "Medium",
            90
        ),
        create_question(
            "In a binary classification problem with 95% negative samples and 5% positive samples, which metric would be MOST misleading if used alone?",
            [
                "F1-score",
                "Precision",
                "Accuracy",
                "ROC-AUC"
            ],
            2,
            "Accuracy can be highly misleading in imbalanced datasets. A model that always predicts the majority class (negative) would achieve 95% accuracy without learning anything useful. F1-score, precision, recall, and ROC-AUC are better suited for imbalanced problems as they consider both classes' performance.",
            "Medium",
            85
        ),
        create_question(
            "You apply L2 regularization to your linear regression model. What effect does increasing the lambda (λ) parameter have?",
            [
                "Increases model complexity and overfitting",
                "Decreases bias and increases variance",
                "Pushes feature weights closer to zero",
                "Automatically performs feature selection by setting weights to exactly zero"
            ],
            2,
            "L2 regularization (Ridge) adds a penalty term proportional to the square of weights. Increasing λ penalizes large weights more heavily, pushing them closer to (but not exactly) zero. This reduces model complexity and helps prevent overfitting. L1 regularization (Lasso) is what sets weights to exactly zero, performing feature selection.",
            "Hard",
            100
        ),
        create_question(
            "In k-fold cross-validation with k=5, what percentage of data is used for training in each fold?",
            [
                "20%",
                "50%",
                "80%",
                "100%"
            ],
            2,
            "In 5-fold cross-validation, the data is split into 5 equal parts. In each iteration, 4 parts (80%) are used for training and 1 part (20%) for validation. This process repeats 5 times, with each fold serving as the validation set once.",
            "Medium",
            70
        ),
        create_question(
            "A data scientist notices their Random Forest model performs worse than a single Decision Tree. What is the most likely cause?",
            [
                "Random Forests always perform worse than Decision Trees",
                "The trees in the forest are too correlated due to similar features being selected",
                "The number of trees is too high",
                "Random Forests can't handle categorical variables"
            ],
            1,
            "Random Forests work by averaging predictions from multiple decorrelated trees. If the trees are highly correlated (e.g., due to limited feature diversity, too few features sampled per split, or highly imbalanced data), the ensemble loses its advantage. The strength of Random Forest comes from the diversity of its constituent trees.",
            "Hard",
            95
        ),
        create_question(
            "In the context of supervised learning, which statement about the bias-variance tradeoff is correct?",
            [
                "High bias and high variance both lead to overfitting",
                "Increasing model complexity always reduces both bias and variance",
                "Low bias and high variance typically indicate overfitting",
                "Regularization primarily reduces bias"
            ],
            2,
            "Low bias means the model fits the training data well, while high variance means predictions vary significantly with different training sets. This combination is characteristic of overfitting - the model captures noise in training data but doesn't generalize well. Regularization typically reduces variance, not bias.",
            "Hard",
            100
        ),
        create_question(
            "You're using Gradient Boosting and notice training is very slow. Which hyperparameter would speed up training MOST without significantly hurting performance?",
            [
                "Increase the learning rate significantly",
                "Reduce the maximum depth of trees",
                "Increase the number of estimators",
                "Remove all regularization"
            ],
            1,
            "Reducing max depth makes trees shallower and faster to build. Gradient Boosting builds trees sequentially, so faster individual trees significantly speed up training. While increasing learning rate can reduce iterations needed, setting it too high can hurt convergence. Increasing estimators or removing regularization would slow training or hurt performance.",
            "Medium",
            90
        ),
        create_question(
            "In a multi-class classification problem with 10 classes, what is the output shape of a one-hot encoded label vector for a single sample?",
            [
                "(1,)",
                "(10,)",
                "(1, 10)",
                "(10, 1)"
            ],
            1,
            "One-hot encoding creates a binary vector with length equal to the number of classes. For 10 classes, the vector has shape (10,) with all zeros except a 1 at the index corresponding to the class. For example, class 3 would be [0,0,0,1,0,0,0,0,0,0].",
            "Medium",
            75
        ),
        create_question(
            "You're using K-Nearest Neighbors (KNN) for classification. The features have very different scales (e.g., age: 0-100, income: 0-1000000). What preprocessing step is MOST important?",
            [
                "One-hot encoding",
                "Feature normalization/standardization",
                "PCA dimensionality reduction",
                "No preprocessing needed"
            ],
            1,
            "KNN uses distance metrics to find nearest neighbors. Without normalization, features with larger scales (like income) will dominate the distance calculation, making smaller-scale features (like age) nearly irrelevant. Normalization (e.g., StandardScaler, MinMaxScaler) ensures all features contribute equally to distance calculations.",
            "Medium",
            85
        ),
        create_question(
            "In the context of decision trees, what does 'pruning' accomplish?",
            [
                "Removes features that are not important",
                "Reduces overfitting by removing branches that provide little predictive power",
                "Increases the maximum depth of the tree",
                "Balances the dataset by removing samples"
            ],
            1,
            "Pruning removes sections of the tree that provide little power to classify instances, reducing complexity and preventing overfitting. Pre-pruning stops tree growth early using criteria like max depth, while post-pruning removes branches after the tree is fully grown. This is different from feature selection.",
            "Medium",
            80
        ),
        create_question(
            "You're building a model to predict housing prices. Which algorithm would be LEAST appropriate for this regression task?",
            [
                "Linear Regression",
                "Random Forest Regressor",
                "Logistic Regression",
                "Gradient Boosting Regressor"
            ],
            2,
            "Logistic Regression is specifically designed for classification tasks (predicting categories), not regression (predicting continuous values). Despite its name, it outputs probabilities for class membership. For housing price prediction, you need regression algorithms like Linear Regression, Random Forest Regressor, or Gradient Boosting Regressor.",
            "Medium",
            75
        ),
        create_question(
            "In ensemble learning, what is the primary difference between bagging and boosting?",
            [
                "Bagging trains models sequentially, boosting trains in parallel",
                "Bagging reduces variance, boosting primarily reduces bias",
                "Bagging can only use decision trees, boosting can use any model",
                "Bagging is supervised, boosting is unsupervised"
            ],
            1,
            "Bagging (Bootstrap Aggregating) trains models independently in parallel on random subsets of data, reducing variance by averaging diverse models (e.g., Random Forest). Boosting trains models sequentially, where each model focuses on correcting errors of previous ones, primarily reducing bias (e.g., AdaBoost, Gradient Boosting).",
            "Hard",
            95
        ),
        create_question(
            "You're using Silhouette Score to evaluate clustering results. What does a score close to +1 indicate?",
            [
                "Clusters are poorly separated and overlap significantly",
                "Samples are well-matched to their cluster and far from other clusters",
                "The number of clusters is incorrect",
                "The clustering algorithm failed"
            ],
            1,
            "Silhouette Score ranges from -1 to +1. A score close to +1 means samples are well-matched to their own cluster (cohesive) and poorly-matched to neighboring clusters (separated). Scores near 0 indicate overlapping clusters, and negative scores suggest samples are assigned to wrong clusters.",
            "Medium",
            85
        ),
        create_question(
            "In Principal Component Analysis (PCA), the first principal component is defined as:",
            [
                "The axis with minimum variance in the data",
                "The axis that best separates the classes",
                "The axis with maximum variance in the data",
                "The original feature with highest correlation to the target"
            ],
            2,
            "PCA finds orthogonal axes that capture maximum variance in the data. The first principal component is the direction along which the data varies the most. Subsequent components capture remaining variance in orthogonal directions. PCA is unsupervised and doesn't use class labels or target variables.",
            "Medium",
            90
        ),
        create_question(
            "You're comparing two models: Model A has precision=0.8 and recall=0.6, Model B has precision=0.6 and recall=0.8. Which statement is correct?",
            [
                "Model A has fewer false positives relative to true positives",
                "Model B is better at identifying all positive cases",
                "Both models have the same F1-score",
                "All of the above"
            ],
            3,
            "Model A's higher precision (0.8) means fewer false positives relative to true positives. Model B's higher recall (0.8) means it identifies more of the actual positive cases. Both have F1-score = 2*(0.8*0.6)/(0.8+0.6) = 0.686. The choice between them depends on whether false positives or false negatives are more costly in your application.",
            "Hard",
            100
        ),
        create_question(
            "In K-Means clustering, what happens during the 'assignment step'?",
            [
                "New centroids are calculated",
                "Each point is assigned to the nearest centroid",
                "The number of clusters k is determined",
                "Outliers are removed from the dataset"
            ],
            1,
            "K-Means alternates between two steps: (1) Assignment step - each data point is assigned to its nearest centroid based on Euclidean distance, and (2) Update step - centroids are recalculated as the mean of all points assigned to that cluster. This process repeats until convergence.",
            "Medium",
            80
        ),
        create_question(
            "You notice your SVM model is not performing well on non-linearly separable data. What should you do?",
            [
                "Decrease the regularization parameter C",
                "Use a kernel function like RBF or polynomial",
                "Increase the number of support vectors",
                "Switch to a linear kernel"
            ],
            1,
            "Kernel functions transform data into higher-dimensional space where it may become linearly separable. The RBF (Radial Basis Function) and polynomial kernels are popular for non-linear problems. The linear kernel only works for linearly separable data. The number of support vectors is determined by the algorithm, not a parameter you set.",
            "Medium",
            90
        ),
        create_question(
            "In the context of feature engineering, what is 'feature interaction'?",
            [
                "Removing correlated features",
                "Creating new features by combining existing ones (e.g., multiplication)",
                "Normalizing features to the same scale",
                "Selecting the most important features"
            ],
            1,
            "Feature interaction (or feature crossing) creates new features by combining existing ones to capture relationships that aren't apparent in individual features. For example, combining 'hour' and 'day_of_week' might reveal that traffic is high on 'Friday evenings'. Common operations include multiplication, division, or logical combinations.",
            "Medium",
            85
        ),
        create_question(
            "What is the primary advantage of using stratified sampling in train-test split?",
            [
                "It increases the total amount of data available",
                "It ensures class distribution is similar in train and test sets",
                "It automatically handles imbalanced datasets",
                "It reduces training time"
            ],
            1,
            "Stratified sampling ensures that the proportion of samples for each class is approximately the same in both training and test sets. This is especially important for imbalanced datasets, ensuring the test set is representative and evaluation metrics are reliable. Regular random sampling might create unrepresentative splits by chance.",
            "Medium",
            80
        ),
        create_question(
            "You're tuning hyperparameters using Grid Search. With 3 hyperparameters having 4, 3, and 5 possible values respectively, and 5-fold cross-validation, how many total model fits are performed?",
            [
                "60",
                "300",
                "12",
                "15"
            ],
            1,
            "Grid Search tests all combinations: 4 × 3 × 5 = 60 combinations. With 5-fold cross-validation, each combination is evaluated 5 times. Total fits = 60 × 5 = 300. This demonstrates why Grid Search can be computationally expensive, especially with many hyperparameters or large datasets. RandomizedSearchCV is often more efficient.",
            "Hard",
            95
        )
    ]
    return questions


def populate_deep_learning():
    """20 Deep Learning Questions"""
    questions = [
        create_question(
            "During backpropagation in a deep neural network, you observe that gradients in early layers are extremely small. What problem are you facing?",
            [
                "Exploding gradients",
                "Vanishing gradients",
                "Overfitting",
                "Learning rate is too high"
            ],
            1,
            "Vanishing gradients occur when gradients become extremely small as they propagate backward through many layers, especially with activation functions like sigmoid or tanh. This prevents early layers from learning effectively. Solutions include using ReLU activations, batch normalization, residual connections, or different architectures like LSTM/GRU for sequences.",
            "Medium",
            90
        ),
        create_question(
            "In a CNN for image classification, what is the primary purpose of pooling layers?",
            [
                "Increase the spatial dimensions of feature maps",
                "Reduce spatial dimensions and computational cost while maintaining important features",
                "Add non-linearity to the network",
                "Normalize activations"
            ],
            1,
            "Pooling layers (e.g., Max Pooling, Average Pooling) downsample feature maps by reducing their spatial dimensions (width and height). This reduces computational cost, helps prevent overfitting, and provides translation invariance by retaining important features while discarding spatial details. Convolutional layers provide feature learning, activation functions add non-linearity.",
            "Medium",
            85
        ),
        create_question(
            "You're training a deep network and loss suddenly becomes NaN. What is the MOST likely cause?",
            [
                "Learning rate is too low",
                "Learning rate is too high causing exploding gradients",
                "Batch size is too small",
                "The model is underfitting"
            ],
            1,
            "NaN (Not a Number) typically results from numerical instability, most commonly from exploding gradients caused by too high a learning rate. When gradients become very large, weight updates can cause activations or losses to exceed floating-point limits. Solutions include reducing learning rate, gradient clipping, or proper weight initialization.",
            "Medium",
            85
        ),
        create_question(
            "In batch normalization, normalization is applied:",
            [
                "Only to the input layer",
                "To activations within mini-batches during training",
                "Only during the testing phase",
                "To the final output layer only"
            ],
            1,
            "Batch normalization normalizes activations for each mini-batch during training by subtracting the batch mean and dividing by batch standard deviation. This reduces internal covariate shift, allows higher learning rates, and acts as regularization. During inference, it uses running statistics computed during training rather than batch statistics.",
            "Medium",
            90
        ),
        create_question(
            "What is the purpose of dropout in neural networks?",
            [
                "To speed up training",
                "To reduce overfitting by randomly deactivating neurons during training",
                "To increase model capacity",
                "To normalize gradients"
            ],
            1,
            "Dropout randomly sets a fraction of neuron outputs to zero during training, forcing the network to learn redundant representations and preventing co-adaptation of neurons. This acts as powerful regularization to reduce overfitting. During inference, dropout is turned off and weights are scaled to account for the missing activations during training.",
            "Medium",
            80
        ),
        create_question(
            "In transfer learning for image classification, which approach is typically BEST when you have very limited training data?",
            [
                "Train the entire pre-trained model from scratch",
                "Freeze all layers and only train a new classifier head",
                "Fine-tune all layers with a high learning rate",
                "Don't use a pre-trained model at all"
            ],
            1,
            "With very limited data, freezing the pre-trained layers (feature extractor) and only training the new classifier head prevents overfitting while leveraging learned features. As you get more data, you can fine-tune deeper layers with a lower learning rate. Training from scratch requires large datasets to learn good representations.",
            "Hard",
            95
        ),
        create_question(
            "What is the key innovation of Residual Networks (ResNet)?",
            [
                "Using very small 1x1 convolutions",
                "Skip connections that add input to output of layers",
                "Replacing pooling with strided convolutions",
                "Using batch normalization"
            ],
            1,
            "ResNet introduces skip connections (or residual connections) that add the input of a layer block to its output. This creates an identity mapping that allows gradients to flow directly through the network, solving the vanishing gradient problem and enabling training of very deep networks (100+ layers). The network learns residual functions F(x) instead of the full mapping H(x).",
            "Hard",
            100
        ),
        create_question(
            "In a neural network, the ReLU activation function outputs:",
            [
                "Values between 0 and 1",
                "Values between -1 and 1",
                "max(0, x) - zero for negative inputs, the value itself for positive inputs",
                "The sigmoid of the input"
            ],
            2,
            "ReLU (Rectified Linear Unit) is defined as f(x) = max(0, x). It outputs 0 for all negative inputs and the input value itself for positive inputs. ReLU is popular because it's computationally efficient, helps mitigate vanishing gradients (unlike sigmoid/tanh), and often leads to faster convergence. However, it can suffer from 'dying ReLU' where neurons output zero for all inputs.",
            "Medium",
            75
        ),
        create_question(
            "You're building a sequence-to-sequence model for machine translation. Which architecture is MOST appropriate?",
            [
                "Convolutional Neural Network (CNN)",
                "Vanilla Feedforward Neural Network",
                "Encoder-Decoder architecture with attention",
                "Single LSTM layer"
            ],
            2,
            "Sequence-to-sequence tasks (variable-length input to variable-length output) are best handled by encoder-decoder architectures. The encoder processes the input sequence into a context representation, and the decoder generates the output sequence. Attention mechanisms allow the decoder to focus on relevant parts of the input, dramatically improving translation quality. CNNs are for spatial data, feedforward networks can't handle variable sequences.",
            "Hard",
            100
        ),
        create_question(
            "What is the purpose of padding in convolutional layers?",
            [
                "To add more parameters to the model",
                "To maintain spatial dimensions of feature maps",
                "To increase training time",
                "To reduce overfitting"
            ],
            1,
            "Padding adds extra pixels (usually zeros) around the input border. This allows the convolutional filter to be applied to edge pixels and prevents the feature map from shrinking. 'SAME' padding maintains the input size, while 'VALID' padding (no padding) reduces it. Padding also ensures corner/edge features are processed as thoroughly as central features.",
            "Medium",
            80
        ),
        create_question(
            "In the context of deep learning, what does 'epoch' refer to?",
            [
                "One forward pass through a single batch",
                "One complete pass through the entire training dataset",
                "The number of layers in the network",
                "The learning rate schedule"
            ],
            1,
            "An epoch represents one complete pass through the entire training dataset. If your dataset has 1000 samples and batch size is 100, one epoch consists of 10 batches (iterations). Training typically involves multiple epochs, allowing the model to see each sample many times. The number of epochs is a hyperparameter that balances training time and convergence.",
            "Medium",
            70
        ),
        create_question(
            "Which optimization algorithm adapts the learning rate for each parameter individually?",
            [
                "Standard Gradient Descent",
                "Stochastic Gradient Descent (SGD)",
                "Adam (Adaptive Moment Estimation)",
                "Momentum"
            ],
            2,
            "Adam combines ideas from RMSprop and momentum, maintaining adaptive learning rates for each parameter based on estimates of first and second moments of gradients. This makes it effective across a wide range of problems. Standard SGD and momentum use the same learning rate for all parameters. Adam is often a good default choice, though SGD with momentum can sometimes achieve better generalization.",
            "Medium",
            85
        ),
        create_question(
            "In a CNN, what is a '3x3 filter with stride 2' doing?",
            [
                "Looking at a 3x3 region and moving 2 pixels at a time, downsampling the output",
                "Looking at a 2x2 region and moving 3 pixels",
                "Creating 3 output channels",
                "Using 2 layers of 3x3 convolutions"
            ],
            0,
            "The filter size (3x3) defines the receptive field - the region of input examined at each position. Stride determines how many pixels the filter moves between applications. Stride 2 means the filter moves 2 pixels at a time, resulting in an output roughly half the input size (downsampling). Stride 1 (default) maintains size (with appropriate padding), while larger strides reduce spatial dimensions.",
            "Medium",
            90
        ),
        create_question(
            "What problem does Batch Normalization primarily address?",
            [
                "Overfitting",
                "Internal covariate shift (changing distribution of layer inputs during training)",
                "Vanishing outputs",
                "Computational efficiency"
            ],
            1,
            "Batch Normalization addresses internal covariate shift - the change in distribution of network activations during training as parameters update. By normalizing layer inputs, it stabilizes learning, allows higher learning rates, reduces sensitivity to initialization, and provides some regularization. This leads to faster convergence and better performance.",
            "Hard",
            95
        ),
        create_question(
            "You're training an image classifier and want to prevent overfitting. Which techniques would help? (Assume multiple good answers, pick the BEST combination)",
            [
                "Increase model size and remove regularization",
                "Data augmentation, dropout, and L2 regularization",
                "Increase learning rate and remove batch normalization",
                "Use smaller batch sizes only"
            ],
            1,
            "Preventing overfitting requires regularization techniques. Data augmentation increases effective dataset size by creating variations (rotations, flips, crops). Dropout randomly drops neurons during training. L2 regularization penalizes large weights. These can be combined effectively. Increasing model size or learning rate would worsen overfitting. Batch size affects training dynamics but isn't primarily a regularization technique.",
            "Hard",
            100
        ),
        create_question(
            "In LSTM (Long Short-Term Memory) networks, what is the primary purpose of the forget gate?",
            [
                "To completely erase the memory cell",
                "To control what information to discard from the cell state",
                "To add new information to the cell",
                "To produce the final output"
            ],
            1,
            "The forget gate in LSTM decides what information to discard from the cell state by outputting values between 0 (forget completely) and 1 (keep completely) for each element in the cell state. This allows the network to learn what past information is relevant to keep and what to discard, enabling effective learning of long-term dependencies.",
            "Medium",
            90
        ),
        create_question(
            "What is 'gradient clipping' used for?",
            [
                "To prevent vanishing gradients",
                "To prevent exploding gradients by limiting maximum gradient magnitude",
                "To increase training speed",
                "To normalize layer outputs"
            ],
            1,
            "Gradient clipping prevents exploding gradients by capping the maximum magnitude of gradients during backpropagation. If the gradient norm exceeds a threshold, it's scaled down. This is especially important in RNNs where gradients can grow exponentially. Common methods include clipping by value (element-wise) or by norm (scaling the entire gradient vector).",
            "Medium",
            85
        ),
        create_question(
            "In a convolutional layer, if the input is 32x32x3 (height x width x channels) and you apply 64 filters of size 5x5, what is the number of channels in the output (assuming valid padding)?",
            [
                "3",
                "5",
                "64",
                "192"
            ],
            2,
            "The number of output channels equals the number of filters applied. Each of the 64 filters produces one feature map (channel), resulting in 64 output channels. The spatial dimensions would be 28x28 (32-5+1 with valid padding), giving a final output shape of 28x28x64. Input channels (3) are combined by each filter to produce one output channel.",
            "Medium",
            85
        ),
        create_question(
            "What is the main advantage of using pre-trained word embeddings (like Word2Vec or GloVe) in NLP tasks?",
            [
                "They eliminate the need for any training",
                "They capture semantic relationships learned from large corpora",
                "They work only for English",
                "They guarantee perfect accuracy"
            ],
            1,
            "Pre-trained embeddings like Word2Vec and GloVe are trained on massive text corpora (billions of words) to learn dense vector representations that capture semantic and syntactic relationships. Words with similar meanings have similar vectors (e.g., 'king' and 'queen'). Using these as initialization or features gives your model a head start, especially valuable with limited training data.",
            "Medium",
            90
        ),
        create_question(
            "You have a dataset of 1000 samples. Which batch size would make your training most similar to standard Gradient Descent?",
            [
                "Batch size = 1",
                "Batch size = 32",
                "Batch size = 100",
                "Batch size = 1000"
            ],
            3,
            "Standard (Batch) Gradient Descent computes gradients using the entire dataset before updating weights. Using batch size = dataset size (1000) achieves this. Smaller batches give Stochastic Gradient Descent (SGD) or Mini-batch GD. Batch size = 1 is pure SGD (one sample per update). Mini-batch (e.g., 32, 64) balances computational efficiency, convergence speed, and generalization.",
            "Medium",
            80
        )
    ]
    return questions


# This is part 1 - Due to length, I'll continue with AI, NLP, and Generative AI in the next section
if __name__ == "__main__":
    db = QuestionDatabase()

    print("Populating Machine Learning questions...")
    db.add_bulk_questions("Machine Learning", populate_machine_learning())
    print(f"✓ Added {len(populate_machine_learning())} Machine Learning questions")

    print("Populating Deep Learning questions...")
    db.add_bulk_questions("Deep Learning", populate_deep_learning())
    print(f"✓ Added {len(populate_deep_learning())} Deep Learning questions")

    print(f"\nTotal questions in database: {db.get_question_count()}")
    print("Database saved to questions_db.json")
