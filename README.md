# Assessment
## Ex 1
The model uses several important architectural choices beyond the backbone:

Pooling Strategy: I chose to use the [CLS] token embedding followed by a learned pooling layer (Linear + Tanh) rather than simple mean pooling. This allows the model to learn a more sophisticated sentence representation.

Embedding Dimension: Maintained the BERT base dimension (768) through the pooling layer to preserve information capacity.

Dropout Usage: Implemented dropout in task-specific heads but not in the pooling layer to maintain consistent sentence embeddings.

## Ex 2

The architecture supports two tasks:

Task A: Sentence Classification (3 classes)

Task B: Sentiment Analysis (2 classes)

Key architectural changes for multi-task support:

Shared Backbone: Both tasks share the transformer and pooling layers to learn general-purpose sentence representations.

Task-Specific Heads: Each task has a dedicated head with:

Dropout for regularization

Two-layer architecture with ReLU activation

Dimension reduction (768 → 384 → num_classes) to prevent overfitting

Flexible Forward Pass: The model accepts a task parameter to optionally compute only specific task outputs.

## Ex 3

Entire Network Frozen:

Advantages: Fast training, no catastrophic forgetting

Disadvantages: No adaptation

Best for: tasks are very similar to pre-trained objectives

Frozen Transformer Backbone:

Advantages: Preserves generality while allowing task-specific adaptability

Disadvantages: Difficult to adapt to domain-specific patterns

Best for: Less data cases or when compute resources are not available

One Task Head Frozen:

Advantages: Allows newer task to gain from last task's knowledge while avoiding degradation

Disadvantages: Can miss required synergy between tasks

Best for: Cases where one task is optimized and I want to add another tast without causing interference

Transfer Learning Approach:

My Pre-trained Model Choice: BERT-base-uncased as:

It is great for English text

Has some balance between performance and efficiency

Enough pre-training on a large variety of text

Layer Freezing Strategy:

Freeze first 6 layers (it contains basic language features)

Fine-tune last 6 layers + pooling + task heads

Rationale: The lower layers capture more general language features, while upper layers adapt to specific tasks

## Ex 4

Layer-wise Learning Rates

The implementation includes a LayerwiseLearningRateOptimizer that sets different learning rates for:

Backbone: 1e-5 (lower to preserve knowledge)

Pooling: 2e-5 (medium, push adaptability)

Task Heads: 5e-5 (large, push extreme specific task learning)

Benefits in multi-task setting:

Stops catastrophic forgetting in lower layers

Helps task-specific layers to adapt fast

Balances preservation with task optimization
