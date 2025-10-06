## II. Context (AutoNorm Project Overview)

### A. Project Goal and Scope
The primary objective of the AutoNorm project is to dynamically select or combine Dynamic Transformer (DyT) and Layer Normalization (LN) to enhance the performance and robustness of transformer models. This research focuses specifically on classification and regression tasks.

### B. Modular Architecture and Key Components
The AutoNorm project is structured with a modular architecture, comprising several key components:
*   [`configs.py`](configs.py): Manages centralized hyperparameters and configuration settings for experiments.
*   [`data.py`](data.py): Handles data loading, preprocessing, and augmentation for various datasets used in the project.
*   [`model.py`](model.py): Defines the core `TransformerWithAutoNorm` architecture, integrates the `NormSelector`, and includes essential utility modules such as `DropPath`, `PatchEmbedding`, `DyT`, and `SE`. It also contains ablation models like `FrozenDyTTransformer` and `FrozenLNTransformer` for comparative analysis.
*   [`norm_selector.py`](norm_selector.py): Implements the novel adaptive component, the `NormSelector`, which intelligently combines the outputs of DyT and LN. This module supports `disable_selector` and `random_selector` modes for experimental control.
*   [`factory.py`](factory.py): Provides a factory pattern for streamlined instantiation of different model configurations.
*   [`train.py`](train.py): Contains the implementation of the training loop, incorporating advanced techniques such as cosine learning rate scheduling, MixUp, CutMix, label smoothing, DropPath, Exponential Moving Average (EMA), early stopping, and knowledge distillation.
*   [`utils.py`](utils.py): A collection of helper functions for various tasks, including evaluation metrics (accuracy, RMSE, MAE), plotting, performance profiling (FLOPs, latency), checkpoint management, and robustness analysis.
*   [`main.py`](main.py): Orchestrates the entire experimental pipeline, including pretraining and progressive finetuning stages.

### C. Novel Components and Techniques
The AutoNorm project introduces and leverages several novel components and advanced techniques:
*   **NormSelector**: A key innovation that dynamically weights the outputs of $DyT(x)$ and $LN(x)$ using a Multi-Layer Perceptron (MLP) and a softmax function, allowing for adaptive normalization.
*   **Dynamic Transformer (DyT)**: A simple yet effective learnable scaling mechanism defined as $DyT(x) = x \odot \alpha$, where $\alpha$ is a learnable parameter.
*   **Progressive Finetuning**: A two-phase transfer learning strategy involving an initial head-only finetuning followed by a full finetune of the entire model.
*   **Extensive Data Augmentation**: Employs techniques such as MixUp, CutMix, and CIFAR-specific augmentations to improve model generalization.
*   **Regularization**: Incorporates `DropPath` (stochastic depth) and label smoothing to prevent overfitting.
*   **Model Averaging**: Utilizes Exponential Moving Average (EMA) for model weights to enhance generalization capabilities.
*   **Test-Time Augmentation (TTA)**: Applied to classification tasks to achieve more robust predictions.
*   **Knowledge Distillation**: A technique where student models learn from a larger, more capable teacher model to improve performance.