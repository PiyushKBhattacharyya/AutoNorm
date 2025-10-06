# AutoNorm: Adaptive Normalization for Transformer Models

## Introduction
The `AutoNorm` project introduces and evaluates a novel adaptive normalization mechanism, the `NormSelector`, designed for transformer models. Its primary goal is to dynamically select or combine Dynamic Transformer (DyT) and Layer Normalization (LN) to enhance model performance and robustness across a variety of classification and regression tasks. This approach aims to overcome the limitations of fixed normalization strategies by adapting to input features.

## Key Features and Highlights
*   **NormSelector**: A core innovation that adaptively blends the outputs of DyT and Layer Normalization. It uses a small Multi-Layer Perceptron (MLP) to predict dynamic weights ($w_0, w_1$) such that the output is calculated as $\text{Output} = w_0 \cdot DyT(x) + w_1 \cdot LN(x)$.
*   **Dynamic Transformer (DyT)**: A simple yet effective learnable scaling mechanism, defined as $DyT(x) = x \odot \alpha$, where $\alpha$ is a learnable parameter.
*   **Progressive Finetuning**: An efficient two-phase transfer learning strategy involving initial head-only finetuning followed by full model finetuning on downstream tasks.
*   **Extensive Data Augmentation**: Incorporates advanced techniques such as MixUp, CutMix, and CIFAR-specific augmentations to improve model generalization.
*   **Regularization Techniques**: Utilizes DropPath (stochastic depth) for improved model robustness and label smoothing to prevent overfitting.
*   **Model Averaging**: Employs Exponential Moving Average (EMA) for model weights, leading to more stable and generalized performance.
*   **Test-Time Augmentation (TTA)**: Enhances the robustness of classification predictions by averaging predictions over multiple augmented views of the input.
*   **Knowledge Distillation**: Supports the training of student models by leveraging the knowledge from a larger, more powerful teacher model.

## Project Structure
The `AutoNorm` project is organized into several modular components, each responsible for a specific aspect of the experimental pipeline:

*   [`configs.py`](configs.py): Centralizes all hyperparameters and configuration settings, making experiments reproducible and easy to manage.
*   [`data.py`](data.py): Manages data loading, preprocessing, and augmentation for diverse datasets, including popular benchmarks like MNIST, CIFAR10, and CaliforniaHousing.
*   [`model.py`](model.py): Defines the core `TransformerWithAutoNorm` architecture, which seamlessly integrates the `NormSelector`. It also includes essential utility modules such as `DropPath`, `PatchEmbedding`, `DyT`, and `SE` (Squeeze-and-Excitation blocks), along with specialized models for ablation studies (`FrozenDyTTransformer`, `FrozenLNTransformer`).
*   [`norm_selector.py`](norm_selector.py): Implements the novel `NormSelector` component, detailing how it adaptively combines DyT and LN outputs. It also supports `disable_selector` (LayerNorm only) and `random_selector` modes for comparative analysis.
*   [`factory.py`](factory.py): Provides a flexible factory pattern for instantiating different model variants, simplifying the setup of various experimental configurations.
*   [`train.py`](train.py): Encapsulates the training loop, integrating advanced techniques like cosine learning rate scheduling with warmup, MixUp, CutMix, label smoothing, DropPath, Exponential Moving Average (EMA), early stopping, and knowledge distillation.
*   [`utils.py`](utils.py): Contains a collection of helper functions for various tasks, including evaluation metrics (accuracy, RMSE, MAE), plotting results, performance profiling (FLOPs, latency), checkpoint loading, and generating corrupted test loaders for robustness analysis.
*   [`main.py`](main.py): The orchestrator of the entire experimental pipeline, handling tasks from pretraining on general datasets (e.g., CIFAR100, CaliforniaHousing) to progressive finetuning on downstream tasks.

## Setup and Installation

To set up the `AutoNorm` project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AutoNorm.git
    cd AutoNorm
    ```
    *(Note: Replace `https://github.com/your-username/AutoNorm.git` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Example for CUDA 11.8, adjust as needed
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is assumed to exist with all necessary Python packages. If not, it should be created.)*

## How to Run Experiments

Experiments are orchestrated through the [`main.py`](main.py) script, with configurations managed in [`configs.py`](configs.py).

1.  **Review Configuration:**
    Before running an experiment, review and adjust the hyperparameters and settings in [`configs.py`](configs.py) to suit your needs.

2.  **Execute an Experiment:**
    You can run an experiment using the following command structure:
    ```bash
    python main.py --config_name <name_of_your_config>
    ```
    For example, to run an experiment configured for CIFAR10:
    ```bash
    python main.py --config_name cifar10_autonorm_experiment
    ```
    *(Note: Replace `<name_of_your_config>` with the actual configuration name defined in `configs.py`.)*

## Summary of Results

The `AutoNorm` project conducted extensive evaluations, comparing `AutoNorm` against `AutoNorm_DisableSelector` (LayerNorm only), `AutoNorm_RandomSelector`, `FrozenDyT` (DyT only), `FrozenLN` (LayerNorm only), and a `Teacher` baseline.

### Classification Tasks (MNIST, CIFAR10, FashionMNIST, SVHN)
*   **Performance**: `TransformerWithAutoNorm` variants generally outperform the `Teacher` model. However, `AutoNorm` itself is competitive but does not consistently achieve superior performance compared to fixed normalization strategies (e.g., `FrozenDyT`, `FrozenLN`) across all datasets. The optimal normalization strategy appears to be dataset-dependent.
*   **Robustness**: `AutoNorm` variants showed lower robustness to noise but demonstrated better robustness to rotation compared to the `Teacher` model.
*   **Efficiency**: FLOPs and Latency were similar across `AutoNorm` variants, but generally higher than the simpler `Teacher` model.

### Regression Tasks (EnergyEfficiency)
*   **Performance**: `TransformerWithAutoNorm` variants performed similarly to each other and consistently outperformed the `Teacher` model. In some cases, fixed normalization strategies yielded marginally better RMSE than `AutoNorm`.

### Conclusion from Analysis
The `AutoNorm` project successfully implements and evaluates an adaptive normalization approach. While it demonstrates strong performance compared to a simpler baseline, the adaptive `NormSelector` does not consistently outperform fixed normalization strategies across all tasks and metrics. This suggests that the optimal normalization strategy can be dataset-dependent, and further research into the learning mechanism of `NormSelector` or its application in more complex scenarios might be beneficial.

## Future Work and Contributions
We welcome contributions and suggestions for improving `AutoNorm`. Potential areas for future work include:
*   Investigating alternative architectures or learning mechanisms for the `NormSelector` to enhance its adaptability.
*   Exploring the application of `AutoNorm` in more complex transformer models or different domains.
*   Conducting further ablation studies to understand the individual contributions of each component.
*   Optimizing the computational efficiency of the `NormSelector`.

Feel free to open issues or submit pull requests!