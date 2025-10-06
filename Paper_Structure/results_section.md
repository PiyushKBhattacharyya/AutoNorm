# V. Results

This section presents the experimental results for both classification and regression tasks, evaluating the performance, robustness, and efficiency of the `TransformerWithAutoNorm` variants against baseline models.

## A. Classification Tasks (MNIST, CIFAR10, FashionMNIST, SVHN)

### 1. Performance Comparison

Across all classification benchmarks (MNIST, CIFAR10, FashionMNIST, SVHN), the `TransformerWithAutoNorm` variants consistently demonstrated superior performance compared to the `Teacher` model. While `AutoNorm` proved to be competitive, its performance was not consistently superior to fixed normalization strategies.

Specific observations for each dataset include:
*   **MNIST and FashionMNIST:** `AutoNorm_DisableSelector` and `FrozenLN` variants occasionally exhibited slightly higher accuracy.
*   **CIFAR10:** The `AutoNorm_RandomSelector` variant achieved the highest accuracy on this dataset.
*   **SVHN:** `FrozenDyT` emerged as the best-performing model for the SVHN dataset.

### 2. Robustness Analysis

A robustness analysis was conducted on the CIFAR10 dataset to assess model stability under various perturbations:
*   **Noise Robustness:** Transformer models generally showed lower robustness to noise compared to the `Teacher` model.
*   **Rotation Robustness:** Conversely, Transformer models demonstrated improved robustness to rotational transformations.

### 3. Efficiency Analysis

An evaluation of computational efficiency revealed the following:
*   **`TransformerWithAutoNorm` Variants:** FLOPs (Floating Point Operations) and latency metrics were largely similar across all `TransformerWithAutoNorm` variants.
*   **`Teacher` Model:** The `Teacher` model was found to be significantly more efficient in terms of both FLOPs and latency when compared to the Transformer-based models.

## B. Regression Tasks (EnergyEfficiency)

### 1. Performance Comparison

For the EnergyEfficiency regression task, the `TransformerWithAutoNorm` variants exhibited comparable performance:
*   All `TransformerWithAutoNorm` variants performed similarly in terms of RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
*   Crucially, all `TransformerWithAutoNorm` variants consistently outperformed the `Teacher` model.
*   Fixed normalization strategies, specifically `FrozenDyT` and `AutoNorm_DisableSelector`, sometimes yielded marginally better RMSE values than the dynamic `AutoNorm` approach.