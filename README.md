# AutoNorm: Lightweight Meta-Learning for Adaptive Normalization in Transformers

Official PyTorch implementation of **AutoNorm**, a meta-learning framework that adaptively selects normalization layers in Transformer architectures via learned gating, supporting both soft and discrete (Gumbel-softmax) norm selection.

---

## 📜 Abstract

Normalization is critical for the stability and generalization of deep models, especially in Transformer architectures. Traditional methods such as LayerNorm or DyT assume fixed, hand-designed normalization strategies. We propose **AutoNorm**, a lightweight meta-learning framework that dynamically selects between multiple normalization schemes (e.g., LayerNorm and DyT) per layer and input using a learnable selector. The selector supports both soft weighting and discrete gating via Gumbel-softmax. AutoNorm is designed to be efficient and transferable, enabling fine-tuning or freezing across domains. Additionally, it supports knowledge distillation, robustness testing under corruption, and minimal overhead profiling. Our experiments on MNIST and CIFAR-10 demonstrate that AutoNorm matches or exceeds fixed-norm baselines while maintaining low FLOPs and latency.

---

## 📘 Introduction

Transformer-based models rely on normalization layers for stabilizing training and accelerating convergence. However, using a fixed normalization scheme like LayerNorm or DyT across all inputs and layers can underperform across domains or tasks.

**Key contributions:**
- A norm selector module trained to blend or choose between norms per layer and sample.
- Option for Gumbel-softmax gating to enable discrete norm selection.
- Transfer learning pipeline with freeze/finetune strategies.
- Evaluation of model robustness against common image corruptions.
- Profiling of FLOPs and latency to assess overhead.

---

## ⚙️ Methodology

### 🔁 Norm Selector

At each normalization point, we insert a learnable selector that computes:

\[
\text{Norm}(x) = \alpha \cdot \text{LayerNorm}(x) + (1 - \alpha) \cdot \text{DyT}(x)
\]

Where \(\alpha \in [0, 1]\) is a selector weight per sample and layer.

#### Soft Selection (Default)

\[
\alpha = \sigma(w^T h + b)
\]

where \(h\) is a hidden state from the norm context and \(\sigma\) is the sigmoid activation.

#### Discrete Gating (Optional)

We use the Gumbel-Softmax trick for hard selection:

\[
\alpha = \text{GumbelSoftmax}(\mathbf{logits}, \tau, \text{hard=True})
\]

with temperature \(\tau\) annealed during training. This allows end-to-end differentiable sampling of discrete norm paths.

---

### 🔁 Transfer Learning & Knowledge Distillation

We use a two-phase training strategy:
1. **Pretraining:** AutoNorm is trained on MNIST with cross-entropy loss.
2. **Transfer:** The pretrained model is transferred to CIFAR-10 using either:
   - **Freeze:** The selector and early layers are frozen.
   - **Finetune:** All weights are updated.

Optionally, a **TeacherTransformer** is used to provide soft targets via knowledge distillation.

---

## 🔎 Robustness Evaluation

We evaluate the transferability and robustness of AutoNorm on corrupted CIFAR-10 images:

- **Noise:** Additive Gaussian noise
- **Rotation:** Random affine rotation by ±30°

If corrupted loaders fail to initialize, the result defaults to `None`.

---

## 🧪 Experimental Setup

### Datasets
- **MNIST**: Pretraining dataset
- **CIFAR-10**: Transfer + robustness evaluation

### Baselines
| Model              | Description                              |
|--------------------|------------------------------------------|
| FrozenDyT          | Uses only DyT, disables LayerNorm        |
| FrozenLN           | Uses only LayerNorm, disables DyT        |
| OnlyDyT            | Freezes LayerNorm (auto selector used)   |
| OnlyLayerNorm      | Freezes DyT (auto selector used)         |
| Teacher            | Fully trained high-capacity transformer  |
| AutoNorm           | Ours, soft or Gumbel-based selector      |

---

## 📊 Metrics

Each method is evaluated based on:
- **Validation Accuracy**
- **Inference Latency (ms/sample)**
- **FLOPs (per sample)**
- **Robustness to Noise**
- **Robustness to Rotation**

---

## 📈 Results (Sample)

| Method             | Val Acc | Latency (ms) | FLOPs   | Noise Acc | Rotation Acc |
|--------------------|---------|--------------|---------|-----------|----------------|
| AutoNorm-Freeze    | 81.2%   | 1.23         | 5.2M    | 73.1%     | 68.4%          |
| AutoNorm-Finetune  | 86.1%   | 1.28         | 5.2M    | 79.4%     | 75.4%          |
| FrozenDyT          | 77.4%   | 1.21         | 4.9M    | 69.5%     | 66.7%          |
| FrozenLN           | 74.5%   | 1.19         | 4.8M    | 68.0%     | 62.3%          |
| OnlyDyT            | 75.7%   | 1.21         | 5.0M    | 70.2%     | 63.8%          |
| OnlyLayerNorm      | 76.3%   | 1.20         | 5.0M    | 69.9%     | 64.4%          |
| Teacher            | 88.1%   | 1.55         | 8.1M    | 81.4%     | 78.6%          |

---

## 🧰 Usage Instructions

### 💾 Install Dependencies

```bash
pip install torch torchvision matplotlib ptflops tabulate
```

### Run Pipeline
```bash
python main.py
```
This runs:

- MNIST pretraining

- CIFAR-10 transfer (freeze & finetune)

- Robustness evaluation

- FLOPs & latency profiling

- Attention and gradient visualizations

---

## Output
All logs and plots are saved under the `logs/` folder:
```python-repl
logs/
├── summary_results.csv
├── MNIST_Pretrain_accuracy_loss.png
├── AutoNorm_finetune_grad_heatmap.png
├── norm_weights_AutoNorm_finetune.pt
...
```

---

## Limitations
- Gumbel temperature is fixed; adaptive scheduling could improve results.

- AutoML norm exploration is basic (hardcoded ablations only).

- Selector overhead is not rigorously ablated (basic profiling only).