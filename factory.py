from model import TransformerWithAutoNorm, FrozenDyTTransformer, FrozenLNTransformer, BaselineMLPTransformer, FrozenRMSNormTransformer

# Common dataset heads configuration
MODEL_INPUT_DIM = 3072  # for CIFAR10-like data; regression datasets will still work after flatten
HEADS_CONFIG = {
    "MNIST": 10,
    "CIFAR10": 10,
    "FashionMNIST": 10,
    "CaliforniaHousing": 1,
    "Diabetes": 1,
    "BostonHousing": 1
}

def make_autonorm():
    return TransformerWithAutoNorm(input_dim=MODEL_INPUT_DIM)

def make_autonorm_disabled():
    return TransformerWithAutoNorm(input_dim=MODEL_INPUT_DIM, disable_selector=True)

def make_autonorm_random():
    return TransformerWithAutoNorm(input_dim=MODEL_INPUT_DIM, random_selector=True)

def make_frozen_dyt():
    return FrozenDyTTransformer(input_dim=MODEL_INPUT_DIM)

def make_frozen_ln():
    return FrozenLNTransformer(input_dim=MODEL_INPUT_DIM)

def make_baseline_mlp():
    return BaselineMLPTransformer(input_dim=MODEL_INPUT_DIM)

def make_frozen_rms():
    return FrozenRMSNormTransformer(input_dim=MODEL_INPUT_DIM)

def make_only_dyt():
    model = TransformerWithAutoNorm(input_dim=MODEL_INPUT_DIM, disable_selector=False, random_selector=False)
    for p in model.ln.parameters():
        p.requires_grad = False
    return model

def make_only_ln():
    model = TransformerWithAutoNorm(input_dim=MODEL_INPUT_DIM, disable_selector=True, random_selector=False)
    for p in model.dyt.parameters():
        p.requires_grad = False
    return model

model_variants = {
    "AutoNorm": lambda **kwargs: TransformerWithAutoNorm(is_cifar=kwargs.get('is_cifar', True)),
    "AutoNorm_DisableSelector": lambda **kwargs: TransformerWithAutoNorm(is_cifar=kwargs.get('is_cifar', True), disable_selector=True),
    "AutoNorm_RandomSelector": lambda **kwargs: TransformerWithAutoNorm(is_cifar=kwargs.get('is_cifar', True), random_selector=True),
    "FrozenDyT": lambda **kwargs: FrozenDyTTransformer(is_cifar=kwargs.get('is_cifar', True)),
    "FrozenLN": lambda **kwargs: FrozenLNTransformer(is_cifar=kwargs.get('is_cifar', True)),
    "FrozenRMS": lambda **kwargs: TransformerWithAutoNorm(is_cifar=kwargs.get('is_cifar', True), use_rms_only=True),
    "AdaNorm": lambda **kwargs: TransformerWithAutoNorm(is_cifar=kwargs.get('is_cifar', True), force_normalization='adanorm'),
    "FiLM": lambda **kwargs: TransformerWithAutoNorm(is_cifar=kwargs.get('is_cifar', True), force_normalization='film'),
}
