from model import TransformerWithAutoNorm, FrozenDyTTransformer, FrozenLNTransformer, TeacherTransformer

def make_autonorm():
    return TransformerWithAutoNorm(input_dim=3072)

def make_autonorm_disabled():
    return TransformerWithAutoNorm(input_dim=3072, disable_selector=True)

def make_autonorm_random():
    return TransformerWithAutoNorm(input_dim=3072, random_selector=True)

def make_frozen_dyt():
    return FrozenDyTTransformer(input_dim=3072)

def make_frozen_ln():
    return FrozenLNTransformer(input_dim=3072)

def make_teacher():
    return TeacherTransformer(input_dim=3072)

def make_only_dyt():
    model = TransformerWithAutoNorm(input_dim=3072, disable_selector=False, random_selector=False)
    for p in model.ln.parameters():
        p.requires_grad = False
    return model

def make_only_ln():
    model = TransformerWithAutoNorm(input_dim=3072, disable_selector=True, random_selector=False)
    for p in model.dyt.parameters():
        p.requires_grad = False
    return model

model_variants = {
    "AutoNorm": make_autonorm,
    "AutoNorm_DisableSelector": make_autonorm_disabled,
    "AutoNorm_RandomSelector": make_autonorm_random,
    "FrozenDyT": make_frozen_dyt,
    "FrozenLN": make_frozen_ln,
    "Teacher": make_teacher
}