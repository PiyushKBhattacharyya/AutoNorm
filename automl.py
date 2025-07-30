from model import TransformerWithAutoNorm, FrozenDyTTransformer, FrozenLNTransformer
from utils import profile_latency
import torch

def automl_search():
    input_shape = (1, 1, 28, 28)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    candidates = {
        'AutoNorm': TransformerWithAutoNorm(),
        'FrozenDyT': FrozenDyTTransformer(),
        'FrozenLN': FrozenLNTransformer()
    }
    results = {}
    for name, model in candidates.items():
        latency = profile_latency(model, input_shape, device)
        results[name] = latency
    return results