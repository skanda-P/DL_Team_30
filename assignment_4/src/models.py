# PyTorch model definitions for FCNN classifiers, 1-hidden-layer and 3-hidden-layer Autoencoders, Denoising Autoencoders, and ARCHITECTURES registry.
"""
Models for Assignment 4.

Currently contains the FCNN classifier used by Tasks 1, 3, 4 and 5. The nine
architectures are the same ones used in Assignment 3 ("use the same
architecture as Task-1" in Tasks 3/4/5 refers to this shared list).
Autoencoder models for Tasks 2-6 are added below the classifier section.
"""

import torch
import torch.nn as nn

# ------------------------------------------------------------------ classifier

ARCHITECTURES = {
    "arch_3l_v1": [512, 256, 128],
    "arch_3l_v2": [256, 128, 64],
    "arch_3l_v3": [256, 256, 256],

    "arch_4l_v1": [512, 256, 128, 64],
    "arch_4l_v2": [256, 128, 64, 32],
    "arch_4l_v3": [384, 256, 128, 64],

    "arch_5l_v1": [512, 384, 256, 128, 64],
    "arch_5l_v2": [256, 256, 128, 128, 64],
    "arch_5l_v3": [256, 128, 64, 64, 32],
}

ARCH_GROUPS = {
    "3_layers": ["arch_3l_v1", "arch_3l_v2", "arch_3l_v3"],
    "4_layers": ["arch_4l_v1", "arch_4l_v2", "arch_4l_v3"],
    "5_layers": ["arch_5l_v1", "arch_5l_v2", "arch_5l_v3"],
    "all": list(ARCHITECTURES.keys()),
}

ACTIVATION_CHOICES = ["tanh", "logistic"]


class FCNN(nn.Module):
    """Fully connected classifier: input -> [Linear + activation] x L -> Linear(num_classes)."""

    def __init__(self, input_dim, hidden_dims, num_classes=5, activation="tanh"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes

        act = activation.lower()
        if act == "tanh":
            act_fn = nn.Tanh
        elif act in ("logistic", "sigmoid"):
            act_fn = nn.Sigmoid
        else:
            raise ValueError(f"Unsupported activation '{activation}'. Allowed: {ACTIVATION_CHOICES}")

        layers, in_features = [], input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(in_features, h), act_fn()]
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))   # logits; softmax is inside the CE loss
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def initialize_weights(model, seed=42):
    """Xavier-normal weights, zero biases (same scheme as Assignment 3)."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_classifier(arch_name, input_dim, num_classes=5, activation="tanh", seed=42):
    """
    Builds an FCNN for `arch_name` with deterministic initial weights. The same
    (arch, input_dim, seed) always gives the same starting point.
    """
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch_name}'. Available: {list(ARCHITECTURES)}")
    model = FCNN(input_dim, ARCHITECTURES[arch_name], num_classes, activation)
    initialize_weights(model, seed=seed)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for d in (32, 64, 128, 256, 784):
        print(f"input_dim={d}")
        for a in ARCHITECTURES:
            m = build_classifier(a, d)
            print(f"  {a:11s} {ARCHITECTURES[a]!s:28s} params: {count_parameters(m):>9,d}")