import os
import copy
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Experimental architectures: 3 distinct variants for each depth (3, 4, and 5 hidden layers)
ARCHITECTURES = {
    # --- 3 Hidden Layers Variants ---
    "arch_3l_v1": [512, 256, 128],            # Wide funnel
    "arch_3l_v2": [256, 128, 64],             # Compact funnel
    "arch_3l_v3": [256, 256, 256],            # Uniform width

    # --- 4 Hidden Layers Variants ---
    "arch_4l_v1": [512, 256, 128, 64],        # Wide gradual funnel
    "arch_4l_v2": [256, 128, 64, 32],         # Compact funnel
    "arch_4l_v3": [384, 256, 128, 64],        # Mid-wide funnel

    # --- 5 Hidden Layers Variants ---
    "arch_5l_v1": [512, 384, 256, 128, 64],   # Deep wide gradual funnel
    "arch_5l_v2": [256, 256, 128, 128, 64],   # Deep stepped funnel
    "arch_5l_v3": [256, 128, 64, 64, 32],     # Deep compact funnel

    # Aliases for convenience & backward compatibility
    "arch1": [512, 256, 128],
    "arch2": [512, 256, 128, 64],
    "arch3": [512, 384, 256, 128, 64],
}

ARCH_GROUPS = {
    "3_layers": ["arch_3l_v1", "arch_3l_v2", "arch_3l_v3"],
    "4_layers": ["arch_4l_v1", "arch_4l_v2", "arch_4l_v3"],
    "5_layers": ["arch_5l_v1", "arch_5l_v2", "arch_5l_v3"],
    "primary": ["arch_3l_v1", "arch_4l_v1", "arch_5l_v1"],
    "all": [
        "arch_3l_v1", "arch_3l_v2", "arch_3l_v3",
        "arch_4l_v1", "arch_4l_v2", "arch_4l_v3",
        "arch_5l_v1", "arch_5l_v2", "arch_5l_v3"
    ]
}



class FCNN(nn.Module):
    """
    Fully Connected Neural Network (FCNN) for multi-class classification.
    Produces raw logits for PyTorch's CrossEntropyLoss.
    """
    def __init__(self, input_dim=784, hidden_dims=None, num_classes=5, activation="relu"):
        super(FCNN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        if len(hidden_dims) < 3 or len(hidden_dims) > 5:
            # We log a warning if out of bounds, but allow flexible testing
            print(f"Notice: Architecture has {len(hidden_dims)} hidden layers (assignment requires 3 to 5).")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.activation_name = activation.lower()

        act_fn = self._get_activation_fn(self.activation_name)

        layers = []
        in_features = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(act_fn())
            in_features = h_dim

        # Final classification layer (raw logits)
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def _get_activation_fn(self, act_name):
        if act_name == "relu":
            return nn.ReLU
        elif act_name == "tanh":
            return nn.Tanh
        elif act_name == "sigmoid" or act_name == "logistic":
            return nn.Sigmoid
        else:
            raise ValueError(f"Unsupported activation: '{act_name}'. Allowed: {ACTIVATION_CHOICES}")


    def forward(self, x):
        # Flatten if needed: (batch_size, 784)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


ACTIVATION_CHOICES = ["relu", "tanh", "sigmoid"]


def initialize_weights(model, seed=42):
    """
    Deterministically initializes linear layer weights on CPU using Xavier (Glorot) Normal:
      W ~ N(0, std^2), where std = sqrt(2 / (fan_in + fan_out))
      b = 0
    This normal distribution balances variance across layers and is suitable across
    all model architectures and activation functions (ReLU, Tanh, Sigmoid).
    """
    torch.manual_seed(seed)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def get_initial_weights(arch_name, seed=42, checkpoint_dir=CHECKPOINT_DIR):
    """
    Retrieves or generates identical random initial weights for a given architecture config.
    Guarantees requirement (d): All optimizers and activations for this architecture
    start from the EXACT same initial random weights.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    init_path = os.path.join(checkpoint_dir, f"{arch_name}_initial_weights.pt")

    hidden_dims = ARCHITECTURES.get(arch_name, arch_name)
    if not isinstance(hidden_dims, list):
        raise ValueError(f"Unknown architecture: {arch_name}")

    if os.path.exists(init_path):
        state_dict = torch.load(init_path, map_location="cpu")
        return copy.deepcopy(state_dict)

    # Instantiate model and generate deterministic initial weights
    model = FCNN(input_dim=784, hidden_dims=hidden_dims, num_classes=5, activation="relu")
    initialize_weights(model, seed=seed)
    torch.save(model.state_dict(), init_path)
    return copy.deepcopy(model.state_dict())


def build_model(arch_name, num_classes=5, activation="relu", seed=42, checkpoint_dir=CHECKPOINT_DIR):
    """
    Builds the FCNN model and loads the common initial weights for that architecture.
    Every configuration of this architecture (regardless of activation or optimizer)
    starts with 100% identical initial weights.
    """
    if isinstance(arch_name, str):
        if arch_name not in ARCHITECTURES:
            raise ValueError(f"Architecture '{arch_name}' not recognized. Available: {list(ARCHITECTURES.keys())}")
        hidden_dims = ARCHITECTURES[arch_name]
    elif isinstance(arch_name, list):
        hidden_dims = arch_name
        arch_name = f"custom_{len(hidden_dims)}layers"
    else:
        raise TypeError("arch_name must be a string or list of layer sizes")

    model = FCNN(input_dim=784, hidden_dims=hidden_dims, num_classes=num_classes, activation=activation)
    init_state = get_initial_weights(arch_name, seed=seed, checkpoint_dir=checkpoint_dir)
    model.load_state_dict(init_state)
    return model




if __name__ == "__main__":
    print("Testing models and weight reproducibility across all 9 architecture variants...")
    variants = ARCH_GROUPS["all"]
    for arch in variants:
        m1 = build_model(arch, activation="relu")
        m2 = build_model(arch, activation="tanh")
        identical = True
        for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
            if not torch.equal(v1, v2):
                identical = False
                break
        layers_count = len(ARCHITECTURES[arch])
        params_count = sum(p.numel() for p in m1.parameters())
        nodes_str = " -> ".join(str(d) for d in ARCHITECTURES[arch])
        print(f"{arch:11s} ({layers_count} layers: {nodes_str:25s}) | params: {params_count:>9,d} | init match: {identical}")


