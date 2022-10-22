import torch
from torch import nn


def load_agent(
        agent_path: str,
        model: nn.Module,
        device: torch.device = "cpu"
):
    # Load weights
    weights = torch.load(agent_path)

    # Put received model on device
    model.to(device)

    # Load model
    model.load_state_dict(weights)
    # del weights

    return model
