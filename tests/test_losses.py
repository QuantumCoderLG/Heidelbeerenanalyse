import torch

from src.training.losses import build_loss


def test_dice_bce_loss_small_on_perfect_prediction():
    config = {
        "mode": "dice_bce",
        "dice_weight": 1.0,
        "bce_weight": 1.0,
        "smooth": 1e-5,
    }
    loss_fn = build_loss(config)
    logits = torch.full((1, 1, 8, 8), -4.0)
    target = torch.zeros((1, 1, 8, 8))
    target[:, :, 2:6, 2:6] = 1.0
    logits[:, :, 2:6, 2:6] = 4.0
    loss = loss_fn(logits, target)
    assert loss.item() < 0.06


def test_focal_dice_loss_backpropagates():
    config = {
        "mode": "focal_dice",
        "dice_weight": 1.0,
        "focal_weight": 1.0,
        "focal_alpha": 0.5,
        "focal_gamma": 2.0,
        "smooth": 1e-5,
    }
    loss_fn = build_loss(config)
    logits = torch.zeros((2, 1, 6, 6), requires_grad=True)
    target = torch.zeros((2, 1, 6, 6))
    target[:, :, 1:4, 1:4] = 1.0
    target[:, :, 3:5, 3:5] = 1.0
    loss = loss_fn(logits, target)
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)
