import numpy as np
import torch


def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix."""
    H = size[2].item() if torch.is_tensor(size[2]) else size[2]
    W = size[3].item() if torch.is_tensor(size[3]) else size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return int(bbx1), int(bby1), int(bbx2), int(bby2)  # Ensure Python ints


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):  # Added type hints
    """Applies CutMix to a batch of images."""

    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)

    batch_size = x.size(0)

    index = torch.randperm(batch_size, device=device)

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)

    x_cutmix = x.clone()

    patch_to_paste = x[index, :, bby1:bby2, bbx1:bbx2]

    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = patch_to_paste

    img_h = x.size(-2)
    img_w = x.size(-1)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_w * img_h))

    return x_cutmix, y_a, y_b, lam
