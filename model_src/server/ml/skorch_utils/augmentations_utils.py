# server/ml/augmentations/cutmix_utils.py
import numpy as np
import torch


def rand_bbox(size, lam):  # size is torch.Size
    """Generates a random bounding box for CutMix."""
    H = size[2].item() if torch.is_tensor(size[2]) else size[2]  # Get Python int
    W = size[3].item() if torch.is_tensor(size[3]) else size[3]  # Get Python int
    cut_rat = np.sqrt(1. - lam)  # lam is float
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
    # x and y are expected to be on 'device' already when passed from SkorchModelAdapter

    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)  # lam is float

    batch_size = x.size(0)  # Use .size(0) or .shape[0]

    # Ensure index is created on the correct device right away, or moved.
    # .to(device) is correct.
    index = torch.randperm(batch_size, device=device)

    # y_a and y_b will be on the same device as y and y[index]
    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)  # Pass x.shape (torch.Size)

    # x_cutmix will be on the same device as x
    x_cutmix = x.clone()

    # Check devices just before the problematic line for debugging
    # print(f"Device of x_cutmix: {x_cutmix.device}")
    # print(f"Device of x: {x.device}")
    # print(f"Device of index: {index.device}")
    # print(f"Slice indices: {bby1, bby2, bbx1, bbx2} (should be int)")

    # This line is the most likely source if there's a subtle device mismatch
    # with advanced indexing.
    # Ensure x[index,...] does not inadvertently move to CPU if x is sparse or has specific properties.
    # However, x is a dense tensor from DataLoader.
    patch_to_paste = x[index, :, bby1:bby2, bbx1:bbx2]

    # Ensure patch_to_paste is on the same device as x_cutmix (it should be if x and index are)
    # if patch_to_paste.device != x_cutmix.device:
    #    patch_to_paste = patch_to_paste.to(x_cutmix.device) # Explicit move if somehow different

    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = patch_to_paste

    # Adjust lambda to actual pixel ratio
    # Ensure dimensions used for calculation are Python numbers
    img_h = x.size(-2)  # x.size(2) or x.shape[2]
    img_w = x.size(-1)  # x.size(3) or x.shape[3]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_w * img_h))

    return x_cutmix, y_a, y_b, lam
