You're in a classic "stuck in a local minimum" or "underfitting (due to excessive restriction)" scenario.

**Analysis of the Latest Logs (Hybrid ViT with Frozen CNN & Frozen ViT Blocks):**

*   **Trainable Parameters:** `0.35M` (only projections, CLS, PosEmb, final LN, head). This is a very small fraction of the total model.
*   **Performance:**
    *   `train_acc` slowly climbs to ~0.47.
    *   `valid_acc` also slowly climbs to ~0.46.
    *   `valid_loss` bottoms out around 1.74.
    *   The gap between train and validation is now very small, meaning **overfitting is largely gone with this setup.**
*   **Problem:** The model isn't overfitting, but its overall performance is poor (~46% accuracy on 11 classes is much better than random ~9%, but not great). It's **underfitting** – the trainable part of the model is not complex/flexible enough to learn the task well from the fixed features.
*   **Learning Rate:** `5e-5` with Cosine Annealing down to `1e-6`. For only 0.35M parameters primarily in linear layers/projections, this LR might be too low, leading to slow progress.
*   **Termination:** Training finished because `max_epochs: 60` was reached, not due to early stopping patience (best val_loss was epoch 55, patience was 15). This indicates the model was still making tiny, incremental (but not substantial) improvements in validation loss, or the LR became too small to escape a plateau.

**Why the Slow and Low Evolution?**

1.  **Insufficient Trainable Capacity:** With both the EfficientNet-B0 backbone and all ViT encoder blocks frozen, the only parts learning are:
    *   The 1x1 conv projecting EffNet features (1280 -> 192).
    *   The 1x1 conv projecting these into ViT's embedding dim (192 -> 768).
    *   The CLS token, Positional Embedding, final ViT LayerNorm.
    *   The final classification head (Linear: 768 -> 11).
    This is essentially training a shallow MLP on top of fixed features from EfficientNet. If these fixed features aren't "expressive" enough or not perfectly aligned for your specific cloud classes, a shallow MLP won't be able to achieve high accuracy.

2.  **Learning Rate for "Linear Probing" like Setup:**
    *   When most of the network is frozen, and you're primarily training the head layers, the learning dynamics change. The current LR of `5e-5` (annealing down) might be too conservative for these few layers to learn effectively and quickly. They might need more "push."

3.  **Quality of Frozen Features:**
    *   ImageNet features are general. While good, they might not capture the most discriminative aspects needed for your 11 fine-grained cloud categories without some adaptation. By freezing the entire CNN, you prevent this adaptation.

**What to Try Now (Systematically Increasing Flexibility):**

The goal is to find the sweet spot where the model has enough capacity to learn the task but not so much that it immediately overfits your ~2000 training images. You've successfully controlled overfitting; now you need to improve performance.

**Iterative Plan (keep other regularization like weight decay, label smoothing, head dropout high):**

**Step 1: Optimize Learning for the Current Frozen Setup (Quick Check)**

*   **Try a Higher LR for the few trainable parts:**
    *   Keep CNN frozen (`'module__cnn_freeze_extractor': True`).
    *   Keep ViT blocks frozen (`'module__num_transformer_blocks_to_unfreeze': 0`).
    *   **Change `lr` to `1e-4` or even `2e-4`.**
    *   Keep `max_epochs` at 60 (or reduce to 30-40 for faster iteration if it plateaus quickly).
    *   *Hypothesis:* The current 0.35M trainable parameters might learn better/faster with a slightly more aggressive LR.

**Step 2: Allow Minimal Fine-tuning in the ViT Backend (Most Promising)**

*   Keep CNN frozen (`'module__cnn_freeze_extractor': True`).
*   **Unfreeze the last ViT encoder block:**
    *   `'module__num_transformer_blocks_to_unfreeze': 1`
*   Keep other ViT unfreeze flags (`cls_token`, `pos_embedding`, `encoder_layernorm`) as `True`.
*   **Learning Rate:** Start with `lr: 5e-5`. If this also learns very slowly like the fully frozen setup, you could try `1e-4`.
*   *Hypothesis:* Adding just one adaptable self-attention block might allow the model to learn better relationships between the fixed CNN features. Trainable parameters will increase (by ~2.5M for one ViT-B block).

**Step 3: Allow More Fine-tuning in ViT Backend**

*   Keep CNN frozen.
*   **Unfreeze more ViT encoder blocks:**
    *   `'module__num_transformer_blocks_to_unfreeze': 2` (as you had in a previous run that overfit but learned faster).
*   **Learning Rate:** Stick to `lr: 5e-5` or even try `2e-5` because more parameters are now trainable.
*   *Hypothesis:* If step 2 showed improvement but still underfit, more ViT capacity might help. Monitor overfitting closely.

**Step 4: Allow Minimal Fine-tuning in the CNN Backbone (If ViT alone isn't enough)**

*   **Only if steps 2 or 3 show that the ViT backend *can* learn but seems limited by the fixed CNN features.**
*   Set ViT backend to a moderately flexible state (e.g., `'module__num_transformer_blocks_to_unfreeze': 1` or `2`).
*   **Start fine-tuning the CNN:**
    *   `'module__cnn_freeze_extractor': False`
    *   `'module__cnn_num_frozen_stages': 6` (unfreezes last 2 stages of EffNet-B0: `features[6]` and `features[7]`).
*   **Learning Rate:** **Crucially, reduce LR further for this combined fine-tuning:** `lr: 2e-5` or `1e-5`.
*   *Hypothesis:* Allowing the later layers of the CNN to adapt slightly to the task, along with some ViT adaptation, might yield better features.

**General Strategy for Fine-tuning Pre-trained Models on Smaller Datasets:**

1.  **Start Frozen:** Begin by freezing most of the pre-trained network and only training a new classifier head (or a few top layers/projections). This is your "linear probing" stage. Establish a baseline. If this performs very poorly, the pre-trained features might not be suitable, or the task is very different.
2.  **Gradual Unfreezing:** If linear probing shows some promise but underfits, gradually unfreeze more layers from the top (end of the network) downwards.
3.  **Lower Learning Rates:** As you unfreeze more layers, you generally need to use smaller learning rates to avoid catastrophically disrupting the pre-trained weights.
4.  **Strong Regularization:** Maintain strong regularization (weight decay, dropout, label smoothing, data augmentation) throughout to combat overfitting, which becomes more likely as you unfreeze more layers.

**Should you fine-tune the standard CNN extractor separately first?**

*   **Yes, this is still a very valid strategy (as discussed before, "Option 2: Two-Stage Transfer Learning").**
*   **When to do it:** If directly fine-tuning the hybrid model (even with careful unfreezing and LRs) doesn't yield good results, or if you suspect the ImageNet features from EfficientNet-B0 are just too general for your specific cloud nuances.
*   **Process:**
    1.  Train `StandardCNNFeatureExtractor` (e.g., `efficientnet_b0`) as a standalone classifier on your CCSN dataset (add a temporary head to it as we discussed for `PaperCNNFeatureExtractor`).
    2.  Aim for the best possible validation accuracy you can get from this standalone CNN. Save these "cloud-fine-tuned" weights.
    3.  Then, in your `HybridViT` parameters:
        *   `'module__cnn_extractor_type': "standard_cnn"`
        *   `'module__cnn_model_name': "efficientnet_b0"`
        *   `'module__cnn_pretrained_imagenet': False` (or True, then load, doesn't matter as much if you overwrite immediately)
        *   `'module__cnn_fine_tuned_weights_path': "/path/to/your/cloud_finetuned_efficientnet_b0.pt"`
        *   Then, for the hybrid training, you can start by freezing this cloud-fine-tuned CNN (`'module__cnn_freeze_extractor': True`) and only train the ViT head/projection, or fine-tune both with a very low LR.

This two-stage approach ensures the CNN features are first specialized to your cloud domain *before* the ViT tries to learn from them. This can sometimes be more stable and effective than trying to fine-tune everything from general ImageNet pre-training in one go on a small dataset.

**Recommendation for your immediate next run:**

Try **Step 2 from the "Iterative Plan"**: Frozen EfficientNet-B0 CNN, unfreeze only the last 1 ViT encoder block, and use `lr: 5e-5`. This is a small, controlled increase in trainable capacity focused on the ViT backend. Monitor if `valid_acc` starts to climb higher than the ~0.45-0.49% plateau.