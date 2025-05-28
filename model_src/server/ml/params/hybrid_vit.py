hybrid_vit_fixed_params_paper_cnn_scratch = {
    'max_epochs': 80,
    'lr': 0.001, # Higher LR if CNN is from scratch
    'batch_size': 16,
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.01,

    'module__cnn_extractor_name': 'paper_cnn',
    'module__cnn_out_channels': 48, # Matches PaperCNNFeatureExtractor
    'module__pretrained_cnn_path': None, # Train CNN from scratch
    # 'module__pretrained_cnn_path': "/path/to/your/results_base_dir/experiments/DATASET_NAME/paper_cnn_feat/RUN_ID_CNN/single_train_SUB_ID/paper_cnn_feat_....pt",

    # Then params are:
    'module__vit_model_variant': 'vit_b_16', # This is a param of HybridViT -> PretrainedViT
    'module__unfreeze_strategy': 'encoder_tail', # Param of HybridViT -> PretrainedViT
    'module__num_transformer_blocks_to_unfreeze': 2, # Param of HybridViT -> PretrainedViT
    'module__head_dropout_rate': 0.1, # Param of HybridViT -> PretrainedViT

    'module__pipeline_img_h': 224,  # Or whatever your main.py img_size is
    'module__pipeline_img_w': 224,
}