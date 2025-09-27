# An in-depth study on using vision transformer-based models for classification of cloud images

## Abstract

As a source of precipitation and affecting Earth’s energy balance, clouds have an important
role in Earth’s weather and climate, and cloud classification is essential in weather forecasting and climate
monitoring. In this paper, we are conducting an in-depth analysis on using vision transformer-based models
to assess their ability to uncover relevant features of different cloud types. A Vision Transformer (ViT)
model and a hybrid variant of it (HyViT) are proposed and evaluated on three ground-based cloud datasets.
We also present an analysis of the interpretations provided by the Local Interpretable Model-agnostic
Explanations from both computational and meteorological perspectives. The experiments highlighted a
statistically significant performance improvement compared to other deep learning architectures used in
the literature. In terms of the Area Under the Receiver Operator Characteristic Curve performance metric,
our approach outperforms by 0.3%-7.2% three other methods from the literature that were replicated and
tested using our proposed methodology.

## Short description of the repository

This repository contains the code and resources for training and evaluating machine learning models for cloud classification using vision transformer-based architectures.
 
The main components of the repository are:
- `model_src/`: contains the source code for data loading, preprocessing, model architectures, training, evaluation, and utilities. Manual training and testing of models can be done through the `main.py` script.
- `JavaServer`: contains the backend server code for handling requests and serving the frontend application, as well as managing user data and resource metadata.
- `web-app/`: contains the React frontend application code for user interaction, visualization, and management of images, predictions and models.

## See Also

Machine Learning pipeline and models used for research: [model_src](model_src/)
