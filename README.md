# A new perspective on using generative learning and multimodal data fusion for the classification of severe weather phenomena

## Abstract

Clouds play an essential role in Earth's climate, being the source of precipitation, controlling the amount of solar energy that reaches the surface of our planet, as well as affecting the overall Earth radiation budget. Thus, cloud classification is important as it helps in weather forecasting and supervising climate changes. Clouds are organized in many forms, their composition and density are variable, with different colours and heights, which makes their classification a challenging one. This paper proposes several Machine earning (ML) models and hybrid pipelines, including a Diffusion-based and a Vision Transformer-based classifier, for classifying various cloud images, and analyses their performance and ability of learning to extract and classify the features of distinct cloud types. Experiments are conducted on two ground-based cloud data sets of different sizes and characteristics, using data augmentation and by training each of the implemented approaches. The performances of the best proposed versions are then compared to that of other deep learning architectures used in the literature for clouds classification. The comparative analysis highlights metric scores around the same range, in terms of Accuracy, Precision, Recall and F1-score.

## TODO (brainstorming)
- [ ] move code from jupyter notebook to separate files (datasets, models, training code, main file, utils) 
- [ ] fix random seed (s.t. results are reproducible)
- [ ] specify hyper-parameters in config files
- [ ] check for each dataset if the train-test splits are provided and add code for using the original test sets and selecting a subset of the training set for validation
- [ ] maybe refactor the dataset code in order to be easy to run experiments with different datasets
- [ ] maybe report train and val accuracy during training as well (not only the loss)
- [ ] integrate diffusion classifier (reuse dataset and metrics code so that we can be sure that the results obtained with the two models are comparable)
