# CTF-SSCL
This repo is the implementation of the following paper:

**CTF-SSCL: CNN-Transformer for Few-shot Hyperspectral Image Classification Assisted by Semisupervised Contrastive Learning** (TGRS 2024), [[paper]](DOI:10.1109/TGRS.2024.3465225)

## Abstract
Few-shot learning (FSL) has rapidly advanced in the hyperspectral image (HSI) classification, potentially reducing the need for laborious and expensive labeled data collection. Due to the limited receptive field, the convolutional neural network (CNN) struggles to capture long-range dependencies for extracting global features. Additionally, the transformer focuses on global correlation while overlooking the effective representation of local spatial and spectral features. Moreover, contrastive learning has emerged as a powerful technique for improving consistency across different augmented views of samples of the same category.
To this end, we devise a novel CNN-Transformer network for few-shot HSI classification with semisupervised contrastive learning (CTF-SSCL) to boost the classification performance. Specifically, the cascaded CNN-Transformer incorporates a lightweight spatial-spectral interactive convolution module (LSSICM) and a multi-scale transformer (MSFormer) to exploit local features from submaps and global information from the entire patch. Subsequently, the semisupervised contrastive loss, comprising unsupervised and supervised components, serves as an auxiliary to optimize the model with the classification loss. Wherein, recognizing the unified spectral-spatial information in HSI, we propose a spectral feature shift strategy (SFSS) to create sample pairs for the unsupervised contrastive learning, utilizing unsupervised contrastive loss among groups of samples with identical labels. Extensive experiments on four standard benchmarks demonstrate the effectiveness of the proposed CTF-SSCL with varying amounts of labeled samples. The code will be available online at https://github.com/B-Xi/CTF-SSCL.

## Training and Test Process
1. Prepare the training and test data as operated in the paper.
2. Run the 'CTF-SSCL-UP-knn.py' to reproduce the CTF-SSCL results on Pavia University data set.

