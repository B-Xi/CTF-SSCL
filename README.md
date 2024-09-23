## CTF-SSCL: CNN-Transformer for Few-shot Hyperspectral Image Classification Assisted by Semisupervised Contrastive Learning, TGRS, 2024.
[Bobo Xi](https://scholar.google.com/citations?user=O4O-s4AAAAAJ&hl=zh-CN), [Yun Zhang](https://ieeexplore.ieee.org/author/37087032130), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html),  [Zan Li](https://scholar.google.com/citations?user=FL3Mj4MAAAAJ&hl=zh-CN) and [Jocelyn Chanussot](https://jocelyn-chanussot.net/).
***
Code for the paper: [CTF-SSCL: CNN-Transformer for Few-shot Hyperspectral Image Classification Assisted by Semisupervised Contrastive Learning](https://ieeexplore.ieee.org/document/10684809).

We have successfully tested it on Ubuntu 18.04 with PyTorch 1.1.0. Below is the overall architecture of the proposed method. 

<div align=center><p float="center">
<img src="/Overall.png" height="400" width="800"/>
</p></div>
<div align=center>Fig. 2: The overall architecture of the proposed method.</div>  

## Abstract  

Few-shot learning (FSL) has rapidly advanced in the hyperspectral image (HSI) classification, potentially reducing the need for laborious and expensive labeled data collection. Due to the limited receptive field, the convolutional neural network (CNN) struggles to capture long-range dependencies for extracting global features. Additionally, the transformer focuses on global correlation while overlooking the effective representation of local spatial and spectral features. Moreover, contrastive learning has emerged as a powerful technique for improving consistency across different augmented views of samples of the same category.
To this end, we devise a novel CNN-Transformer network for few-shot HSI classification with semisupervised contrastive learning (CTF-SSCL) to boost the classification performance. Specifically, the cascaded CNN-Transformer incorporates a lightweight spatial-spectral interactive convolution module (LSSICM) and a multi-scale transformer (MSFormer) to exploit local features from submaps and global information from the entire patch. Subsequently, the semisupervised contrastive loss, comprising unsupervised and supervised components, serves as an auxiliary to optimize the model with the classification loss. Wherein, recognizing the unified spectral-spatial information in HSI, we propose a spectral feature shift strategy (SFSS) to create sample pairs for the unsupervised contrastive learning, utilizing unsupervised contrastive loss among groups of samples with identical labels. Extensive experiments on four standard benchmarks demonstrate the effectiveness of the proposed CTF-SSCL with varying amounts of labeled samples.

## Training and Test Process
1. Prepare the training and test data as operated in the paper.
2. Run the 'CTF-SSCL-UP-knn.py' to reproduce the CTF-SSCL results on Pavia University data set.

## References

If you find this code helpful, please kindly cite:

[1] B. Xi, Y. Zhang, J. Li, Y. Li, Z. Li and J. Chanussot, "CTF-SSCL: CNN-Transformer for Few-shot Hyperspectral Image Classification Assisted by Semisupervised Contrastive Learning," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3465225.

## Citation Details

BibTeX entry:
```
@ARTICLE{Xi_2024TGRS_CTF-SSCL,
  author={Xi, Bobo and Zhang, Yun and Li, Jiaojiao and Li, Yunsong and Li, Zan and Chanussot, Jocelyn},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CTF-SSCL: CNN-Transformer for Few-shot Hyperspectral Image Classification Assisted by Semisupervised Contrastive Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Transformers;Feature extraction;Contrastive learning;Convolution;Hyperspectral imaging;Training;Few shot learning;few-shot learning;lightweight;multi-scale transformer;semisupervised contrastive learning},
  doi={10.1109/TGRS.2024.3465225}}
```
 
Licensing
--
Copyright (C) 2024 Bobo Xi

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.

