# Automatic Medical Image Segmentation with U-Net and U-Net++

## Project Overview

This dissertation project focuses on the automatic segmentation of medical images using advanced deep learning architectures, specifically U-Net and U-Net++. The primary datasets employed in this project are the Placenta dataset and the BraTS (Brain Tumor Segmentation) dataset. The objective is to enhance the accuracy and efficiency of medical image segmentation, a crucial task traditionally performed manually by radiologists.

## Abstract

Medical MRI image segmentation is a critical task performed by radiologists to quantify pathology. Historically, this task has been handcrafted and time-consuming. In response to this challenge, a successful U-Net deep learning solution was presented at the University of Nottingham (UoN) last year. This work benchmarks the U-Net and introduces U-Net++ with the aim of improving the previous U-Net results. The project demonstrates the effectiveness of these models on the unique placenta dataset from UoN, achieving promising results.

The training process was optimized with the Adam Optimizer, and the loss function employed was a combination of Dice and Binary Cross-Entropy (BCE) loss. The U-Net++ architecture, trained with deep supervision, was also tested with model pruning to enhance inference speed at the cost of some predictive capacity. Cross-validation results for the Placenta dataset yielded a Dice score of 0.9 [0.85, 0.95] for U-Net and 0.92 [0.89, 0.93] for U-Net++. The models were trained and tested on a NVIDIA 2080TI GPU running Red Hat Linux. The code for this implementation is available at: [GitHub Repository](https://github.com/manuelhz/dissertation).

## Table of Contents

1. [Introduction](#introduction)
2. [Datasets](#datasets)
   - [Placenta Dataset](#placenta-dataset)
   - [BraTS Dataset](#brats-dataset)
3. [Methodology](#methodology)
   - [U-Net Architecture](#u-net-architecture)
   - [U-Net++ Architecture](#u-net++-architecture)
4. [Results](#results)
   - [Performance Metrics](#performance-metrics)
   - [Comparison with Previous Work](#comparison-with-previous-work)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)

## Introduction

Semantic segmentation in the medical field plays a significant role in identifying and quantifying pathology, yet manual segmentation is often inefficient. This project utilizes deep learning to automate this process, enabling faster and more accurate segmentation of medical images. The focus is on two datasets: the unique Placenta dataset from UoN and the BraTS dataset for brain tumor segmentation.

## Datasets

### Placenta Dataset
The Placenta dataset comprises 21 3D MRI images, crucial for understanding fetal pathology through placenta segmentation. This dataset is unique to Nottingham University and presents an opportunity for improvement in segmentation techniques.
![](https://github.com/manuelhz/dissertation/blob/main/placenta.png)
![](https://github.com/manuelhz/dissertation/blob/main/placenta2.png)

### BraTS Dataset
The BraTS dataset is a well-known benchmark in brain tumor segmentation, consisting of multimodal 3D images.
![](https://github.com/manuelhz/dissertation/blob/main/brats.png)
![](https://github.com/manuelhz/dissertation/blob/main/brrain2.png)
## Methodology

### U-Net Architecture
U-Net is a convolutional neural network designed for biomedical image segmentation. The architecture consists of a contracting path to capture context and a symmetric expanding path for precise localization.
![](https://github.com/manuelhz/dissertation/blob/main/unet.png)

### U-Net++ Architecture
U-Net++ builds upon the original U-Net by introducing dense skip pathways and deep supervision. This architecture aims to improve feature propagation and reduce the semantic gap between the encoder and decoder.
![](https://github.com/manuelhz/dissertation/blob/main/unetplusplus.png)

## Results

### Performance Metrics
- **Placenta Dataset:**
  - U-Net Dice Score: 0.90 [0.85, 0.95]
  - U-Net++ Dice Score: 0.92 [0.89, 0.93]

### Comparison with Previous Work
The results for the Placenta dataset surpassed previous benchmarks at UoN, demonstrating the effectiveness of U-Net and U-Net++ in this context. In contrast, the BraTS results were lower than competitive benchmarks, highlighting the challenges associated with this dataset.

## Conclusion

The implementation of U-Net and U-Net++ demonstrated promising results, particularly on the Placenta dataset, where the models outperformed previous studies. However, the BraTS dataset revealed limitations that warrant further exploration. This project emphasizes the potential of deep learning in medical image segmentation while acknowledging the need for more advanced techniques and resources.

## Future Work

Future research may include:
- Exploring alternative loss functions and network architectures.
- Investigating the impact of data augmentation and preprocessing techniques.
- Expanding the study to include larger datasets or additional imaging modalities.
- Implementing semi-automated systems to improve segmentation accuracy.

For further details and code access, please visit the [GitHub Repository](https://github.com/manuelhz/dissertation).
