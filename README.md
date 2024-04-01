# CNN Image Classification using Pytorch
This repository offers a comprehensive framework for conducting image classification tasks using Convolutional Neural Networks (CNNs) in PyTorch. It focuses on employing image augmentation for data preprocessing, utilizing transfer learning with pretrained models for custom data, and optimizing and testing these models. This project aims to provide a practical template for preprocessing models and efficiently training them on custom datasets.  

**Training Data(https://drive.google.com/drive/folders/12fc-e8N9md1SqKT-jmTTAQ-_J0Uuk5hB?usp=sharing)**



## Overview

Leveraging deep learning for image classification involves several critical steps: preprocessing data, choosing and optimizing models, and evaluating their performance. This repository simplifies these processes using PyTorch, making advanced machine learning techniques accessible for practical applications. Here's what it covers:


## Image Augmentation (Data Processing)

Effective data preprocessing and augmentation can significantly enhance model performance, particularly in datasets with limited diversity. This project includes examples of:  

- Rotations, translations, and flips
- Color jittering and random cropping
- Implementations using torchvision.transforms

## Transfer Learning and Model Optimization
Transfer learning allows leveraging pretrained models to achieve high accuracy with minimal data. This repository guides you through:  

- Selecting a pretrained model suitable for your dataset
- Fine-tuning models to fit specific classification tasks
- Techniques for model optimization to improve accuracy and efficiency

## Model Training and Testing
Detailed walkthroughs are provided for:

- Setting up training loops
- Choosing loss functions and optimizers
- Evaluating model performance on validation and test sets


## Convolutional Neural Networks (CNNs) Knowlasge Note
A class of deep neural networks, primarily used in analyzing visual imagery. They are particularly known for their ability to extract features through convolutions, compress these features via pooling layers, and make predictions with fully connected layers by weighing the features.

### Convolution Layer
At the heart of a CNN is the process of learning optimal weight parameters. An image is divided into many small regions, and for each region, a weighted sum is calculated to form a feature map. Each element of the feature map is the result of a convolution operation at the corresponding position. For an image with dimensions 32 * 32 * 3 (where 3 represents the RGB color channels), convolutions are performed separately across channels, and their outputs are summed.  

Feature extraction through filters involves using weight parameters to obtain a value (feature value) from convolutions. The size of the filter determines the size of the region in the original image data corresponding to a feature value. Different color channels require different filters. The calculation is done using dot products, including a bias term.  

Choosing different feature matrices (with the same specifications) results in different feature maps, each convolution using the same sum leading to multiple results, known as the depth of the feature map.  

The entire feature extraction process in a convolution layer involves segmenting the input image into many pixels and RGB color channels (e.g., 32 * 32 * 3), selecting filters and stride distances for convolution, adding up the three calculation results along with bias to get the first element of the feature map, and repeating to obtain the entire feature map. Changing the initial weight values can yield different feature maps, known as depth.

### Convolution Layer Parameters
- Stride: Determines how much the window slides. A larger stride results in a larger feature map, capturing finer details.
- Kernel Size: Determines the size of the region chosen for each feature value. Commonly, 3×3 kernels are used.
Edge Padding (Zero Padding): To compensate for the loss of edge features, padding with zeros can be applied around the image.
- Number of Kernels: Determines how many feature maps are produced.  
Convolutional Parameter Sharing  
Using the same set of convolutions to extract features from every region of the image.

### Pooling Layer
With potentially too many features, not all of which are useful, compression (downsampling) is necessary. Pooling selects important features to keep, reducing the dimensions of the feature map without changing its depth. It doesn't involve matrix calculations.

### Pooling Layer Compression
After convolution, the feature map is filtered through a pooling layer, selecting the maximum value (max pooling) from each region.  

### CNN Framework
After each convolution, a nonlinear transformation (similar to backpropagation in neural networks) is applied. Following several convolutions to generate a substantial feature map, pooling is performed. The final result is converted into probability values for classification (fully connected layer).  

Since fully connected layers cannot accept 3D parameters, the processed feature map is flattened into a feature vector, which is then converted into probabilities for classifications.   

Layers that involve parameter calculations are termed as "layers" (convolution and fully connected layers).

### VGG Network
In VGG networks, the convolution size is consistently 3x3. Post-pooling, some information is lost, compensated by doubling the feature value (through the number of feature maps) to mitigate the loss from reductions in length and width.  

### ResNet (Residual Network)
More layers do not necessarily mean better performance. When stacking layers, some may negatively impact performance, necessitating a scheme to eliminate these effects.  

ResNet allows for stacking the original part that learned well, ensuring that even if some parts learned poorly, setting their weight to zero (keeping the useful and nullifying the useless) ensures that the outcome is at least no worse than the original.  

### Receptive Field
The receptive field refers to how many previous values are involved in calculating the current value. Generally, a larger receptive field is preferred.  

Three 3x3 convolutional layers require fewer parameters than a single larger layer, making them more efficient.  

The computation speed is tied to the number of parameters. More convolutions result in finer details, and the inclusion of nonlinear transformations increases accordingly.  