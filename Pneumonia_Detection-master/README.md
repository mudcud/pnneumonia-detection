# Pneumonia Detection with ConvNet using Transfer Learning via ResNet50 

[![made-with-kaggle](https://img.shields.io/badge/Made%20with-Kaggle-lightblue.svg)](https://www.kaggle.com/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![Generic badge](https://img.shields.io/badge/STATUS-COMPLETED-<COLOR>.svg)](https://shields.io/)
[![GitHub license](https://img.shields.io/github/license/teyang-lau/Pneumonia_Detection.svg)](https://github.com/teyang-lau/Pneumonia_Detection/blob/master/LICENSE)



<img src = './Pictures/pneumonia.jpg'>





## **Project Motivation** 

For this project, I applied deep learning and convolutional neural networks onto health data, which is my interest and linked to my background and work experience in psychology and neuroscience. Machine learning will become increasingly important to healthcare, and can help in the diagnosis and prediction of many diseases if used correctly. Here, I also used Pytorch as the deep learning framework as I recently took a course on it and wanted to apply it to a more complicated project.



## **Project Goals** 

1. To build a deep learning model to **detect** pneumonia from x-ray scans
2. Use Pytorch as the framework for building the model
3.  Try out different strategies for improving model performance (e.g., learning rate scheduling, gradient clipping)



## **Project Overview** 

* Resizing images to same size tensors using `ImageFolder` and `torch.transform`
* Data augmentation to create diversity using `torch.transform`
* Using `ResNet50` network to pre-train model on Kaggle's GPU
* Tuning hyperparameters of the model such as learning rate, min-batch size etc
* Evaluating training and validation set to achieve the best model performance



## **About this dataset** 

The [chest X-ray images](https://data.mendeley.com/datasets/rscbjbr9sj/2) (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert. It consists of ~5,000 train images (split into 70% train and 30% validation), and ~600 test images. There is also a data class imbalance, with more pneumonia compared to normal images.

<img src = './Pictures/data_inbalance.png'>



## Chest X-Ray Samples

Here are some sample pictures of the chest x-ray scans and their associated diagnosis. Notice that some pictures are quite hard to distinguish and normal shallow networks/models might not be able to correctly learn and classify them.

<img src = './Pictures/train_samples.png'>



## **Data Augmentation** 

I performed data augmentation by transforming the pictures (random rotation, flipping, random grayscale, and horizontal and vertical shifting) to produce more diversity in the dataset. 

<img src = './Pictures/train_grid.png'>



## **Model Architecture** 

I made use of the power of transfer learning to pretrain the model. As these were already pre-trained, we can use the first layers of weights and add a few additional layers to classify the images into 2 categories. For this project, I used the popular and powerful residual network **ResNet50**. It contains 48 convolutional layers, 1 max pool layer, and 1 adaptive average pooling layer (allowing any image input size). The power of ResNets lies in the **residual blocks/skip-connections**, which allows deeper layers to be trained without hurting performance.

<img src = './Pictures/resnet50_architecture.png'>

From [Ji et al.(2019)](https://www.researchgate.net/publication/331364877_Optimized_Deep_Convolutional_Neural_Networks_for_Identification_of_Macular_Diseases_from_Optical_Coherence_Tomography_Images)



## **Model Training Performance** 

I tuned a few hyperparameters of the model such as the learning rate, mini-batch size and algorithm used. From the plot below, we can see that the training is slightly overfitting.

<img src = './Pictures/train_val_plot.png'>



## **Model Predictions** 

The model achieved an **accuracy of ~89%**. However, since accuracy is not a good metric for class imbalanced data, precision, recall and F1-score should be used instead. The model achieved an almost perfect **recall of 99%**, with only 2 false negatives. **Precision** suffered at **~85%**, probably due to there being less normal images and so the model did not perform that well on them. The harmonic **F1-score is ~92%**. 

<img src = './Pictures/confusion_matrix.png'>



Below are samples of the predictions and their true labels. 



<img src = './Pictures/predictions_labels.png'>



## **Difficulties Faced** 

* **Data class imbalance:** This is quite prevalent in the real world and as data scientists, we should all learn to embrace it and find ways to get around this problem. Usually, one way is to simply collect more data for the undersampled class. However, this is not possible for this dataset and especially for healthcare data, which is very difficult to collect and share. In this kernel, I used **weighted loss** to give more weights to the loss of the normal images. This weighting can be tuned as well to achieve the optimal performance. Stratified sampling is also important in this case, but Pytorch does not yet have any built-in functions.
* **Overfitting:** This dataset is prone to overfitting as seen by the train and validation plots. Therefore, I only selected the model weights that achieved the best/lowest validation loss. However, sometimes the best validation loss are achieved in the early epochs, and it performs very badly on the test set, probably due to class imbalance when batching or sampling. For that reason, I have chosen to select the best validation loss only after certain epochs. However, I am not sure if this is a valid method. For future work, I will research more into this problem.

