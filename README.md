# PulmoVision: Automated Pneumonia Diagnosis Using Deep Learning
Pneumonia is an inflammatory condition primarily affecting the lungs, characterized by symptoms such as cough, chest pain, fever, and difficulty breathing. The goal of this project is to develop an automated system for detecting and classifying pneumonia in medical images.

Live App: https://pulmovision-ai.streamlit.app/ 

## Project Overview

Respiratory infections such as pneumonia remain a significant global health concern, often leading to severe complications if not diagnosed early. This project focuses on designing an intelligent deep learning system capable of analyzing chest X-ray images to identify signs of pneumonia automatically. The objective is to assist healthcare professionals by providing a reliable computer-aided diagnostic solution for faster clinical decision-making.

## Project Objective

The primary objective of this project is to utilize modern deep learning techniques to recognize pneumonia from chest radiographs with high accuracy. By reducing the dependence on manual image interpretation, the proposed system aims to improve diagnostic efficiency, minimize human error, and support early medical intervention.

## Methodology

A transfer learning strategy was adopted to build the classification model. Pre-trained ResNet convolutional neural network architectures were fine-tuned on a labeled chest X-ray dataset to distinguish between normal lungs and pneumonia cases. The implementation was carried out in Python using TensorFlow, while Google Colab's GPU environment accelerated model training. TensorBoard was employed to monitor training progress, evaluate performance metrics, and visualize learning behavior.

## Technologies and Tools
Python
TensorFlow
ResNet Deep Learning Models
Google Colab (GPU)
TensorBoard
Dataset Description

The model was trained and evaluated using a publicly available chest X-ray dataset containing 5,863 JPEG images. The data is divided into three subsets for training, validation, and testing, with images classified into two categories:

Normal
Pneumonia

The radiographs were collected from pediatric patients aged 1 to 5 years at the Guangzhou Women and Children's Medical Center, China. Before inclusion in the dataset, every image underwent quality assessment to eliminate poor-quality scans. Medical experts reviewed and labeled each X-ray, while an additional specialist validated the evaluation set to ensure annotation reliability and reduce diagnostic inconsistencies.

The dataset can be accessed on Kaggle under Chest X-Ray Images (Pneumonia). (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) 

## Results and Outcome

The proposed deep learning model demonstrated strong performance in identifying pneumonia from chest X-ray images and effectively differentiated healthy cases from infected ones. The project highlights the potential of transfer learning in medical image analysis and illustrates how artificial intelligence can serve as a supportive tool for radiologists by enabling faster and more consistent pneumonia screening.
