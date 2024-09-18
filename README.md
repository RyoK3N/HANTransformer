# HANTransformer
Leveraging Hierarchical Attention Network with Transformer (HANTransformer) for document classification tasks using 20 Newsgroups dataset.

# Hierarchical Attention Network with Transformer for Document Classification


1.Table of Contents
2.Introduction
3.Features
4.Architecture
5.Installation
6.Data Preprocessing
7.Training
8.Evaluation
9.Usage
10.Project Structure
11.Contributing
12.License
13.Acknowledgements

## Introduction
The Hierarchical Attention Network with Transformer (HANTransformer) is a sophisticated model designed for document classification tasks. By leveraging a hierarchical structure, the model effectively captures both word-level and sentence-level information, integrating Part-of-Speech (POS) tags and rule-based embeddings to enhance its understanding of textual data. This project demonstrates the implementation, preprocessing, and training processes required to deploy the HANTransformer using the 20 Newsgroups dataset.

## Features

__Hierarchical Structure:__ Processes documents at both word and sentence levels.

__Transformer-Based:__ Utilizes multi-head self-attention mechanisms for contextual understanding.

__Incorporates POS Tags and Rule Embeddings:__ Enhances feature representation with linguistic information.

__Scalable Preprocessing:__ Efficiently tokenizes and encodes data using multiprocessing.

__Flexible Configuration:__ Easily adjustable hyperparameters for various use-cases.
Comprehensive Training Pipeline: Includes training, evaluation, and model saving functionalities.

## Architecture
The HANTransformer model comprises several key components:

__Fusion Layer:__ Combines word embeddings, POS tag embeddings, and rule embeddings using a gating mechanism.

__Positional Encoding:__ Adds learnable positional information to embeddings.

__Multi-Head Self-Attention:__ Captures dependencies and relationships within the data.

__Transformer Encoder Layers:__ Stacks multiple layers of attention and feed-forward networks for deep feature extraction.

__Attention Mechanisms:__ Applies attention at both word and sentence levels to generate meaningful representations.

__Classification Head:__ Outputs logits corresponding to the target classes.