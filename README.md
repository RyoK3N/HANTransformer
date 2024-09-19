# HANTransformer
🔥 __Leveraging Hierarchical Attention Network__ with __Transformer (HANTransformer)__ for __document classification__ tasks using __20 Newsgroups dataset__.

# Hierarchical Attention Network with Transformer for Document Classification



## Table of Contents
1.[Introduction](#introduction)

2.[Features](#features)

3.[Architecture](#architecture)

4.[Installation](#installation)

5.[Data Preprocessing](#data-preprocessong)

6.[Training](#training)

7.[Evaluation](#evaluation)

8.[Usage](#usage)

9.[Project Structure](#project-structure)

10.[Contributing](#contribution)

11.[License](#license)

12.[Acknowledgements](#acknowledgements)


## Introduction <a name="introduction"></a>
The Hierarchical Attention Network with Transformer (HANTransformer) is a sophisticated model designed for document classification tasks. By leveraging a hierarchical structure, the model effectively captures both word-level and sentence-level information, integrating Part-of-Speech (POS) tags and rule-based embeddings to enhance its understanding of textual data. This project demonstrates the implementation, preprocessing, and training processes required to deploy the HANTransformer using the 20 Newsgroups dataset.

## Features

1.__Hierarchical Structure:__ Processes documents at both word and sentence levels.

2.__Transformer-Based:__ Utilizes multi-head self-attention mechanisms for contextual understanding.

3.__Incorporates POS Tags and Rule Embeddings:__ Enhances feature representation with linguistic information.

4.__Scalable Preprocessing:__ Efficiently tokenizes and encodes data using multiprocessing.

5.__Flexible Configuration:__ Easily adjustable hyperparameters for various use-cases.
Comprehensive Training Pipeline: Includes training, evaluation, and model saving functionalities.

## Architecture
The HANTransformer model comprises several key components:

__Fusion Layer:__ Combines word embeddings, POS tag embeddings, and rule embeddings using a gating mechanism.

__Positional Encoding:__ Adds learnable positional information to embeddings.

__Multi-Head Self-Attention:__ Captures dependencies and relationships within the data.

__Transformer Encoder Layers:__ Stacks multiple layers of attention and feed-forward networks for deep feature extraction.

__Attention Mechanisms:__ Applies attention at both word and sentence levels to generate meaningful representations.

__Classification Head:__ Outputs logits corresponding to the target classes.
