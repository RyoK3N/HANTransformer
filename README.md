# HANTransformer
🔥 __Leveraging Hierarchical Attention Network__ with __Transformer (HANTransformer)__ for __document classification__ tasks using __20 Newsgroups dataset__.

# Hierarchical Attention Network with Transformer for Document Classification



## Table of Contents
1).[Introduction](#introduction)

2).[Features](#features)

3).[Architecture](#architecture)

4).[Installation](#installation)

5).[Data Preprocessing](#data-preprocessong)

6).[Training](#training)

7).[Evaluation](#evaluation)

8).[Usage](#usage)

9).[Project Structure](#project-structure)

10).[Contributing](#contribution)

11).[License](#license)

12).[Acknowledgements](#acknowledgements)


## Introduction <a name="introduction"></a>
The Hierarchical Attention Network with Transformer (HANTransformer) is a sophisticated model designed for document classification tasks. By leveraging a hierarchical structure, the model effectively captures both word-level and sentence-level information, integrating Part-of-Speech (POS) tags and rule-based embeddings to enhance its understanding of textual data. This project demonstrates the implementation, preprocessing, and training processes required to deploy the HANTransformer using the 20 Newsgroups dataset.

## Features

    1).__Hierarchical Structure:__ Processes documents at both word and sentence levels.
    
    2).__Transformer-Based:__ Utilizes multi-head self-attention mechanisms for contextual understanding.
    
    3).__Incorporates POS Tags and Rule Embeddings:__ Enhances feature representation with linguistic information.
    
    4).__Scalable Preprocessing:__ Efficiently tokenizes and encodes data using multiprocessing.
    
    5).__Flexible Configuration:__ Easily adjustable hyperparameters for various use-cases.
    Comprehensive Training Pipeline: Includes training, evaluation, and model saving functionalities.

## Architecture
The HANTransformer model comprises several key components:

    1).__Fusion Layer:__ Combines word embeddings, POS tag embeddings, and rule embeddings using a gating mechanism.
    
    2).__Positional Encoding:__ Adds learnable positional information to embeddings.
    
    3).__Multi-Head Self-Attention:__ Captures dependencies and relationships within the data.
    
    4).__Transformer Encoder Layers:__ Stacks multiple layers of attention and feed-forward networks for deep feature extraction.
    
    5).__Attention Mechanisms:__ Applies attention at both word and sentence levels to generate meaningful representations.
    
    6).__Classification Head:__ Outputs logits corresponding to the target classes.

## Installation

### Prerequisites

        Python 3.7+
        pip package manager

### Clone the Repository

        git clone https://github.com/yourusername/your-repo-name.git
        cd your-repo-name

### Create a Virtual Environment 

    conda create -n HANT python=3.11
    conda activate HANT

### Install Dependencies

        pip install -r requirements.txt

### Download spaCy Model

        python -m spacy download en_core_web_sm

## Data Preprocessing

The preprocessing pipeline tokenizes the text data, builds vocabularies, encodes the texts, POS tags, and rules, and saves the processed data for training.

__Steps:__

        1).Tokenization: Splits documents into sentences and words.
        
        2).Vocabulary Building: Constructs vocabularies for words, POS tags, and rules.
        
        3).Encoding: Converts tokens and tags into numerical IDs.
        
        4).Mask Creation: Generates attention and sentence masks to handle padding.
        
        5).Saving Processed Data: Stores the preprocessed data in JSON format.





