# CS726 Assignments

This repository contains assignments completed for the CS726 course, showcasing implementation and analysis of various advanced machine learning and probabilistic modeling techniques.

## Assignment 1: Junction Tree Inference in Graphical Models

**Directory**: `assgn1/Epsilon Delta_22b2104_22b0447_22b0424/`

### Overview
Implementation of a complete inference engine for probabilistic graphical models using the Junction Tree Algorithm for exact inference.

### 🚀 Features
- **Graph Triangulation**: 
  - Detects simplicial vertices
  - Implements minimum-degree heuristic
  - Adds fill-in edges to make the graph chordal

- **Junction Tree Construction**: 
  - Identifies maximal cliques using Bron-Kerbosch algorithm
  - Builds the tree from cliques using a maximum spanning tree of the clique graph

- **Potential Assignment**: 
  - Assigns potential functions to cliques with proper indexing
  - Creates custom mapping functions for efficient computation

- **Message Passing**: 
  - Performs sum-product message passing for belief propagation
  - Computes clique and variable marginals
  - Calculates the partition function (Z)

- **Top-K Assignments**: 
  - Extracts the k most probable variable assignments from the model
  - Implements priority-based message passing for assignment ranking

### Technologies
- Python
- Graph algorithms
- Probabilistic inference techniques

## Assignment 2: Diffusion Probabilistic Models

**Directory**: `assgn2/Epsilon Delta_22b2104_22b0447_22b0424/`

### Overview
Implementation and analysis of Denoising Diffusion Probabilistic Models (DDPM) with conditional generation capabilities.

### Key Features
- **DDPM Implementation**: Built the core diffusion model framework
- **Conditional Generation**: Implemented Classifier-Free Guidance (CFG) for controlled generation
- **Training Pipeline**: Developed efficient training procedures for diffusion models
- **Sample Generation**: Created methods to generate and reproduce samples from trained models
- **Evaluation Metrics**: Integrated quantitative measures to assess generation quality

### Technologies
- PyTorch
- Deep generative models
- Neural network architectures

## Assignment 3: Large Language Model Decoding Strategies

**Directory**: `assgn3/cs726_assgmt3/`

### Overview
Exploration and implementation of efficient decoding strategies for large language models with a focus on machine translation tasks.

### Key Features
- **Decoding Algorithms**: Implemented greedy search and nucleus sampling
- **Constrained Decoding**: Developed methods to generate text with specific word-level constraints
- **Medusa Decoding**: 
  - Single-head implementation for improved generation speed
  - Multi-head variant with parallel token prediction capabilities
- **Performance Evaluation**: 
  - Quality metrics (BLEU, ROUGE)
  - Speed metrics (Real-Time Factor)
- **Translation Task**: Applied to Hindi-to-English translation using Llama-2 models

### Technologies
- PyTorch
- Transformers
- NLP evaluation frameworks

## Assignment 4: Sampling and Estimation Algorithms

**Directory**: `assg4/cs726_assgmt4/`

### Overview
Implementation and comparative analysis of statistical sampling methods and parameter estimation techniques.

### Key Features
- **Sampling Algorithms**: Various techniques for efficient statistical sampling
- **Gaussian Distribution Estimation**: Methods to estimate parameters of Gaussian distributions
- **Comparative Analysis**: Evaluation of different sampling and estimation approaches
- **Modular Implementation**: Organized code structure for different estimation tasks

### Technologies
- Statistical computing
- Monte Carlo methods
- Parameter estimation algorithms

## Team Information

This repository contains work completed by team "Epsilon Delta" (Student IDs: 22b2104, 22b0447, 22b0424).

## Usage Instructions

Each assignment directory contains its own implementation files, documentation, and specific instructions. Please refer to individual README files or reports within each assignment folder for detailed usage guidelines.

## Requirements

- Python 3.8+
- PyTorch (for Assignments 2 and 3)
- Standard scientific Python libraries (NumPy, SciPy, Matplotlib)
- Assignment-specific dependencies as listed in each folder

---

Feel free to clone, test, and extend the work!



