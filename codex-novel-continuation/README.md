[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hZWYFfQo)
# ST456 Course Project

## Project Overview

This repository contains our ST456 deep learning project on retrieval-augmented English novel continuation. The goal is to generate the next paragraph of a story from the same narrative world and test whether retrieval improves contextual coherence and character consistency.

The implementation is being prepared for Colab-first execution. Data download, preprocessing, training, generation, and evaluation will be handled through code and scripts rather than manual local setup.

## Setup

### Local setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

### Colab setup

In Colab, clone the repository and install the dependencies in a setup cell:

```bash
!git clone <your-private-repo-url>
%cd st456-project-wt2026-gaogou456
!pip install -r requirements.txt
```

All required datasets will be downloaded by project scripts into the working directory. No manual dataset download step will be required.

## Planned Workflow

1. Download public-domain Sherlock Holmes texts using code.
2. Clean and split the texts into paragraph-level continuation examples.
3. Train a plain continuation baseline.
4. Train a retrieval-augmented continuation model.
5. Evaluate with automatic metrics and human evaluation materials.

## Course Information

Deadline: Thursday **30/04/2026, 23:59**

The project should demonstrate neural network architecture design, implementation, training, evaluation, and interpretation. The final report should follow the ICML style and stay within the course page limits.
