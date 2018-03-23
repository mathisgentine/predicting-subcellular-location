# Predicting Subcellular Location

## Overview

Within the last few years the complete sequence has been determined for over 3000 genomes. Predicting the function of a protein has proved to be a difficult task where no clear homology to proteins of known function exists. Knowing the subcellular location of such proteins might be a crucial feature to determine their function. This work presents an approach for predicting the subcellular location (nuclear, mitochondrial, cytosolic or secreted) of non-homologous proteins. This method extracts several N-terminal and global features, and performs classification using a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel.

This work corresponds to the coursework of the module COMPGI10 Bioinformatics, University College London.
Please find further information at: http://www0.cs.ucl.ac.uk/staff/D.Jones/coursework/

## Prerequisites
- Python 3.5
- Numpy 1.14.0
- Pandas 0.22.0
- Scikit-learn 0.19.1
- Biopython 1.70
- Matplotlib 2.1.2
- Seaborn 0.8.1

## Installation
Run `sudo pip install -r requirements.txt`

## Structure
- data/
  - cyto.fasta: Amino acid sequences from cytosol in FASTA format
  - mito.fasta: Amino acid sequences from mitochondria in FASTA format
  - nucleus.fasta: Amino acid sequences from nucleus in FASTA format
  - secreted.fasta: Secreted amino acid sequences in FASTA format
  - blind.fasta: 20 unlabeled amino acid sequences in FASTA format
- src/
  - data_pipeline.py: Parses FASTA data and creates features from the amino acid sequences
  - utils.py: Subcellular location prediction utilities
  - eda.ipynb: Exploratory Data Analysis
  - svm.py: Trains a Support Vector Machine to perform subcellular location prediction
  - rf.py: Trains a Random Forest to perform subcellular location prediction
  - nn.py: Trains a feed-forward network to perform subcellular location prediction
