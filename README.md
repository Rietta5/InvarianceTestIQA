InvarianceTestIQA
================

This repository contains all the code needed to reproduce the paper *Invariance of deep image quality metrics to affine transformations* (Alabau-Bosque et al.): https://arxiv.org/abs/2407.17927. By Nuria Alabau-Bosque, Paula Daudén-Oliver, Jorge Vila-Tomás, Valero Laparra and Jesús Malo.

## Abstract
> Deep architectures are the current state-of-the-art in predicting subjective image quality. Usually, these models are evaluated according to their ability to correlate with human opinion in databases with a range of distortions that may appear in digital media. However, these oversee affine transformations which may represent better the changes in the images actually happening in natural conditions. Humans can be particularly invariant to these natural transformations, as opposed to the digital ones. In this work, we evaluate state-of-the-art deep image quality metrics by assessing their invariance to affine transformations, specifically: rotation, translation, scaling, and changes in spectral illumination. We propose a methodology to assign invisibility thresholds for any perceptual metric. This methodology involves transforming the distance measured by an arbitrary metric to a common distance representation based on available subjectively rated databases. We psychophysically measure an absolute detection threshold in that common representation and express it in the physical units of each affine transform for each metric. By doing so, we allow the analyzed metrics to be directly comparable with actual human thresholds. We find that none of the state-of-the-art metrics shows human-like results under this strong test based on invisibility thresholds. This means that tuning the models exclusively to predict the visibility of generic distortions may disregard other properties of human vision as for instance invariances or invisibility thresholds.

## Repository walkthrough
- `walkthrough.ipynb`: Notebook containing the methodology developed in the paper. It is explained step by step so that anyone can reproduce it with their own metric.
- `Baldur.jpeg` & `im178.png`: example images used during the notebook to showcase different transformations and how to calculate the metric values.
- `Fit_Ellipses_TID.png`: Notebook explaining how to fit illuminant change ellipses with TID13 data.
- `*.csv`: Necessary data to reproduce the paper.
- `utils.py`: Helper functions used during `walkthrough.ipynb`.

## How to reproduce
1. Clone the repository: `git clone https://github.com/Rietta5/InvarianceTestIQA.git`
2. Run `walkthrough.ipynb`
