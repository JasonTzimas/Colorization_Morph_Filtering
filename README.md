# Image Colorization and Morphological filtering based Instance Segmentation

<p align="center">
  <img src="Images/OpenCV_logo_no_text.png" alt="Image description" width="300" height="300">
</p>

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/username/project/actions)

## Description
1. **_PartA_:**
  Educational Code that implements Image colorization from scratch by first cropping 3 images of different channels from a single image, aligns them using Normalized Cross Correlation (NCC) and then combines them in different permutations to give the final RGB Image.

2. **_PartB_:**
  Educational Code that implements Instance Segmentations of multiple well separated objects within an image. The first part of the process is to perform [Otsu's thresholding](https://en.wikipedia.org/wiki/Otsu%27s_method) to get a first segmentation of the objects. Then, morphological filtering is performed to fill any holes/gaps and refine edges. Finally, a Connected-Components Algorithm returns the separated object instance masks.
 


## Table of Contents
- [Part A](#Part_A)
  - [Frame Detection](#frame-detection)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Part_A

### Frame Detection

The input images are of the following form:
<p align="center">
  <img src="Images/OpenCV_logo_no_text.png" alt="Image description" width="300" height="300">
</p>

## Installation

To install the dependencies, run:

```bash
pip install -r requirements.txt
