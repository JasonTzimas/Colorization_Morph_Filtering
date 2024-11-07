import cv2
import numpy as np
import matplotlib.pyplot as plt
from include.functions import *
import os
import sys
import itertools
import argparse


def main(input_image, out_folder):

    root = os.getcwd()
    output_path = os.path.join(root, out_folder)

    # PART B: Getting can.jpg file
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    thresh, bin_image = otsus_thresholding(image)


    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(bin_image, cmap="gray")
    axs.set_axis_off()
    axs.set_title("Otsu's Thresholded Can Image", fontsize=16)
    name = "thresh_outsus.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)

    morph_image = opening(bin_image, kernel_erosion=5, kernel_dilation=3)
    morph_image = closing(morph_image, kernel_erosion=3, kernel_dilation=3)
    morph_image = dilation(morph_image, kernel_size=3)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(bin_image, cmap="gray")
    axs.set_axis_off()
    axs.set_title("Binary Image", fontsize=16)
    name = "binary_image.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(morph_image, cmap="gray")
    axs.set_axis_off()
    axs.set_title("Binary Image after morphological filtering", fontsize=16)
    name = "binary_image_after_morph.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)

    label_im, counts = connectivity_labeling(morph_image)
    rgb_image = label_to_color_image_fast(label_im)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(rgb_image)
    axs.set_axis_off()
    axs.set_title("Labeled Image", fontsize=16)
    name = "labeled_image.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)

    # Superimpose initial image with label image
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(image, cmap="gray")
    axs.imshow(rgb_image, alpha=0.5)
    axs.set_axis_off()
    axs.set_title("Labeled Image vs Original", fontsize=16)
    name = "labeled_vs_original.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)


    # Calculating Moments
    # First we need to segment into different lists for each label
    d ={i:[] for i in range(1, counts + 1)}
    for i in range(label_im.shape[0]):
        for j in range(label_im.shape[1]):
            if label_im[i, j] != 0:
                d[label_im[i, j]].append((i, j))

    xis ={i:[it[1] for it in d[i]] for i in range(1, counts + 1)}
    yis ={i:[it[0] for it in d[i]] for i in range(1, counts + 1)}

    mx = {i:np.mean(np.array(xis[i])) for i in range(1, counts + 1)}
    my = {i:np.mean(np.array(yis[i])) for i in range(1, counts + 1)}
    mu01 = {i:np.sum(np.array(xis[i]) - mx[i]) for i in range(1, counts + 1)}
    mu10 ={i:np.sum(np.array(yis[i]) - my[i]) for i in range(1, counts + 1)}
    mu11 = {i:np.sum((np.array(xis[i]) - mx[i]) * (np.array(yis[i]) - my[i])) for i in range(1, counts + 1)}
    mu02 = {i:np.sum((np.array(xis[i]) - mx[i])**2) for i in range(1, counts + 1)}
    mu20 = {i:np.sum((np.array(yis[i]) - my[i])**2) for i in range(1, counts + 1)}
    thetas = {i:np.arctan(2 * mu11[i]/(mu20[i] - mu02[i] + np.sqrt((mu20[i] - mu02[i])**2 + 4 * mu11[i]**2)  + 1e-10)) + np.pi/2 for i in range(1, counts + 1)}


    # Superimpose initial image with label image
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(image, cmap="gray")
    axs.imshow(rgb_image, alpha=0.5)
    axs.scatter(mx.values(), my.values(), linewidth=4, color="red")
    length1, length2 = 10, 30

    # Draw orientation lines
    for i in range(1, counts + 1):
        
        x_center, y_center = mx[i], my[i]
        axs.text(x_center, y_center, 'Can {}'.format(i), fontsize=18)
        x_start = x_center + np.sin(thetas[i]) * length1
        x_end = x_center - np.sin(thetas[i])  * length1
        y_start = y_center + np.cos(thetas[i])  * length1
        y_end = y_center - np.cos(thetas[i]) * length1
        axs.plot([x_start, x_end], [y_start, y_end], 'yellow')  # Line color is yellow for visibility

    # Draw orientation lines
    for i in range(1, counts + 1):
        x_center, y_center = mx[i], my[i]
        x_start = x_center + np.sin(thetas[i] + np.pi / 2) * length2
        x_end = x_center - np.sin(thetas[i] + np.pi / 2)* length2
        y_start = y_center + np.cos(thetas[i] + np.pi / 2) * length2
        y_end = y_center - np.cos(thetas[i] + np.pi / 2) * length2
        axs.plot([x_start, x_end], [y_start, y_end], 'blue')  # Line color is yellow for visibility

    axs.set_axis_off()
    axs.set_title("Axes and Centroids", fontsize=16)
    name = "centroids_orientations.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process an image and save results.")
    parser.add_argument("input_image", type=str, help="Path to the input image file")
    parser.add_argument("out_folder", type=str, help="Path to the output folder")

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.input_image, args.out_folder)

