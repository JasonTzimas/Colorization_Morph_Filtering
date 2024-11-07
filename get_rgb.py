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
    path_name = out_folder
    full_path = os.path.join(root, path_name)
    os.makedirs(full_path, exist_ok=True)


    # # Let's start with a single image
    image1 = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    fig, axs = plt.subplots(1, 1, figsize=(5, 10))
    axs.imshow(image1, cmap="gray")
    axs.set_axis_off()
    name = "original1.png"
    path = os.path.join(full_path, name)
    fig.savefig(path)

    # Plot the histogram
    mean_vert = np.mean(image1, axis=0)
    mean_hor = np.mean(image1, axis=1)


    # Plot the mean values
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(np.arange(image1.shape[1]), mean_vert, linewidth=3)
    axs[1].plot(np.arange(image1.shape[0]), mean_hor, linewidth=3)
    axs[0].set_title("Mean axis=0 intensity of the image", fontsize=18)
    axs[1].set_title("Mean axis=1 intensity of the image", fontsize=18)
    fig.tight_layout()
    name = "mean_per_axis1.png"
    path = os.path.join(full_path, name)
    fig.savefig(path)


    thresh = 65
    hor_coordinates = [i for i in range(image1.shape[1]) if mean_vert[i] <= thresh and mean_vert[i-1] > thresh or mean_vert[i] > thresh and mean_vert[i-1] <= thresh]
    vert_coordinates = [i for i in range(image1.shape[0]) if mean_hor[i] <= thresh and mean_hor[i-1] > thresh or mean_hor[i] > thresh and mean_hor[i-1] <= thresh]


    # Plot the above points on top of the previous figures
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(np.arange(image1.shape[1]), mean_vert, linewidth=3)
    axs[1].plot(np.arange(image1.shape[0]), mean_hor, linewidth=3)
    axs[0].scatter(hor_coordinates, thresh * np.ones_like(np.array(hor_coordinates)), linewidth=4, color="red")
    axs[1].scatter(vert_coordinates, thresh * np.ones_like(np.array(vert_coordinates)), linewidth=4, color="red")
    axs[0].set_title("Mean axis=0 intensity of the image and identified borders", fontsize=16)
    axs[1].set_title("Mean axis=1 intensity of the image and identified borders", fontsize=16)
    fig.tight_layout()
    name = "mean_per_axis_with_borders1.png"
    path = os.path.join(full_path, name)
    fig.savefig(path)

    x_coords = hor_coordinates[1:-1]
    y_coords = vert_coordinates[1:-1]

    img_coords = [[[y_coords[i], x_coords[0]], [y_coords[i], x_coords[1]], [y_coords[i+1], x_coords[0]], [y_coords[i+1], x_coords[1]]] for i in range(0, len(y_coords), 2)]


    # Get cropped images and display them
    images = []
    for i, subimage in enumerate(img_coords):
        image = image1[subimage[0][0]:subimage[2][0], subimage[0][1]:subimage[3][1]]
        images.append(image)
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.imshow(image, cmap="gray")
        axs.set_axis_off()
        name = "cropped1_{}.png".format(i)
        path = os.path.join(full_path, name)
        fig.savefig(path)

    first_padded_image = np.pad(images[0], ((30, 30), (30, 30)), constant_values=0)


    #Find highest inner product and match first and second picture
    nccs = []
    max_ncc = 0
    for x in range(first_padded_image.shape[1] - images[1].shape[1] - 1):
        for y in range(first_padded_image.shape[0] - images[1].shape[0] - 1):
            ncc = NCC(first_padded_image, images[1], x, y)
            nccs.append(ncc)
            if ncc >= max_ncc:
                max_ncc = ncc
                x_max1 = x
                y_max1 = y    


    # Repeat the same for the third picture
    max_ncc = 0
    nccs = []
    for x in range(first_padded_image.shape[1] - images[2].shape[1] - 1):
        for y in range(first_padded_image.shape[0] - images[2].shape[0] - 1):
            ncc = NCC(first_padded_image, images[2], x, y)
            nccs.append(ncc)
            if ncc >= max_ncc:
                max_ncc = ncc
                x_max2 = x
                y_max2 = y      



    image2 = np.zeros_like(first_padded_image)
    image2[y_max1:y_max1+images[1].shape[0], x_max1:x_max1+images[1].shape[1]] = images[1]
    image3 = np.zeros_like(first_padded_image)
    image3[y_max2:y_max2+images[2].shape[0], x_max2:x_max2+images[2].shape[1]] = images[2]

    new_images = [first_padded_image, image2, image3]
    perms = list(itertools.permutations([0, 1, 2], 3))
    for i, perm in enumerate(perms):
        rgb_image = np.dstack((new_images[perm[0]], new_images[perm[1]], new_images[perm[2]]))
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.imshow(rgb_image)
        axs.set_axis_off()
        name = "rgb1_{}.png".format(i)
        path = os.path.join(full_path, name)
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

