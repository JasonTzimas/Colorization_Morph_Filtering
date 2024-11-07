import numpy as np
import cv2

def NCC(f, g, x, y):
    len_x = g.shape[1]
    len_y = g.shape[0]
    mu_f = np.mean(f[y:y+len_y, x:x+len_x])
    mu_g = np.mean(g)
    sigma_f = np.std(f[y:y+len_y, x:x+len_x])
    sigma_g = np.std(g)
    N = np.sum(g.shape)
    ncc = np.mean(np.multiply((f[y:y+len_y, x:x+len_x] - mu_f) / sigma_f, (g - mu_g) / sigma_g))

    return ncc


# Custom Otsu's implementation
def otsus_thresholding(image):
    flat_image = image.flatten()
    histogram, _ = np.histogram(flat_image, bins=256, range=(0, 256), density=True)

    # Initialization
    max_sigma = 0
    p1 = 0
    p2 = 1
    sumT = np.sum(np.arange(256) * histogram)
    sumF = 0
    mu1 = 0
    for i in range(256):
        p1 += histogram[i]
        if p1 == 0:
            continue
        p2 = 1 - p1
        if p2 == 0:
            break
        sumF += i * histogram[i]

        mu1 = sumF / p1
        mu2 = (sumT - sumF) / p2
        sigma_B = p1 * p2 * (mu1 - mu2) ** 2 
        
        if sigma_B > max_sigma:
            max_sigma = sigma_B
            best_thresh = i

    # Apply threshold
    _, bin_image = cv2.threshold(image, best_thresh, 255, cv2.THRESH_BINARY)
    return best_thresh, bin_image



def erosion(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, constant_values=0)
    x = image.shape[1] + kernel_size - 1
    y = image.shape[0] + kernel_size - 1
    kernel = 255 * np.ones((kernel_size, kernel_size))
    center = kernel_size // 2
    eroded_image = 255 * np.ones_like(image)
    for i in range(center, y - center):
        for j in range(center, x - center):
            if np.any(padded_image[i-center:i+center+1, j-center:j+center+1] == np.zeros_like(kernel)):
                eroded_image[i - center, j - center] = 0
    
    return eroded_image


def dilation(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, constant_values=0)
    x = image.shape[1] + kernel_size - 1
    y = image.shape[0] + kernel_size - 1
    kernel = 255 * np.ones((kernel_size, kernel_size))
    center = kernel_size // 2
    dilated_image = np.zeros_like(image)
    for i in range(center, y - center):
        for j in range(center, x - center):
            if np.any(padded_image[i-center:i+center+1, j-center:j+center+1] == kernel):
                dilated_image[i - center, j - center] = 255
    
    return dilated_image

def opening(image, kernel_erosion=3, kernel_dilation=3):
    eroded_image = erosion(image, kernel_size=kernel_erosion)
    dilated_image = dilation(eroded_image, kernel_size=kernel_dilation)
    return dilated_image

def closing(image, kernel_erosion=5, kernel_dilation=3):
    dilated_image = dilation(image, kernel_size=kernel_dilation)
    eroded_image = erosion(dilated_image, kernel_size=kernel_erosion)
    return eroded_image



def label_to_color_image_fast(label_image):
    # Get unique labels and their indices in the original image
    unique_labels, inverse_indices = np.unique(label_image, return_inverse=True)
    
    # Generate random colors for each unique label
    np.random.seed(0)  # For reproducibility
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))
    
    # Assign the first color (usually for label 0) as black or any other color for 'no label'
    colors[0] = [0, 0, 0]  # Assuming the first label is 0 and represents 'no label'
    
    # Map each label to its corresponding color
    color_image = colors[inverse_indices].reshape(label_image.shape + (3,))
    
    return color_image


def connectivity_labeling(bin_image):
    label = 1
    label_image = np.zeros_like(bin_image)
    for i in range(label_image.shape[0]):
        for j in range(label_image.shape[1]):
            if bin_image[i, j] == 255:
                if label_image[i, j] == 0: # Pixel not yet labeled
                    # Label all component-connected pixels
                    label_image[i, j] = label
                    queue = [(i, j)]
                    while queue:
                        tail = queue[-1]
                        queue.pop()
                        new_queue = [(k, n) for k in range(tail[0]-1, tail[0]+2) for n in range(tail[1]-1, tail[1]+2) if (k != tail[0] or n != tail[1])
                                     and k >= 0 and k <= label_image.shape[0] - 1
                                      and n >= 0 and n <= label_image.shape[1] - 1 and (bin_image[k, n] == 255) and label_image[k, n] == 0]
                        for pix in new_queue: 
                            label_image[pix] = label
                        queue = new_queue + queue
                    label += 1

    return label_image, label - 1



def calculate_hu_moments(mu20, mu02, mu11, mu30, mu12, mu21, mu03, mu00):
    # Normalized central moments
    nu20 = mu20 / mu00**2
    nu02 = mu02 / mu00**2
    nu11 = mu11 / mu00**2
    nu30 = mu30 / mu00**2.5
    nu12 = mu12 / mu00**2.5
    nu21 = mu21 / mu00**2.5
    nu03 = mu03 / mu00**2.5

    # Hu Moments
    hu1 = nu20 + nu02
    hu2 = (nu20 - nu02)**2 + 4*nu11**2
    hu3 = (nu30 - 3*nu12)**2 + (3*nu21 - nu03)**2
    hu4 = (nu30 + nu12)**2 + (nu21 + nu03)**2
    hu5 = (nu30 - 3*nu12) * (nu30 + nu12) * ((nu30 + nu12)**2 - 3*(nu21 + nu03)**2) + (3*nu21 - nu03) * (nu21 + nu03) * (3*(nu30 + nu12)**2 - (nu21 + nu03)**2)
    hu6 = (nu20 - nu02) * ((nu30 + nu12)**2 - (nu21 + nu03)**2) + 4*nu11 * (nu30 + nu12) * (nu21 + nu03)
    hu7 = (3*nu21 - nu03) * (nu30 + nu12) * ((nu30 + nu12)**2 - 3*(nu21 + nu03)**2) - (nu30 - 3*nu12) * (nu21 + nu03) * (3*(nu30 + nu12)**2 - (nu21 + nu03)**2)

    return hu1, hu2, hu3, hu4, hu5, hu6, hu7

