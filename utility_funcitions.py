import os
import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance


def surface_plot (matrix, title, fig, **kwargs):
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(title)
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)


def load_image(image_path, get_gray=True):
    image = io.imread(image_path)
    gray = lab = rgb = None
    
    if(len(image.shape)==2):
        gray = image
    elif len(image.shape)>=3:
        lab = color.rgb2lab(image)
        rgb = (color.lab2rgb(lab)*255).astype(np.uint8)
        gray = (color.rgb2gray(image)*255).astype(int)
    else:
        print('ERROR: image shape not allowed')
    
    if get_gray:
        return gray
    else:
        return image, lab


def compare_image(images: list, titles: list, size=(16,8)):
    fig, axs = plt.subplots(1, len(images), figsize=size)

    for i, (img, title) in enumerate(zip(images, titles)):
        axs[i].set_title(title)
        axs[i].tick_params(axis="both",which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
        axs[i].imshow(img, cmap='gray',vmin=0,vmax=255)

    plt.show()

def bilateral_filter_slow(img, sigma_d, sigma_r, window_size=7):
    height = img.shape[1]
    length = img.shape[0]
    half_wsize = int(window_size/2)
    
    filtered_img = np.zeros(img.shape)
    for y in range(height):
        for x in range(length):
            summation=0
            normalization=0
            for i in range(window_size):
                for j in range(window_size):
                    p_y = y + i - half_wsize
                    p_x = x + j - half_wsize

                    if (p_x < 0 or p_x >= length or p_y < 0 or p_y >= height):
                        continue

                    #C = c([p_x, p_y], [x, y], sigma_d)
                    #S = s(img[p_x, p_y], img[x, y], sigma_r)
                    #weight = C*S

                    weight = w([p_x, p_y], [x, y], sigma_d, img[p_x, p_y], img[x, y], sigma_r)
                    summation = np.add(summation, weight * img[p_x, p_y])
                    normalization += weight
            
            #print("({},{}): {}".format(x,y,filtered_pixel), summation, normalization)
            filtered_pixel = (summation / normalization).astype(int)
            filtered_img[x, y] = filtered_pixel
            
    return filtered_img

def padding(a, size, pad_value=0):
    shape = np.shape(a)
    shape_pad = list(shape)
    
    shape_pad[0] = shape[0] + size*2
    shape_pad[1] = shape[1] + size*2
    a_pad = np.full(shape_pad, pad_value)
    a_pad[size:shape[0]+size,size:shape[1]+size] = a
    return a_pad

def random_noise(img, min_val, max_val):
    img_n = img + np.random.randint(min_val, high=max_val, size=img.shape, dtype=int)
    img_n[img_n<0] = 0
    img_n[img_n>255] = 255

    return img_n


# Combination of the Closeness and Similarity function (NOT USED)
def w(xi, X, sigma_d, phi, F, sigma_r):
    d = distance.euclidean(xi, X)
    if isinstance(phi, list):
        # List
        delta = distance.euclidean(phi, F)
    else:
        # Scalar
        delta = abs(phi - F)
    return np.exp((-0.5*(d**2/sigma_d**2))+
                    (-0.5*(delta**2/sigma_r**2)))