# Created by: Angelic Phan
# CS 410: Computer Vision
# Final Project: hybrid images
# Due 3/17/19
'''
Project description:
A hybrid image is an image of an object that looks like one object from a close distance,
but then looks like a different object from a further distance. For example, a hybrid image
that looks like Albert Einstein from a close distance, but looks like Marilyn Monroe from a
further distance. This is the result of using filters to create a high frequency image and a
low frequency image, combining them to create the hybrid image, and using spatial distance to
perceive them differently. The low frequency image is the image you see from far away.
This will need a Gaussian filter. The high frequency image is the image you see from up close.
This will need a Laplacian filter, which is (original image – gaussian filter).

Two different sigma values for the Gaussian filter will be needed to best suit the needs of each
image. For my particular project, I've decided to manually choose the different sigma values by
passing them in through the command line.

I will also write out resulting gaussian and laplacian images of the input images in addition to
the resulting image:
- gaussian1.bmp: gaussian image result of input image 1
- gaussian2.bmp: gaussian image result of input image 2
- laplacian.bmp: laplacian image result of input image 2

To call the program: 
python3 hybrid_images.py to_be_lowpass_img.bmp to_be_highpass_img.bmp resulting_img.bmp sigma_lowpass_value sigma_highpass_value

Sources:
Oliva, Aude, Antonio Torralba & Phillippe G. Schyns. “Hybrid Images”, SIGGRAPH, 2006,
    http://cvcl.mit.edu/publications/OlivaTorralb_Hybrid_Siggraph06.pdf. Accessed February 7 2019.
Unknown. “Project 1: Image Filtering and Hybrid Images”, Georgia Tech,
    https://www.cc.gatech.edu/~hays/compvision/proj1/. Accessed February 7 2019.
Liu, Feng. “Introduction to Visual Computing: Lecture 2”, 2019,
    http://web.cecs.pdx.edu/~fliu/courses/cs410/notes/Lecture2.pdf. Accessed February 7 2019.
'''

import cv2
import numpy as np
import sys

def get_gaussian(sigma, mask):
    '''
    G∂ = ((1 / 2π(∂**2)) * e) ** -((x**2 + y**2) / 2(∂**2))
    constant factor at front makes volume sum to 1
    convolve each row of image with 1D kernel to produce new image
    then convolve each column of new image with same 1D kernel to yield output image
        - normalize output by dividing by sum of all weights
    '''
    gaussian = np.zeros((mask, mask),dtype=np.float32)
    sumed = 0
    x0 = round(mask/2)
    y0 = round(mask/2)
    for y in range(mask):
        for x in range(mask):
            gaussian[y][x] = (1 / (2 * np.pi * (sigma**2))) * (np.exp( -1 * ((((x - x0)**2) + ((y - y0)**2)) / (2 * (sigma**2)))))
            sumed += gaussian[y][x]
    gaussian = gaussian / sumed #normalize

    return gaussian


def apply_gaussian(img, gaussian, mask):
    '''
    constant factor at front makes volume sum to 1
    convolve each row of image with 1D kernel to produce new image
    then convolve each column of new image with same 1D kernel to yield output image
        - img (convolve) gaussian = result
    '''
    img_gaussian = np.zeros_like(img,dtype=np.float32)
    height, width, _ = img.shape
    # traverse image to get info
    offset = round(mask/2)
    for y in range(height):
        for x in range(width):
            Sum = 0 # keep track of sum of the neighborhood
            # traverse gaussian kernel
            for yg in range(0, mask):
                for xg in range(0, mask):
                    iy = y + yg - offset # relates image and kernel indices to get the correct pixel
                    ix = x + xg - offset
                    if yg == offset and xg == offset: # center pixel
                        ny = iy
                        nx = ix
                    if iy >= 0 and ix >= 0 and iy < height and ix < width: # is it within range?
                        Sum = (img[iy][ix] * gaussian[yg][xg]) + Sum # convolve
            #pixel that matches center of filter matrix has the value of the sum of products of the neighborhood
            img_gaussian[ny][nx] = Sum

    return img_gaussian

def apply_laplacian(img2, img2_gaussian):
    # result = img2 - gaussian_img2
    img1_laplacian = np.zeros_like(img2,dtype=np.float32)
    img_laplacian = None
    img_laplacian = np.subtract(img2, img2_gaussian)

    return img_laplacian

def combine_images(img1, img2):
    # result = img1 + img2 = low-frequency + high-frequency
    img_result = None
    img_result = np.add(img1, img2)

    return img_result

#===================================================================================================

if __name__ == "__main__":
    print('===================================================================')
    print('PSU CS 410, Winter 2019, Final Project: hybrid images, Angelic Phan')
    print('===================================================================')

    print("Generating images...")

    # ===== Take in arguments
    path_file_image1 = sys.argv[1]
    path_file_image2 = sys.argv[2]
    path_file_image_result = sys.argv[3]
    Gsigma = sys.argv[4]
    Lsigma = sys.argv[5]

    # ===== Read input images
    img1 = cv2.imread(path_file_image1)
    img2 = cv2.imread(path_file_image2)
    # Convert to 0 - 255 np.float
    img1.astype(np.float32)
    img2.astype(np.float32)
    # Convert range to 0 - 1
    np.divide(img1, 255)
    np.divide(img2, 255)

    # ===== Convert sigma into int and compute kernel values
    Gsigma = int(Gsigma)
    Gmask = (2 * Gsigma) + 1
    Lsigma = int(Lsigma)
    Lmask = (2 * Lsigma) + 1

    # ===== Get Gaussian filter
    Ggaussian = get_gaussian(Gsigma, Gmask)
    Lgaussian = get_gaussian(Lsigma, Lmask)
    
    # ===== Get Gaussian filtered image
    img1_gaussian = apply_gaussian(img1, Ggaussian, Gmask)
    img2_gaussian = apply_gaussian(img2, Lgaussian, Lmask)
    # ===== Get Laplacian filtered image
    img2_laplacian = apply_laplacian(img2, img2_gaussian)
    # ===== Combine the two source images
    img_result = combine_images(img1_gaussian, img2_laplacian)
    
    # ===== Write out resulting image
    # Convert to range 0 - 255
    np.multiply(img_result, 255)
    # Convert to np.uint8
    img_result.astype(np.uint8)
    # save image to path_file_image_result
    cv2.imwrite(path_file_image_result, img_result)

    # ===== Write out LAPLACIAN image
    # Convert to range 0 - 255
    np.multiply(img2_laplacian, 255)
    # Convert to np.uint8
    img2_laplacian.astype(np.uint8)
    # save image to path_file_image_result
    cv2.imwrite("laplacian.bmp", img2_laplacian)

    # ===== Write out GAUSSIAN image
    # Convert to range 0 - 255
    np.multiply(img1_gaussian, 255)
    # Convert to np.uint8
    img1_gaussian.astype(np.uint8)
    # save image to path_file_image_result
    cv2.imwrite("gaussian1.bmp", img1_gaussian)

    # ===== Write out GAUSSIAN image
    # Convert to range 0 - 255
    np.multiply(img2_gaussian, 255)
    # Convert to np.uint8
    img2_gaussian.astype(np.uint8)
    # save image to path_file_image_result
    cv2.imwrite("gaussian2.bmp", img2_gaussian)

    # ===== Program Completed
    print("Image generation completed!")
