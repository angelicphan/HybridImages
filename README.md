# HybridImages

This was completed individually as a final project for CS410: Computer Vision, taught by Professor Liu Feng at Portland State University during the Winter 2019 term.

This project uses Python 3.

# Project description

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

# To clone the Project

1. Change directory to the desired project location
2. Run: ```git clone https://github.com/angelicphan/HybridImages.git```

# To run the program

1. Download requirements: ```pip3 install -r requirements.txt```
2. Change to the FinalProject folder
3. Run: ```python3 hybrid_images.py to_be_lowpass_img.bmp to_be_highpass_img.bmp resulting_img.bmp sigma_lowpass_value sigma_highpass_value```

Sample images within the folder are provided and obtained from Oliva et. al.

Example results are in the three sub-folders titled in the order that the program was ran.

Example call: ```python3 hybrid_images submarine.bmp fish.bmp sf.bmp 5 3```

# Warnings

- The program takes a really long time to run because of the convolution process, so the sigma values should remain small. Otherwise it will take AN ETERNITY
- You must guess the sigma values that will work best through trial and error

# Sources

- Oliva, Aude, Antonio Torralba & Phillippe G. Schyns. “Hybrid Images”, SIGGRAPH, 2006,
    http://cvcl.mit.edu/publications/OlivaTorralb_Hybrid_Siggraph06.pdf. Accessed February 7 2019.
- Unknown. “Project 1: Image Filtering and Hybrid Images”, Georgia Tech,
    https://www.cc.gatech.edu/~hays/compvision/proj1/. Accessed February 7 2019.
- Liu, Feng. “Introduction to Visual Computing: Lecture 2”, 2019,
    http://web.cecs.pdx.edu/~fliu/courses/cs410/notes/Lecture2.pdf. Accessed February 7 2019.
