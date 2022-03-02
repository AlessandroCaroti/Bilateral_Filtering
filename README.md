# Bilateral_Filtering (shift-invariant Gaussian filtering)

A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images. It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels. Crucially, the weights depend not only on Euclidean distance of pixels, but also on radiometric differences (e.g., range differences, such as color intensity, depth distance, etc.). This preserves sharp edges. 

This repository contains a project based on "Bilateral Filtering for Gray and Color Images", a 1998 article by C. Tomasi and R. Manduchi about a image-processing technique.