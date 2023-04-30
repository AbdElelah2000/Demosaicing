# Demosaicing
This project is focused on implementing a highly effective demosaicing algorithm based on linear regression. The goal is to restore full-color images from raw images captured by digital cameras using a color filter array (CFA) and the Bayer Pattern. The Bayer Pattern creates a mosaic of color pixels, and demosaicing algorithms are used to digitally restore the full-color images.

The project consists of the following tasks and requirements:

Simulate four types of mosaic patches from full-color patches.
Solve a linear least square problem for each case and obtain eight optimal coefficient matrices.
Apply the coefficient matrices to each patch of a simulated mosaic image to approximate the missing colors.
Measure the root mean square error (RMSE) between the demosaiced image and the ground truth.
Run the program on test raw mosaic data provided (to be released prior to the deadline), record the process and output results in a video, and submit the demo video.

This script produces a lower RMSE error than that of the MATLAB built-in demosaic() function. Here's a sample of the input and output of the script:


Input:



![image](https://user-images.githubusercontent.com/82244228/235380852-cfb27a99-cfa4-406c-8dbb-7503bdb3ae0a.png)

Output:




![image](https://user-images.githubusercontent.com/82244228/235380866-1dd8364d-ad73-41e3-a177-0a45c563eade.png)
![image](https://user-images.githubusercontent.com/82244228/235380873-ab4f5859-6e4f-47b6-85a2-20ff1271d848.png)
