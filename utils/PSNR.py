import cv2
import numpy as np
import math
 

 
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



im1= cv2.imread("1.bmp")
im2= cv2.imread("2.png")
print(psnr(im1,im2))
