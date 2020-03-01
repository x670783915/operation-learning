from scipy import misc
import matplotlib.pyplot as plt

im = misc.imread('1.jpg')
print(im.dtype)
print(im.size)
print(im.shape)
misc.imsave('misc1.png',im)
plt.imshow(im)
plt.show()
print(im)

import imageio
im2 = imageio.imread('1.jpg')
print(im2.dtype)
print(im2.size)
print(im2.shape)
plt.imshow(im)
plt.show()
print(im2)
imageio.imsave('imageio.png',im2)

