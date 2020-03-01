from skimage import io

im = io.imread('1.jpg')
print(im.shape) # numpy矩阵，(h,w,c)
print(im.dtype)
print(im.size)
io.imshow(im)
io.imsave('sk.png',im)
print(im)

im2 = io.imread('1.jpg',as_grey=True)  #读入灰度图
print(im2.dtype)
print(im2.size)
print(im2.shape)
io.imshow(im2)
io.imsave('sk_gray.png',im2)
io.show()
print(im2)


from skimage import color
im3 = io.imread('1.jpg')
im3 = color.rgb2grey(im3)
print(im3.dtype)
print(im3.size)
print(im3.shape)
io.imshow(im3)
io.show()

'''
skimage.color.rgb2grey(rgb)
skimage.color.rgb2hsv(rgb)
skimage.color.rgb2lab(rgb)
skimage.color.gray2rgb(image)
skimage.color.hsv2rgb(hsv)
skimage.color.lab2rgb(lab)

'''


