from PIL import Image
import numpy as np

img = Image.open('1.jpg')
print(img.format) 
print(img.size) #注意，省略了通道 (w，h)
print(img.mode)  #L为灰度图，RGB为真彩色,RGBA为加了透明通道
img.show() # 显示图片

gray = Image.open('1.jpg').convert('L')
gray.show()

#读取不到图片会抛出异常IOError，我们可以捕捉它，做异常处理
try:
    img2 = Image.open('2.jpg')
except IOError:
    print('fail to load image!')

#pillow读进来的图片不是矩阵，我们将图片转矩阵,channel last
arr = np.array(img3)
print(arr.shape)
print(arr.dtype)
print(arr)


arr_gray = np.array(gray)
print(arr_gray.shape)
print(arr_gray.dtype)
print(arr_gray)


#矩阵再转为图像
new_im = Image.fromarray(arr)
new_im.save('3.png')

#分离合并通道
r, g, b = img.split()
img = Image.merge("RGB", (b, g, r))

img = img.copy() #复制图像

img3 = Image.open('1.jpg')
roi = img3.crop((0,0,300,300)) #(左上x，左上y，右下x，右下y)坐标
roi.show()
