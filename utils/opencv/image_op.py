import cv2
import numpy as np

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = cv2.imread('1.jpg')
cv2.imshow('src',img)
print(img.shape) # (h,w,c)
print(img.size) # 像素总数目
print(img.dtype)
print(img)
cv2.waitKey()

#gray = cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE) #灰度图
#cv2.imshow('gray',gray)
#也可以这么写，先读入彩色图，再转灰度图
src = cv2.imread('1.jpg')
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
print(gray.shape)
print(gray.size)
print(gray)
cv2.waitKey()

#注意，计算图片路径是错的，Opencv也不会提醒你，但print img时得到的结果是None
img2 = cv2.imread('2.jpg')
print(img2)

#如何解决“读到的图片不存在的问题”？ #加入判断语句，如果为空，做异常处理
img2 = cv2.imread('2.jpg')
if img2 == None:
    print('fail to load image!')

#注意到，opencv读入的图片的彩色图是一个channel last的三维矩阵（h,w,c），即（高度，宽度，通道）
#有时候在深度学习中用到的的图片矩阵形式可能是channel first，那我们可以这样转一下
print(img.shape)
img = img.transpose(2,0,1)
print(img.shape)

#有时候还要扩展维度，比如有时候我们需要预测单张图片，要在要加一列做图片的个数，可以这么做
img = np.expand_dims(img, axis=0)
print(img.shape)

data_list = [] 
loop:
    im = cv2.imread('xxx.png')
    data_list.append(im)
data_arr = np.array(data_list)

#因为opencv读入的图片矩阵数值是0到255，有时我们需要对其进行归一化为0~1
img3 = cv2.imread('1.jpg')
img3 = img3.astype("float") / 255.0  #注意需要先转化数据类型为float
print(img3.dtype)
print(img3)

#存储图片
cv2.imwrite('test1.jpg',img3) #得到的是全黑的图片，因为我们把它归一化了
#所以要得到可视化的图，需要先*255还原
img3 = img3 * 255
cv2.imwrite('test2.jpg',img3)  #这样就可以看到彩色原图了

#opencv读入的矩阵是BGR，如果想转为RGB，可以这么转
img4 = cv2.imread('1.jpg')
img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)

#访问像素
print(img4[10,10])  #3channels
print(gray[10,10]) #1channel
img4[10,10] = [255,255,255]
gray[10,10] = 255
print(img4[10,10])  #3channels
print(gray[10,10]) #1channel

#roi操作
roi = img4[200:550,100:450,:]
cv2.imshow('roi',roi)
cv2.waitKey()



