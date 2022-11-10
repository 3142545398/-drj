import cv2
import pydicom
import os
import numpy as np

# 使用一样的方法创建测试集和训练集
files_dir = 'Data/'
# files_dir_save_npy = 'train_npy/'
files_dir_save_npy = 'test_npy/'


if not os.path.exists(files_dir_save_npy):
    os.makedirs(files_dir_save_npy)

# 数据样本
list_sample_names = os.listdir(files_dir) # os.listdir(path)中有一个参数，就是传入相应的路径，将会返回那个目录下的所有文件名



# 训练集：
# for i in range(len(list_sample_names)-333):
# 测试集
for i in range(len(list_sample_names)-1000):
    i = i + 1000
    # if not os.path.exists(files_dir_save + list_sample_names[i]):
    #     os.makedirs(files_dir_save + list_sample_names[i])

    # 获取所有病例对应的文件目录
    sample_single_dir = files_dir + list_sample_names[i] + '/'
    # 获取该路径对应病例的所有影像（来自不同层）的目录
    list_pics = os.listdir(sample_single_dir)
    temp_cts = []

    pic_single_name_half = list_pics[0][:-9]
    # 遍历病例下的所有影像
    for j in range(len(list_pics)):
        temp_pic_name_ID = str(j+1)
        if len(temp_pic_name_ID)  == 3:
            temp_pic_name_ID = '00' + temp_pic_name_ID
        if len(temp_pic_name_ID)  == 2:
            temp_pic_name_ID = '000' + temp_pic_name_ID
        if len(temp_pic_name_ID)  == 1:
            temp_pic_name_ID = '0000' + temp_pic_name_ID
        if len(list_pics[0]) < 14:
            try:
                temp_ct = pydicom.read_file(sample_single_dir + 'image-' + str(j+1) + '.dcm')
            except Exception:
                print('Error')
        else:
            try:
                temp_ct = pydicom.read_file(sample_single_dir + pic_single_name_half + temp_pic_name_ID + '.dcm')
            except Exception:
                print('Error')
        temp_cts.append(temp_ct)

    # print('ok')

    # 把单个病例得的所有影像归为一个三维向量
    image_cts_np = np.zeros((len(temp_cts), 512, 512),dtype=np.int16)

    # print(temp_cts[0].ImageOrientationPatient[5])
    # if temp_cts[0].ImageOrientationPatient[5] != 0:
    #     print(list_sample_names[i])
    for k in range(len(temp_cts)):
        temp_image = temp_cts[k].pixel_array # 获取像素矩阵
        temp_image.astype(np.int16) # np类型的a如果直接修改如：a.dtype='int16'，那么直接会修改a，会导致长度的不一致，如果要直接修改则要采用astype方法如：b=a.astype('int16')，a保持不变，b的长度等于a，并且type由a变成了into6，或者调用b=np.array(a,dtype='int16'),效果和astype一样。另外b=np.array(a,dtype=np.int16)中的np.int16是一样的
                                    # float类型默认float64=float，int类型默认int64=int，uint类型默认uint64=uint。产生数组的时候如果指定，默认就是float64
        temp_image[temp_image < 0] = 0
        temp_image[temp_image > 4096] = 0
        intercept = temp_cts[k].RescaleIntercept
        slope = temp_cts[k].RescaleSlope

        # 读取Dicom图像数据与得到CT值图像（CT图）
        if slope != 1:
            temp_image = slope * temp_image.astype(np.float64)
            temp_image = temp_image.astype(np.int16)
        temp_image = temp_image + intercept

        image_cts_np[k,:,:] = temp_image
        # cv2.imshow('temp', temp_image)
        # cv2.waitKey()
    #     cv2.imwrite(files_dir_save + list_sample_names[i] + '/' + str(k) + '.png', temp_image)
    np.save(files_dir_save_npy + list_sample_names[i] + '.npy', image_cts_np)




    print(i, 'ok')
