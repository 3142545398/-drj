
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# 图像显示中文的问题
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

import seaborn as sns
sns.set(font="Kaiti", style="ticks", font_scale=1.4)


dcm = pydicom.read_file("Data/A0D3BE0B-0CA5-4E02-A94E-8F8A765EB0CF/8aebbf39-e6ef-47be-8cd0-513b7edfc58f_00001.dcm")
print("\nDICOM具体内容如下所示：")
print(dcm)

# 打印矩阵大小
temp_image = dcm.pixel_array
temp_image.astype(np.int16)
print("\n图像大小为：")
print(temp_image.shape)


# 获取像素点个数
lens = temp_image.shape[0]*temp_image.shape[1]
# 获取像素点的最大值和最小值
arr_temp = np.reshape(temp_image,(lens,))
max_val = max(arr_temp)
min_val = min(arr_temp)
# 图像归一化
temp_image = (temp_image-min_val)/(max_val-min_val)
# 绘制图像并保存
plt.figure(figsize=(12,12),dpi=250) # 图像大小可以根据文件不同进行设置
plt.imshow(temp_image,cmap=plt.cm.bone)
plt.title("肝部")
plt.show()
















# # from torch import torchvision.models.resnet50
#
# import os
# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# import csv
# import models_3d as mm
# import torch.nn as nn
# import random
# import matplotlib.pyplot as plt
# import cv2
# import scipy.ndimage
#
# from sklearn.metrics import f1_score
#
# # 使用GPU
# device = torch.device('cuda')
# print(torch.__version__)  # 1.11.0 + cu115
#
# # 截取中央部分的样本块
# def get_crop_from_np(img_np, out_size):
#     d, h, w = img_np.shape
#     new_d, new_h, new_w = out_size
#
#     front = int((d-new_d)/2)
#     top = np.random.randint(0, h-new_h)
#     left = np.random.randint(0, w-new_w)
#
#     img_np_crop = img_np[front:front+new_d, top:top+new_h, left:left+new_w]
#
#     return img_np_crop
#
# # 截图z轴的中央部分，x和y轴随机截取，组成一个样本块
# def get_crop_from_np_ct(img_np, out_size):
#     d, h, w = img_np.shape
#     new_d, new_h, new_w = out_size
#
#     front = int((d-new_d)/2)
#     top = int((h-new_h)/2)
#     left = int((w-new_w)/2)
#
#     img_np_crop_ct = img_np[front:front+new_d, top:top+new_h, left:left+new_w]
#
#     return img_np_crop_ct
#
#
# # 读取转换保存好的npy数据，恢复成numpy三维数组
# def get_np_from_npy(file_dir, flag_crop=True):
#     image_np = np.load(file_dir)
#     image_np = image_np.astype(np.uint8)
#     MIN_BOUND = 0
#     MAX_BOUND = 200
#     image_np[image_np < MIN_BOUND] = 0
#     image_np[image_np > MAX_BOUND] = 0
#
#     flag_flip = random.randint(-1, 2)
#
#     #print("原本大小")
#     image_np.shape
#     #print(image_np.shape)
#     #print("--------------")
#     for i in range(image_np.shape[0]):
#         img_temp = image_np[i, :, :]
#         img_temp = cv2.equalizeHist(img_temp)# 利用均衡化进行对比度拉伸
#         if flag_crop:
#             if flag_flip < 2:
#                 img_temp = cv2.flip(img_temp, flag_flip)# 随机进行镜像翻转
#         image_np[i, :, :] = img_temp
#
#         f_scale_0 = float(128) / float(image_np.shape[0])
#         f_scale_1 = float(150) / float(image_np.shape[1])
#     if flag_crop:
#         image_np = scipy.ndimage.zoom(input=image_np, zoom=[f_scale_0, f_scale_1, f_scale_1])#更改尺寸
#         image_np_ex = get_crop_from_np(img_np=image_np, out_size=(64, 128, 128))
#
#     else:
#         image_np = scipy.ndimage.zoom(input=image_np, zoom=[f_scale_0, f_scale_1, f_scale_1])
#         image_np_ex = get_crop_from_np_ct(img_np=image_np, out_size=(64, 128, 128))
#
#     return image_np_ex
#
#
#
#
#
#
# # 对之前得到的三维数组进行操作
# img_np = get_np_from_npy("TrainOrigin_temp_npy/A0D43175-2EC7-4655-974C-C7F79CDE533B.npy")
# print(img_np.shape)
# # 处理图像数据
# img_tensor = torch.Tensor(img_np) # 深拷贝（改变原有数据时使用）
# print("---------------------")
# print(img_tensor.shape)
# img_tensor = torch.stack([img_tensor], 0)
# print("---------------------")
# print(img_tensor.shape)
#
#
#
#
#
# # 样本类，得到每次取样的样本张量、标签张量
# class my_dataset():
#     def __init__(self, list_filenames, data_dir, dict_label, flag_crop=True):
#         self.list_filenames = list_filenames
#         self.data_dir = data_dir
#         self.dict = dict_label
#         self.flag_crop = flag_crop
#
#     def __getitem__(self, item):
#         img_name = self.list_filenames
#         # 对之前得到的三维数组进行操作
#         img_np = get_np_from_npy(self.data_dir + img_name, flag_crop=self.flag_crop)
#
#         # 处理图像数据
#         img_tensor = torch.Tensor(img_np) # 深拷贝（改变原有数据时使用）
#         img_tensor = torch.stack([img_tensor], 0)
#
#         label = "1"
#         # 处理图像标签
#         label_tensor = torch.from_numpy(np.array([int(label)], dtype=np.float32))  # 浅拷贝（不改变原有数据时使用）
#                                                                                    # 从numpy.ndarray创建一个张量
#         label_tensor = label_tensor.long()
#
#         return img_tensor, label_tensor
#
#     def __len__(self):
#         return len(self.list_filenames)
#
# train_data = my_dataset(list_filenames="A01CDD9E-7257-4167-AB8B-7FA54771BBFE.npy", data_dir="TrainOrigin_temp_npy/", dict_label="1",
#                             flag_crop=True)
#
# train_data_loader = DataLoader(train_data, shuffle=True, batch_size=2, num_workers=0)
# print(len(train_data_loader))# 20
# count = 0
# for i, (img, labels) in enumerate(train_data_loader):
#     print( img.shape)
#     print( labels)
#     count += 1
#
# print(count)# 20




# if not os.path.exists(files_dir_save_npy):
#     os.makedirs(files_dir_save_npy)
#
# # 数据样本
# list_sample_names = os.listdir(files_dir) # os.listdir(path)中有一个参数，就是传入相应的路径，将会返回那个目录下的所有文件名
#
#
#
# # 训练集：
# # for i in range(len(list_sample_names)-333):
# # 测试集
# for i in range(len(list_sample_names)-1000):
#     i = i + 1000
#     # if not os.path.exists(files_dir_save + list_sample_names[i]):
#     #     os.makedirs(files_dir_save + list_sample_names[i])
#
#     sample_single_dir = files_dir + list_sample_names[i] + '/'
#     print(sample_single_dir)
#     # 获取该路径对应病例的所有影像（来自不同层）
#     list_pics = os.listdir(sample_single_dir)
#     temp_cts = []
#
#     pic_single_name_half = list_pics[0][:-9]
#    print(pic_single_name_half)


