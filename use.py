'''
模型的使用（训练、评估和预测）
'''

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv
import model as mm
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage

from sklearn.metrics import f1_score

# 使用GPU
device = torch.device('cuda')
print(torch.__version__)  # 1.11.0 + cu115


# ==================== 数据预处理阶段所需函数 ====================
# 截取方式一
# 截取中央部分的样本块
def cut_1(img_np, out_size):
    d, h, w = img_np.shape
    new_d, new_h, new_w = out_size

    front = int((d-new_d)/2)
    top = np.random.randint(0, h-new_h)
    left = np.random.randint(0, w-new_w)

    img_np_crop = img_np[front:front+new_d, top:top+new_h, left:left+new_w]

    return img_np_crop

# 截取方式二
# 截图z轴的中央部分，x和y轴随机截取，组成一个样本块
def cut_2(img_np, out_size):
    d, h, w = img_np.shape
    new_d, new_h, new_w = out_size

    front = int((d-new_d)/2)
    top = int((h-new_h)/2)
    left = int((w-new_w)/2)

    img_np_crop_ct = img_np[front:front+new_d, top:top+new_h, left:left+new_w]

    return img_np_crop_ct


# 读取转换保存好的npy数据，恢复成维度一致的三维数组
def numpy_to_numpy(file_dir, flag_crop=True):
    '''
    初始的三维数组：（k,512,512）  -k:单个病例下所对应的原始图像个数（每个病例几乎均不同）
    转化后的三维数组：（64,128,128）
    '''
    image_np = np.load(file_dir) # k * 512 * 512
    image_np = image_np.astype(np.uint8)
    MIN_BOUND = 0
    MAX_BOUND = 200
    image_np[image_np < MIN_BOUND] = 0
    image_np[image_np > MAX_BOUND] = 0

    flag_flip = random.randint(-1, 2)

    # 对单个病例下所有图像逐一进行如下操作
    for i in range(image_np.shape[0]):
        img_temp = image_np[i, :, :] # 512 * 512
        img_temp = cv2.equalizeHist(img_temp)# 利用均衡化进行对比度拉伸
        if flag_crop:
            if flag_flip < 2: # 1/4的概率进行镜像翻转
                img_temp = cv2.flip(img_temp, flag_flip)# 随机进行镜像翻转
        image_np[i, :, :] = img_temp

        f_scale_0 = float(128) / float(image_np.shape[0])
        f_scale_1 = float(150) / float(image_np.shape[1])


    if flag_crop: # 3/4的概率使用截取方式一
        image_np = scipy.ndimage.zoom(input=image_np, zoom=[f_scale_0, f_scale_1, f_scale_1]) # 一样，进行值的缩放
        image_np_ex = cut_1(img_np=image_np, out_size=(64, 128, 128)) # 更改尺寸

    else: # 1/4的概率使用截取方式一
        image_np = scipy.ndimage.zoom(input=image_np, zoom=[f_scale_0, f_scale_1, f_scale_1]) # 一样，进行值的缩放
        image_np_ex = cut_2(img_np=image_np, out_size=(64, 128, 128)) # 更改尺寸

    return image_np_ex


# 样本类，得到每次取样的样本张量、标签张量
class my_dataset():
    def __init__(self, list_filenames, data_dir, dict_label, flag_crop=True):
        self.list_filenames = list_filenames
        self.data_dir = data_dir
        self.dict = dict_label
        self.flag_crop = flag_crop

    def __getitem__(self, item):
        img_name = self.list_filenames[item]
        # 对之前得到的三维数组进行操作
        img_np = numpy_to_numpy(self.data_dir + img_name, flag_crop=self.flag_crop)

        # 处理图像数据
        img_tensor = torch.Tensor(img_np) # 深拷贝（改变原有数据时使用）
        img_tensor = torch.stack([img_tensor], 0)

        # 处理图像标签
        label = self.dict[img_name[:-4]]
        label_tensor = torch.from_numpy(np.array([int(label)], dtype=np.float32))  # 浅拷贝（不改变原有数据时使用）
                                                                                   # 从numpy.ndarray创建一个张量
        label_tensor = label_tensor.long()

        return img_tensor, label_tensor

    def __len__(self):
        return len(self.list_filenames)


# 整数转化为张量
def int_to_tensor(x,device):
    tensor = torch.from_numpy(np.array([int(x)], dtype=np.float32)) # 从numpy.ndarray创建一个张量，浅复制
    tensor = tensor.long().to(device)
    return tensor


# ==================== 模型评价阶段所需函数 ====================
# 模型评价指标一（样本平衡）
# 计算验证集合的正确率（准确率Accuracy）
def get_Accuracy(data_loader, model):
    correct = 0
    total = 0
    model.eval() # 开启模型评估模式
    for i,(img, labels) in enumerate(data_loader):
        img = img.to(device)
        labels = labels.to(device)
        outputs = model(img)
        labels = labels.squeeze_()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * float(correct) / float(total)

    return accuracy

# 模型评价指标二（样本不平衡）
# 计算验证集合的得分((分类0的F1-Score+分类1的F1-Score)/2)
def get_Score(data_loader, model):
    model.eval()
    TP_0 = 0
    TN_0 = 0
    FN_0 = 0
    FP_0 = 0
    TP_1 = 0
    TN_1 = 0
    FN_1 = 0
    FP_1 = 0
    precision_0 = 0
    precision_1 =0
    recal_0 = 0
    recal_1 = 0
    F1_0 = 0
    F1_1 =0
    Score = 0
    acc = 0
    model.eval()  # 开启模型评估模式
    for i, (img, labels) in enumerate(data_loader):
        img = img.to(device)
        labels = labels.to(device)
        outputs = model(img)
        labels = labels.squeeze_()
        _, predicted = torch.max(outputs.data, 1)

        for i in range(2):
            # 分类0的相关指标(此时理解正样本为0)
            # TP    predict 和 label 同时为0
            TP_0 += ((predicted[i] == int_to_tensor(0, device)) & (labels[i] == int_to_tensor(0, device))).sum().item()
            # TN    predict 和 label 同时为1
            TN_0 += ((predicted[i] == int_to_tensor(1, device)) & (labels[i] == int_to_tensor(1, device))).sum().item()
            # FN    predict 1 label 0
            FN_0 += ((predicted[i] == int_to_tensor(1, device)) & (labels[i] == int_to_tensor(0, device))).sum().item()
            # FP    predict 0 label 1
            FP_0 += ((predicted[i] == int_to_tensor(0, device)) & (labels[i] == int_to_tensor(1, device))).sum().item()

            # 分类1的相关指标(此时理解正样本为1)
            TP_1 += ((predicted[i] == int_to_tensor(1, device)) & (labels[i] == int_to_tensor(1, device))).sum().item()
            # TN    predict 和 label 同时为0
            TN_1 += ((predicted[i] == int_to_tensor(0, device)) & (labels[i] == int_to_tensor(0, device))).sum().item()
            # FN    predict 0 label 1
            FN_1 += ((predicted[i] == int_to_tensor(0, device)) & (labels[i] == int_to_tensor(1, device))).sum().item()
            # FP    predict 1 label 0
            FP_1 += ((predicted[i] == int_to_tensor(1, device)) & (labels[i] == int_to_tensor(0, device))).sum().item()

    precision_0 = TP_0 / (TP_0 + FP_0)
    precision_1 = TP_1 / (TP_1 + FP_1)
    recal_0 = TP_0 / (TP_0 + FN_0)
    recal_1 = TP_1 / (TP_1 + FN_1)
    F1_0 = 2 * recal_0 * precision_0 / (recal_0 + precision_0)
    F1_1 = 2 * recal_1 * precision_1 / (recal_1 + precision_1)
    Score = 100 * ((F1_0 + F1_1)/2)
    acc = 100 * ((TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1))
    print('Acceracy(使用公式):{:.2f}% '.format(acc, 0))
    return Score




if __name__=='__main__':
    import time
    time_start = time.time()  # 开始计时

    #  确定是否进行训练流程
    train_flag = True  # True-表示进行训练，False-表示根据训练保存模型进行前向预测

    # 指定训练集合路径
    data_dir = 'train_npy/'
    list_file_names = os.listdir(data_dir)

    # 指定模型保存路径
    model_dir = 'model.pb'

    # 设置训练样本标签
    label_dir = 'train_label.csv'
    list_label = csv.reader(open(label_dir, 'r'))
    dict_label = {'0': '0'}
    for row in list_label:
        dict_label[row[0]] = row[1]

    # 将数据集划分为训练集和验证集
    num_train = int(len(list_file_names) * 0.7)  # 指定训练集和验证集的占比
    random.shuffle(list_file_names)
    random.shuffle(list_file_names)
    list_file_names_train = []
    list_file_names_test = []
    list_file_names_train = list_file_names[:num_train]
    list_file_names_test = list_file_names[num_train:]

    # 24
    # 可以从1变大
    n_batch_size = 2 # 设置块大小
    n_show_step = 50

    # windows没法跑多线程
    train_data = my_dataset(list_filenames=list_file_names_train, data_dir=data_dir, dict_label=dict_label,
                            flag_crop=True)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=n_batch_size, num_workers=0)

    test_data = my_dataset(list_filenames=list_file_names_test, data_dir=data_dir, dict_label=dict_label,
                           flag_crop=False)
    test_data_loader = DataLoader(test_data, shuffle=True, batch_size=n_batch_size, num_workers=0)

    data_all = my_dataset(list_filenames=list_file_names, data_dir=data_dir, dict_label=dict_label, flag_crop=True)
    data_all_loader = DataLoader(data_all, shuffle=True, batch_size=n_batch_size, num_workers=0)

    # 设置迭代次数
    epoch_max = 5



    # 设置模型
    # 第一种
    # model = mm.C3D_Simple() # 只能单个，无法汇总

    # 第二种
    # model = mm.C3D_ResNet_simple() # 每次迭代都一样

    # 第三种
    model = mm.ResNet(mm.BasicBlock, [2, 2, 2, 2])  # 采用ResNet-18的结构
    # 大于18层的网络就进行不下去了
    # model = mm.ResNet(mm.Bottleneck, [3, 4, 6, 3])  # 采用ResNet-50的结构

    list_loss = []
    list_aceracy = []
    list_score = []
    list_rate_submit = []

    if train_flag:
        # 执行训练过程
        model = nn.DataParallel(model, device_ids=[0]) # DataParallel（）函数的作用就是将一个batchsize的输入数据均分到多个GPU上分别计算
                                                       # device_ids只能有一个参数：因为本地台式机只有一个GPU，如果调用数量超出会报错
        model.to(device) # 表示将模型转移到指定设备

        # 损失
        criterion = nn.CrossEntropyLoss() # nn.CrossEntropyLoss()为交叉熵损失函数，用于解决多分类问题，也可用于解决二分类问题。在使用nn.CrossEntropyLoss()其内部会自动加上Sofrmax层
        # 优化
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # optimizer用来保存当前的状态，并能够根据计算得到的梯度来更新参数

        total_step_train = len(train_data_loader)
        total_step_all = len(data_all_loader)
        for epoch in range(epoch_max):
            if epoch == 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
            if epoch < 100:
                for i, (img, labels) in enumerate(train_data_loader):
                    model.train() # 开启训练模式
                    # 转移到该设备上
                    img = img.to(device)
                    labels = labels.to(device)
                    # 得到预测值
                    output = model(img)
                    labels = labels.squeeze_()
                    # 计算损失
                    loss_end = criterion(output, labels)
                    # 优化
                    optimizer.zero_grad()
                    loss = loss_end
                    # 借助后向进行梯度下降法
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % n_show_step == 0:
                        list_loss.append(loss.item())
                        print('Epoch[{}/{}], Step[{}/{}] Loss:{:.7f}'.format(epoch + 1, epoch_max, i + 1,
                                                                             total_step_train, loss.item()))
            else:
                for i, (img, labels) in enumerate(data_all_loader):
                    model.train()
                    img = img.to(device)
                    labels = labels.to(device)
                    output = model(img)
                    labels = labels.squeeze_()
                    loss_end = criterion(output, labels)

                    optimizer.zero_grad()
                    loss = loss_end
                    loss.backward()
                    optimizer.step()

                    # 每处理10个块显示一次
                    if (i + 1) % n_show_step == 0:
                        list_loss.append(loss.item())
                        print(
                            'Epoch[{}/{}], Step[{}/{}] Loss:{:.7f}'.format(epoch + 1, epoch_max, i + 1, total_step_all,
                                                                           loss.item()))
            # 正确率(Acceracy)
            aceracy_temp = get_Accuracy(test_data_loader, model)
            list_aceracy.append(aceracy_temp)
            print('Acceracy(使用定义):{:.2f}% '.format(aceracy_temp, 0))

            # 得分(Score)
            score_temp = get_Score(test_data_loader,model)
            list_score.append(score_temp)
            print('Score:{:.2f}% '.format(score_temp, 0))



        # 绘制损失变化图
        fig_1 = plt.figure(1)
        line = plt.plot(list_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.ylim(top=1)
        plt.savefig('./Loss.png')
        # plt.show()
        # plt.close()


        # 绘制准确率(Acceracy)变化图
        fig_2 = plt.figure(2)
        line = plt.plot(list_aceracy)
        # line_2 = plt.plot(list_rate_submit)
        plt.xlabel('epoch')
        plt.ylabel('aceracy value')
        plt.ylim(top=100)
        plt.savefig('./Acceracy.png')
        plt.show()
        plt.close()

        # 绘制得分(Score)变化图
        fig_3 = plt.figure(3)
        line = plt.plot(list_score)
        # line_2 = plt.plot(list_rate_submit)
        plt.xlabel('epoch')
        plt.ylabel('score value')
        plt.ylim(top=100)
        plt.savefig('./Score.png')
        plt.show()
        plt.close()

        model.eval()
        torch.save(model.state_dict(), model_dir)

    else:
        # 执行测试过程
        model = nn.DataParallel(model, device_ids=[0]) # device_ids=[0, 1] : 而本地台式机只有一个GPU，调用数量超出所以报错
        model.load_state_dict(torch.load(model_dir))
        model.eval().to(device)

        # 加载测试集
        data_dir_test = 'test_npy/'
        list_file_names = os.listdir(data_dir_test)
        import datetime

        # 设置保存结果的路径
        date = datetime.datetime.now()
        result_dir = './result_' + str(date.year) + '-' + str(date.month) + '-' + str(date.day) + '-' + str(
            date.hour) + '.csv'
        ret_file = open(result_dir, 'a', newline='')
        ret_write = csv.writer(ret_file, dialect='excel')

        # 预测
        for i in range(len(list_file_names)):
            list_temp_ret = []
            test_data_tensor = torch.Tensor(numpy_to_numpy(data_dir_test + list_file_names[i], flag_crop=False))
            test_data_tensor = torch.stack([test_data_tensor], 0).to(device)
            test_data_tensor = torch.stack([test_data_tensor], 0).to(device)

            # 预测结果
            out = model(test_data_tensor)
            _, value_predict = torch.max(out.data, 1)

            # 写入保存的结果路径下
            list_temp_ret.append(list_file_names[i][:-4])
            list_temp_ret.append(int(value_predict))
            print(i, len(list_file_names))
            ret_write.writerow(list_temp_ret)

    time_end = time.time()  # 结束计时

    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')




# import csv
# import numpy as np
# file_types = ['ret',]
# original_lable = []
# predit_lable = []
#
# with open('test_label.csv') as f:
#     original_file = csv.reader(f)
#     # 循环获取每一行的内容
#     for row in original_file:
#         original_lable.append(int(row[1]))
#
# with open('ret_2022-10-30-1.csv') as f:
#     predit_file = csv.reader(f)
#     # 循环获取每一行的内容
#     for row in predit_file:
#         predit_lable.append(int(row[1]))
#
# count = 0
# for i in range(len(predit_lable)):
#     if original_lable[i] == predit_lable[i]:
#         count +=1
# print('正确率为：{:.2%}' .format(count/len(predit_lable)))

