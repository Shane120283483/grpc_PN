"""
深度学习算法分类器
"""
import os
import zipfile
import SimpleITK as sitk
import cv2
import requests
from zipfile import ZipFile
import numpy as np
import torch
import torch.nn as nn
import shutil
from matplotlib import pyplot as plt
import scipy
import scipy.ndimage

fmap_block = list()
grad_block = list()


def unzip_dcm(cur_dcm_url):
    cur_r = requests.get(cur_dcm_url, stream=True)  # 获取url对应zip数据
    # print(cur_r.status_code)
    if cur_r.status_code == 404:  # url无效
        return None
    os.makedirs("unzip_data", exist_ok=True)
    cur_zip_path = "unzip_data/cur.zip"
    with open(cur_zip_path, 'wb') as f:  # 将zip数据写到本地保存
        f.write(cur_r.content)
    out_dir_path = "unzip_data/cur_series"
    if os.path.exists(out_dir_path):
        shutil.rmtree(out_dir_path)  # 先移除清空series目录
    os.makedirs(out_dir_path, exist_ok=True)  # 然后重新创建目录
    # with ZipFile(cur_zip_path, 'r') as zfile:
    #     nl = zfile.namelist()
    #     for id, l in enumerate(nl):
    #         zfile.extract(l, path=out_dir_path)
    if not zipfile.is_zipfile(cur_zip_path):  # url对应文件不是zip
        return None
    with ZipFile(cur_zip_path, 'r') as zfile:  # 解压zip
        zfile.extractall(out_dir_path)
    dcm_dir_path = ""
    for root, dirs, files in os.walk(out_dir_path):  # 找到.dcm的父目录
        for d in dirs:
            if os.listdir(os.path.join(root, d))[0].endswith('.dcm'):
                dcm_dir_path = os.path.join(root, d)
                break
    return dcm_dir_path


def dcm2nii(dcms_path, nii_path):
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)


def read_img(path, itkResize=(69, 95, 79)):
    img = sitk.ReadImage(path)
    imgResampled = resize_image_itk(img, itkResize)  # resize数据, resamplemethod= sitk.sitkLinear
    data = sitk.GetArrayFromImage(imgResampled)
    data = data[np.newaxis, np.newaxis, :, :, :]  # 对应于权重weight的5维，因为batch还有1维
    return data


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


def predict(data_path, model_path="models/best_6acc.pth"):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MobileNet().to(device)
    # 加载已保存模型
    model.load_state_dict(torch.load(model_path), False)
    # model = torch.load(model_path)  # 加载已保存模型
    model = model.eval()

    data = read_img(data_path)
    data = np.array(data, 'float32')
    data = (data - min(data)) / (max(data) - min(data))  # 输入归一化
    data -= data.mean()
    data /= data.std()  # 输入标准化

    data = torch.FloatTensor(data).to(device)
    output = model(data)  # 输出标准化
    mean = output.mean()
    std = output.std()
    output = (output - mean) / std
    # print(output)
    pred = torch.argmax(output, 1)
    print(pred)
    return pred[0].cpu().numpy()  # N 0 P 1


class MobileNet(nn.Module):
    """
    MobileNetv1
    """

    def __init__(self, num_classes=2):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes

        def conv_bn(inp, oup, stride):
            """
            inp:输入通道数
            oup：输出通道数
            """
            return nn.Sequential(
                nn.Conv3d(inp, oup, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):  # 深度可分离卷积
            """
            inp:输入通道数
            oup：输出通道数
            """
            return nn.Sequential(
                nn.Conv3d(inp, inp, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                nn.ReLU(inplace=True),

                nn.Conv3d(inp, oup, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm3d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            #             conv_bn(3, 32, 2),
            conv_bn(1, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            #             nn.AvgPool3d(7),
            nn.AvgPool3d(3),
        )
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def backward_hook(module, grad_in, grad_out):
    """
    获取梯度
    :param module:
    :param grad_in:
    :param grad_out:
    :return:
    """
    # 在backward_hook函数中，grad_out是一个tuple类型的，要取得特征图的梯度需要这样grad_block.append(grad_out[0].detach())
    grad_block.append(grad_out[0].detach())


def forward_hook(module, input, output):
    """
    获取feature map
    :param module:
    :param input:
    :param output:
    :return:
    """
    fmap_block.append(output)


def gen_cam(feature_map, grads, size=(69, 95, 79)):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam初始化
    weight = np.mean(grads, axis=(1, 2, 3))  # 梯度均值作为全局权重
    for i, w in enumerate(weight):
        cam += w * feature_map[i, :, :, :]  # i指示feature map的第i个通道和weight的第i个w
    cam = np.maximum(cam, 0)  # RELU
    # cam = scipy.ndimage.zoom(cam, (69/cam.shape[0], 95/cam.shape[1], 79/cam.shape[2]), order=3)  # 放大
    cam = scipy.ndimage.zoom(cam, (size[0]/cam.shape[0], size[1]/cam.shape[1], size[2]/cam.shape[2]), order=3)  # 放大
    # cam -= np.min(cam)
    # cam /= np.max(cam)  # cam归一化
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # 归一化是否应该放在放大之前？
    return cam


def gen_layer_cam(feature_map, grads, size=(69, 95, 79)):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam初始化
    grads = np.maximum(grads, 0)  # RELU
    for i, w in enumerate(grads):
        feature_map[i, :, :, :] = w * feature_map[i, :, :, :]
        cam += feature_map[i, :, :, :]
    cam = np.maximum(cam, 0)  # RELU
    cam = scipy.ndimage.zoom(cam, (size[0] / cam.shape[0], size[1] / cam.shape[1], size[2] / cam.shape[2]),
                             order=3)  # 放大
    cam -= np.min(cam)
    cam /= (np.max(cam) - np.min(cam))  # cam归一化
    return cam


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor([[a,  b]], grad_fn=<AddmmBackward>)
    :param index: int, 指定类别, 0/1对应N/P
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
        print("index:", index)
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]  # 扩充2维
    print("index:", index)
    index = torch.from_numpy(index)
    print("index:", index)
    one_hot = torch.zeros(1, 2).scatter_(1, index, 1)  # scatter_(dim, index, src)
    # one_hot = torch.zeros(1, 3).scatter_(1, index, 1)  # scatter_(dim, index, src)
    print("one_hot:", one_hot)
    one_hot.requires_grad = True  # 保留梯度信息
    class_vec = torch.sum(one_hot * ouput_vec)
    print("class_vec:", class_vec)
    return class_vec


def show_CAM(image_dir, model_path="models/best_6acc.pth"):
    classes = ('Normal', 'PD')
    data = read_img(image_dir)
    data = np.array(data, 'float32')
    data -= data.mean()
    data /= data.std()  # 输入标准化
    device = torch.device("cpu")
    model = MobileNet().to(device)
    model.load_state_dict(torch.load(model_path), False)
    model = model.eval()
    # 注册hook
    # model.conv3c.register_forward_hook(
    #     forward_hook)  # 在网络执行forward()后，执行forward_hook函数，此函数必须具有hook(module, input, output)形式
    # model.conv3c.register_backward_hook(backward_hook)  # 在网络执行backward()后，执行backward_hook函数
    model.model[-2][-3].register_forward_hook(forward_hook)  # TODO MobileNet的CAM离谱
    model.model[-2][-3].register_backward_hook(backward_hook)  # 在网络执行backward()后，执行backward_hook函数
    data = torch.FloatTensor(data).to(device)
    output = model(data)  # 执行前向传播
    print(output)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    model.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()  # 这里反向传播获得梯度，触发backward_hook

    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    # cam = gen_layer_cam(fmap, grads_val)
    cam = gen_cam(fmap, grads_val)

    # CAM直接保存为nii
    # img = sitk.GetImageFromArray(cam)
    # img = sitk.SaveImage(path)/////////////////////////////TODO

    # # CAM的二维显示
    # for i in range(cam.shape[0]):
    #     img = sitk.ReadImage(image_dir)
    #     data = sitk.GetArrayFromImage(img)
    #     plt.subplot(10, 14, i + 1)
    #     plt.imshow(data[i, :, :], cmap='gray')
    #     plt.xticks([])
    #     plt.yticks([])
    #     heatmap = cv2.applyColorMap(np.uint8(255 * cam[i, :, :]), cv2.COLORMAP_JET)  # BGR
    #     plt.subplot(10, 14, i + 71)
    #     plt.imshow(heatmap[:, :, [2, 1, 0]])  # RGB
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     result_dir = "CAM_result/LayerCAM/" + model_path.split("/")[-1][:-4]
    #     os.makedirs(result_dir + "/" + 'train/P', exist_ok=True)
    #     os.makedirs(result_dir + "/" + 'train/N', exist_ok=True)
    #     os.makedirs(result_dir + "/" + 'test/P', exist_ok=True)
    #     os.makedirs(result_dir + "/" + 'test/N', exist_ok=True)
    # plt.savefig(result_dir+"/"+"{}/{}.png".format(image_dir.split("/")[-3]+"/"+image_dir.split("/")[-2], image_dir.split("/")[-1][:-4]))
    # # plt.show()
    return idx, cam
