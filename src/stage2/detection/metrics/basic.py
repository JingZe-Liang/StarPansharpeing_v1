import numpy as np
import scipy.io as sio
import torch
from sklearn import metrics
from torch import Tensor


def cosin_similarity(x, y):
    norm_x = np.sqrt(np.sum(x**2, axis=-1))
    norm_y = np.sqrt(np.sum(y**2, axis=-1))
    x_y = np.sum(np.multiply(x, y), axis=-1)
    similarity = np.clip(x_y / (norm_x * norm_y), -1, 1)
    return np.arccos(similarity)


# borrow from https://github.com/shendb2022/HTD-IRN/tree/main


def ts_generation(data, gt, type=0):
    """
    using different methods to select the target spectrum
    :param type:
    0-均值光谱
    1-空间去偏（L2范数）后的均值光谱
    2-光谱去偏（光谱角）后的均值光谱
    3-空间中位光谱（L2范数的中位数）
    4-光谱中位光谱（光谱角的中位数）
    5-距离所有目标像素欧式距离最近的目标像素
    6-距离所有目标像素余弦距离最近的目标像素
    7-距离均值光谱欧式距离最近的目标像素
    8-距离均值光谱余弦距离最近的目标像素
    :return:
    """
    ind = np.where(gt == 1)
    ts = data[ind]
    avg_target_spectrum = np.mean(ts, axis=0)
    avg_target_spectrum = np.expand_dims(avg_target_spectrum, axis=-1)
    if type == 0:
        return avg_target_spectrum
    elif type == 1:
        spatial_distance = np.sqrt(np.sum((ts - avg_target_spectrum.T) ** 2, axis=-1))
        arg_distance = np.argsort(spatial_distance)
        saved_num = int(ts.shape[0] * 0.8)
        saved_spectrums = ts[arg_distance[:saved_num]]
        removed_deviation_target_spectrum = np.mean(saved_spectrums, axis=0)
        removed_deviation_target_spectrum = np.expand_dims(
            removed_deviation_target_spectrum, axis=-1
        )
        return removed_deviation_target_spectrum
    elif type == 2:
        spatial_distance = cosin_similarity(ts, avg_target_spectrum.T)
        arg_distance = np.argsort(spatial_distance)
        saved_num = int(ts.shape[0] * 0.8)
        saved_spectrums = ts[arg_distance[:saved_num]]
        removed_deviation_target_spectrum = np.mean(saved_spectrums, axis=0)
        removed_deviation_target_spectrum = np.expand_dims(
            removed_deviation_target_spectrum, axis=-1
        )
        return removed_deviation_target_spectrum
    elif type == 3:
        dist_list = np.zeros([ts.shape[0]])
        for i in range(ts.shape[0]):
            dist_list[i] = np.mean(np.sqrt(np.sum(np.square(ts - ts[i]), axis=-1)))
        arg_distance = np.argsort(dist_list)
        mid = ts.shape[0] // 2
        mid_target_spectrum = ts[arg_distance[mid]]
        mid_target_spectrum = np.expand_dims(mid_target_spectrum, axis=-1)
        return mid_target_spectrum
    elif type == 4:
        dist_list = np.zeros([ts.shape[0]])
        for i in range(ts.shape[0]):
            dist_list[i] = np.mean(cosin_similarity(ts, ts[i]))
        arg_distance = np.argsort(dist_list)
        mid = ts.shape[0] // 2
        mid_target_spectrum = ts[arg_distance[mid]]
        mid_target_spectrum = np.expand_dims(mid_target_spectrum, axis=-1)
        return mid_target_spectrum
    elif type == 5:
        min_distance = 10000
        opd_i = 0
        for i in range(ts.shape[0]):
            dist = np.mean(np.sqrt(np.sum(np.square(ts - ts[i]), axis=-1)))
            # print(dist)
            if dist < min_distance:
                min_distance = dist
                opd_i = i
        target_spectrum = ts[opd_i]
        target_spectrum = np.expand_dims(target_spectrum, axis=-1)
        return target_spectrum
    elif type == 6:
        min_distance = 10000
        opd_i = 0
        for i in range(ts.shape[0]):
            dist = np.mean(cosin_similarity(ts, ts[i]))
            # print(dist)
            if dist < min_distance:
                min_distance = dist
                opd_i = i
        target_spectrum = ts[opd_i]
        target_spectrum = np.expand_dims(target_spectrum, axis=-1)
        return target_spectrum

    elif type == 7:
        distance = np.sqrt(np.sum((ts - avg_target_spectrum.T) ** 2, axis=-1))
        arg_distance = np.argsort(distance)
        avg_L2_target_spectrum = ts[arg_distance[0]]
        avg_L2_target_spectrum = np.expand_dims(avg_L2_target_spectrum, axis=-1)
        return avg_L2_target_spectrum
    elif type == 8:
        distance = cosin_similarity(ts, avg_target_spectrum.T)
        # print(distance)
        arg_distance = np.argsort(distance)
        avg_cosin_target_spectrum = ts[arg_distance[0]]
        avg_cosin_target_spectrum = np.expand_dims(avg_cosin_target_spectrum, axis=-1)
        return avg_cosin_target_spectrum
    else:
        return avg_target_spectrum


def comput_AUC_scores(output: np.ndarray, target: np.ndarray):
    y_l = np.reshape(target, [-1, 1], order="F")
    y_p = np.reshape(output, [-1, 1], order="F")

    ## calculate the AUC value
    fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
    fpr = fpr[1:]
    tpr = tpr[1:]
    threshold = threshold[1:]
    auc1 = round(metrics.auc(fpr, tpr), 4)
    auc2 = round(metrics.auc(threshold, fpr), 4)
    auc3 = round(metrics.auc(threshold, tpr), 4)
    auc4 = round(auc1 + auc3 - auc2, 4)
    auc5 = round(auc3 / auc2, 4)

    return [auc1, auc2, auc3, auc4, auc5]


def intersection_and_union(output: Tensor, target: Tensor, K: int, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape

    output = output.view(-1)
    target = target.view(-1)

    output[target == ignore_index] = ignore_index
    intersection = output[output == target]  # output上分对的类别

    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(
        intersection.float().cpu(), bins=K, min=0, max=K - 1
    )  # output上分对的类别中每类的个数
    area_output = torch.histc(
        output.float().cpu(), bins=K, min=0, max=K - 1
    )  # output每类的个数
    area_target = torch.histc(
        target.float().cpu(), bins=K, min=0, max=K - 1
    )  # target每类的个数
    area_union = area_output + area_target - area_intersection

    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()
