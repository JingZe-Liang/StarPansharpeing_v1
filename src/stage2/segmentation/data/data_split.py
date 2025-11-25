import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from src.utilities.io import read_image
from src.utilities.logging import log

# !!! HyperSIGMA use semantic segmentation way to classification/segmentation
# !!! but they did not patching the image as the usual setting as in segmentation,
# !!! since there is only one fxxking image !
# So they just per-class-sampling pixel of GTs and corresponding pixels
# This will CAUSE HEAVILY SPATIAL INFOMATION LEACKAGE !!!
# But anyway since this maybe a usual setting and handling in this field,
# I prefer not to make it changed.
# Please check test function 'test_seg_data_split' in this file.


def single_image_split_dataset(
    gt_flatten: Float[np.ndarray, "h*w"],
    class_num: int,
    train_ratio: float = 0.99,
    val_ratio: float = 0.01,
    train_num: int = 10,
    val_num: int = 5,
    samples_type: str = "ratio",
):
    train_index = []
    test_index = []
    val_index = []
    if samples_type == "ratio":
        for i in range(class_num):
            idx = np.where(gt_flatten == i + 1)[-1]
            sample_len = len(idx)
            train_num = np.ceil(sample_len * train_ratio).astype("int32")
            val_num = np.ceil(sample_len * val_ratio).astype("int32")
            np.random.shuffle(idx)
            train_index.append(idx[:train_num])
            val_index.append(idx[train_num : train_num + val_num])
            test_index.append(idx[train_num + val_num :])
    else:
        sample_num = train_num
        for i in range(class_num):
            idx = np.where(gt_flatten == i + 1)[-1]
            sample_len = len(idx)
            max_index = np.max(sample_len) + 1
            np.random.shuffle(idx)
            if sample_num > max_index:
                sample_num = 10
            else:
                sample_num = train_num

            train_index.append(idx[:sample_num])
            val_index.append(idx[sample_num : sample_num + val_num])
            test_index.append(idx[sample_num + val_num :])

    train_index = np.concatenate(train_index, axis=0).astype("int32")
    val_index = np.concatenate(val_index, axis=0).astype("int32")
    test_index = np.concatenate(test_index, axis=0).astype("int32")

    return train_index, val_index, test_index


def oversample_weak_classes(X, y):
    uniq_cls, cls_count = np.unique(y, return_counts=True)
    label_inverser_r = np.max(cls_count) / cls_count

    # repeat for every label and concat
    newX = X[y == uniq_cls[0], :, :, :].repeat(round(label_inverser_r[0]), axis=0)
    newY = y[y == uniq_cls[0]].repeat(round(label_inverser_r[0]), axis=0)

    for label, ri in zip(uniq_cls[1:], label_inverser_r[1:]):
        cX = X[y == label, :, :, :].repeat(round(ri), axis=0)
        cY = y[y == label].repeat(round(ri), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))

    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]

    return newX, newY


# * --- Label related --- #


def single_image_get_seg_label(gt_flatten: Float[np.ndarray, "h*w"], train_index, val_index, test_index):
    train_samples_gt = np.zeros(gt_flatten.shape)
    for i in range(len(train_index)):
        train_samples_gt[train_index[i]] = gt_flatten[train_index[i]]

    test_samples_gt = np.zeros(gt_flatten.shape)
    for i in range(len(test_index)):
        test_samples_gt[test_index[i]] = gt_flatten[test_index[i]]

    val_samples_gt = np.zeros(gt_flatten.shape)
    for i in range(len(val_index)):
        val_samples_gt[val_index[i]] = gt_flatten[val_index[i]]

    return train_samples_gt, test_samples_gt, val_samples_gt


def single_image_get_label_mask(
    train_samples_gt,
    test_samples_gt,
    val_samples_gt,
    data_gt: Float[np.ndarray, "h w"],
    class_num: int,
):
    height, width = data_gt.shape
    # train
    train_label_mask = np.zeros([height * width, class_num], dtype=np.float32)
    temp_ones = np.ones([class_num], dtype=np.float32)
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [height * width, class_num])

    # test
    test_label_mask = np.zeros([height * width, class_num], dtype=np.float32)
    temp_ones = np.ones([class_num], dtype=np.float32)
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [height * width, class_num])

    # val
    val_label_mask = np.zeros([height * width, class_num], dtype=np.float32)
    temp_ones = np.ones([class_num], dtype=np.float32)
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [height * width, class_num])

    return train_label_mask, test_label_mask, val_label_mask


def label_to_one_hot(data_gt: np.ndarray | Tensor, class_num: int, flatten_hw=True):
    if isinstance(data_gt, np.ndarray):
        # To Tensor
        data_gt = torch.as_tensor(data_gt).to(torch.int32)
    oh_data_gt = torch.nn.functional.one_hot(data_gt, num_classes=class_num)  # (h, w, n_class)

    if flatten_hw:
        oh_data_gt = oh_data_gt.view(-1, class_num)

    return oh_data_gt.cpu().numpy()


def single_image_slide_train_data(img_size: int, img: Float[np.ndarray, "h w c"], img_gt: Float[np.ndarray, "h w"]):
    """
    Sliding window from HyperSIGMA.
    """

    H0, W0, C = img.shape
    if H0 < img_size:
        gap = img_size - H0
        mirror_img = img[(H0 - gap) : H0, :, :]
        mirror_img_gt = img_gt[(H0 - gap) : H0, :]
        img = np.concatenate([img, mirror_img], axis=0)
        img_gt = np.concatenate([img_gt, mirror_img_gt], axis=0)
    if W0 < img_size:
        gap = img_size - W0
        mirror_img = img[:, (W0 - gap) : W0, :]
        mirror_img_gt = img_gt[(W0 - gap) : W0, :]
        img = np.concatenate([img, mirror_img], axis=1)
        img_gt = np.concatenate([img_gt, mirror_img_gt], axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H + 1) * img_size - H
        mirror_img = img[(H - gap) : H, :, :]
        mirror_img_gt = img_gt[(H - gap) : H, :]
        img = np.concatenate([img, mirror_img], axis=0)
        img_gt = np.concatenate([img_gt, mirror_img_gt], axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap) : W, :]
        mirror_img_gt = img_gt[:, (W - gap) : W]
        img = np.concatenate([img, mirror_img], axis=1)
        img_gt = np.concatenate([img_gt, mirror_img_gt], axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size

    sub_imgs = []
    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size, :]
            sub_imgs.append(z)
    sub_imgs = np.array(sub_imgs)  # [num_H*num_W,img_size,img_size, C]

    return sub_imgs, num_H, num_W, img_gt, img


def single_img_slide_test_data(img_size, img):
    H0, W0, C = img.shape
    if H0 < img_size:
        gap = img_size - H0
        mirror_img = img[(H0 - gap) : H0, :, :]
        img = np.concatenate([img, mirror_img], axis=0)
    if W0 < img_size:
        gap = img_size - W0
        mirror_img = img[:, (W0 - gap) : W0, :]
        img = np.concatenate([img, mirror_img], axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H + 1) * img_size - H
        mirror_img = img[(H - gap) : H, :, :]
        img = np.concatenate([img, mirror_img], axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap) : W, :]
        img = np.concatenate([img, mirror_img], axis=1)
        # gap = img_size - num_W*img_size
        # img = img[:,(W - gap):W,:]
    H, W, C = img.shape
    print("padding img:", img.shape)

    num_H = H // img_size
    num_W = W // img_size

    sub_imgs = []
    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size, :]
            sub_imgs.append(z)
    sub_imgs = np.array(sub_imgs)  # [num_H*num_W,img_size,img_size, C ]

    return sub_imgs, num_H, num_W, img


# * --- Wrapper --- #


class SingleImageHyperspectralSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sampled_data: Tensor,
        ori_gt: np.ndarray,
        sampled_gt: np.ndarray,
        sampled_index: np.ndarray,
        mask: np.ndarray,
        slide_info: dict,
        norm_type: str = "min_max",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.data = torch.as_tensor(sampled_data)  # (N, c, h_patch, w_patch)

        # Apply normalization to the data
        self.data = self._norm_img(self.data, norm_type=norm_type, eps=eps)

        self.gt = torch.as_tensor(ori_gt).to(torch.int64)
        self._sampled_index = torch.as_tensor(sampled_index).to(torch.int64)
        self._sampled_gt = torch.as_tensor(sampled_gt).to(torch.int64)  # (train_indices,)
        self._mask = mask
        self._norm_type = norm_type

        self.n_rows = slide_info["n_rows"]
        self.n_cols = slide_info["n_cols"]
        self.total_patches = self.data.shape[0]

    def _norm_img(
        self,
        data: Float[Tensor, "N c h w"],
        norm_type: str = "min_max",
        eps: float = 1e-8,
    ):
        if norm_type == "min_max":
            # Min-max normalization to [0, 1]
            min_val = data.amin(dim=(2, 3), keepdim=True)  # (N, c, 1, 1)
            max_val = data.amax(dim=(2, 3), keepdim=True)  # (N, c, 1, 1)

            # Avoid division by zero
            normalized_data = (data - min_val) / (max_val - min_val + eps)

            return normalized_data
        elif norm_type == "standard_norm":
            # Standard normalization (z-score): mean=0, std=1
            mean = data.mean(dim=(2, 3), keepdim=True)  # (N, c, 1, 1)
            std = data.std(dim=(2, 3), keepdim=True)  # (N, c, 1, 1)

            # Avoid division by zero
            normalized_data = (data - mean) / (std + eps)

            return normalized_data
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    @property
    def gt_for_loss(self):
        return self._sampled_gt[self._sampled_index].to(torch.int64)  # (n_indices,)

    @staticmethod
    def indexing_prediction_for_loss(prediction: Float[Tensor, "b c h w"]): ...

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return {
            "img": self.data[index],
            "gt_full": self.gt,
            "sample_index": self._sampled_index,
            "mask": self._mask,
        }

    @classmethod
    def from_files(
        cls,
        mat_data_file: str,
        mat_gt_file: str,
        window_size: int = 128,
        train_ratio=0.99,
        val_ratio=0.01,
        norm_type: str = "min_max",
        eps: float = 1e-8,
    ):
        img, gt = map(read_image, [mat_data_file, mat_gt_file])
        assert isinstance(img, np.ndarray) and isinstance(gt, np.ndarray)
        assert img.ndim == 3 and gt.ndim == 2
        assert img.shape[:-1] == gt.shape

        # Per-class train/test/val sampling
        indices, data, samples_gt, masks, info_of_original = cls._split_data(
            img, gt, window_size, None, train_ratio, val_ratio
        ).values()
        dataset = cls(
            data,
            gt,
            samples_gt[0],
            indices[0],
            masks[0],
            info_of_original,
            norm_type=norm_type,
            eps=eps,
        )  # only train set

        return dataset

    @staticmethod
    def _split_data(
        data,
        gt,
        img_size: int,
        n_class: int | None = None,
        train_ratio=1.0,
        val_ratio=0.0,
    ):
        n_class = n_class or gt.max()
        train_index, val_index, test_index = single_image_split_dataset(
            gt_flatten=gt.flatten(),
            class_num=n_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            samples_type="ratio",
        )
        train_samples_gt, test_samples_gt, val_samples_gt = single_image_get_seg_label(
            gt.flatten(), train_index, val_index, test_index
        )
        train_label_mask, test_label_mask, val_label_mask = single_image_get_label_mask(
            train_samples_gt, test_samples_gt, val_samples_gt, gt, n_class
        )
        train_gt, test_gt, val_gt = map(
            lambda x: np.reshape(x, gt.shape),
            [train_samples_gt, test_samples_gt, val_samples_gt],
        )
        img_train, n_rows, n_cols, gt, data = single_image_slide_train_data(
            img_size,
            data,  # ? why data here
            gt,
        )

        # To Tensor
        img_train = torch.as_tensor(img_train).permute(0, 3, 1, 2).float()
        # img_val = torch.as_tensor(img_val).permute(0, 3, 1, 2).float()
        # img_test = torch.as_tensor(img_test).permute(0, 3, 1, 2).float()

        # No zero label
        train_samples_gt[train_index] = train_samples_gt[train_index] - 1

        log(
            f"[HyperSegmentationDataset]: Prepare segmentation dataset done. ",
            f"Total {img_train.shape[0]} training samples",
        )

        return {
            "indices": [train_index, val_index, test_index],
            "data": img_train,
            "samples_gt": [train_samples_gt, val_samples_gt, test_samples_gt],
            "masks": [train_label_mask, val_label_mask, test_label_mask],
            "slide_info": {"n_rows": n_rows, "n_cols": n_cols},
        }


# * --- Test --- #


def test_seg_data_split():
    from src.utilities.io import read_image

    data_path = "data/Downstreams/ClassificationCollection/cls/mat/PaviaU.mat"
    gt_path = "data/Downstreams/ClassificationCollection/cls/cls_GT/PaviaU_gt.mat"

    data = read_image(data_path)
    gt = read_image(gt_path)

    # Split data
    train_index, val_index, test_index = single_image_split_dataset(
        gt_flatten=gt.flatten(),
        class_num=gt.max(),
        train_ratio=0.8,
        val_ratio=0.1,
        samples_type="ratio",
    )
    train_samples_gt, test_samples_gt, val_samples_gt = single_image_get_seg_label(
        gt.flatten(),
        train_index,
        val_index,
        test_index,
    )
    train_label_mask, test_label_mask, val_label_mask = single_image_get_label_mask(
        train_samples_gt, test_samples_gt, val_samples_gt, gt, gt.max()
    )

    # Reshape back to 2d
    train_gt, test_gt, val_gt = map(
        lambda x: np.reshape(x, gt.shape),
        [train_samples_gt, test_samples_gt, val_samples_gt],
    )
    pass


def test_dataset():
    data_path = "data/Downstreams/ClassificationCollection/cls/mat/PaviaU.mat"
    gt_path = "data/Downstreams/ClassificationCollection/cls/cls_GT/PaviaU_gt.mat"

    ds = SingleImageHyperspectralSegmentationDataset.from_files(
        data_path, gt_path, window_size=128, train_ratio=1.0, val_ratio=0.0
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    for x in dl:
        print(x.shape, "min:", x.min(), "max:", x.max())


if __name__ == "__main__":
    # test_seg_data_split()
    test_dataset()
