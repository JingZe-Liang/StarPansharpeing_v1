import numpy as np
import scipy.io as sio


class Loader:
    def __init__(self): ...

    def load_anomaly_detection_data(self, dataset="indian"):
        if dataset == "cri":
            mat = sio.loadmat("Nuance_Cri_400_400_46_1254.mat")
            coarse_det_dict = sio.loadmat("/coarse_det/Cri_coarse_det_map.mat")
            print(mat["hsi"].shape)
            print(mat["hsi_gt"].shape)
            print(np.sum(mat["hsi_gt"]))
            r, c, d = mat["hsi"].shape
            original = mat["hsi"].reshape(r * c, d)
            gt = mat["hsi_gt"].reshape(r * c, 1)
            coarse_det = coarse_det_dict["show"]
        elif dataset == "pavia":
            mat = sio.loadmat("/Paiva_108_120_102_43.mat")
            coarse_det_dict = sio.loadmat("/coarse_det/Pavia_coarse_det_map.mat")
            print(mat["hsi"].shape)
            print(mat["hsi_gt"].shape)
            print(np.sum(mat["hsi_gt"]))
            r, c, d = mat["hsi"].shape
            original = mat["hsi"].reshape(r * c, d)
            gt = mat["hsi_gt"].reshape(r * c, 1)
            coarse_det = coarse_det_dict["show"]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        rows = np.arange(gt.shape[0])  # start from 0
        # ID(row number), data, class number
        All_data = np.c_[rows, original, gt]

        # Removing background and obtain all labeled data
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # All ID of labeled  data

        return All_data, labeled_data, rows_num, coarse_det, r, c, dataset

    def load_target_detection_data(self, dataset="indian"):
        if dataset == "mosaic":
            mat = sio.loadmat("mosaic.mat")
            coarse_det_dict = sio.loadmat("coarse_det/Mosaic.mat")
            print(mat["X"].shape)
            print(mat["TL"].shape)
            print(np.sum(mat["TL"]))
            r, c, d = mat["X"].shape
            original = mat["X"].reshape(r * c, d)
            gt = mat["TL"].reshape(r * c, 1)
            coarse_det = coarse_det_dict["Mosaic"]
        elif dataset == "aviris":
            mat = sio.loadmat("aviris.mat")
            coarse_det_dict = sio.loadmat("coarse_det/AVIRIS.mat")
            print(mat["X"].shape)
            print(mat["TL"].shape)
            print(np.sum(mat["TL"]))
            r, c, d = mat["X"].shape
            original = mat["X"].reshape(r * c, d)
            gt = mat["TL"].reshape(r * c, 1)
            coarse_det = coarse_det_dict["AVIRIS"]
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        rows = np.arange(gt.shape[0])  # start from 0
        # ID(row number), data, class number
        All_data = np.c_[rows, original, gt]

        # Removing background and obtain all labeled data
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # All ID of labeled data

        return All_data, labeled_data, rows_num, coarse_det, r, c, dataset
