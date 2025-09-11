"""
Pansharpening Metrics Analysis

Author: Zihan Cao
Date: 2023/09/10
Email: iamzihan666@gmail.com
License: GPL v3

---------------------------------------------------------

Copyright (c) ZihanCao,
University of Electronic Science and Technology of China (UESTC),
Mathematical School
"""

from functools import partial
from warnings import warn

import numpy as np
import torch
import torch.multiprocessing as mp
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor
from torchmetrics.aggregation import MeanMetric

from src.utilities.logging import log_print

from .utils._metric_legacy import analysis_accu
from .utils.indexes_evaluation_FS import indexes_evaluation_FS


def to_numpy(*args):
    l = []
    for i in args:
        if isinstance(i, torch.Tensor):
            l.append(i.detach().cpu().numpy())
    return l


def dict_to_str(d, decimals=4):
    n = len(d)

    # func = lambda k, v: f"{k}: {torch.round(v, decimals=decimals).item() if isinstance(v, torch.Tensor) else round(v, decimals)}"
    def func(k, v):
        if isinstance(v, torch.Tensor):
            return f"{k}: {round(v.item(), decimals)}"
        elif isinstance(v, np.ndarray):
            return f"{k}: {np.round(v, decimals=decimals)}"
        elif isinstance(v, (float, int)):
            return f"{k}: {round(v, decimals)}"
        else:
            raise ValueError(f"Unsupported type: {type(v)}")

    s = ""
    for i, (k, v) in enumerate(d.items()):
        s += func(k, v) + (", " if i < n - 1 else "")
    return s


def normalize_to_01(x):
    # normalize tensor to [0, 1]
    if isinstance(x, torch.Tensor):
        x -= x.flatten(-2).min(-1, keepdim=True)[0][..., None]
        x /= x.flatten(-2).max(-1, keepdim=True)[0][..., None]
    elif isinstance(x, np.ndarray):
        x -= x.min((-2, -1), keepdims=True)
        x /= x.max((-2, -1), keepdims=True)
    else:
        raise TypeError("x should be tensor or numpy array")

    return x


def psnr_one_img(img_gt, img_test):
    """
    calculate PSNR for one image
    :param img_gt: ground truth image, numpy array, shape [H, W, C]
    :param img_test: test or inference image, numpy array, shape [H, W, C]
    :return: PSNR, float type
    """
    assert img_gt.shape == img_test.shape, (
        "image 1 and image 2 should have the same size"
    )
    return peak_signal_noise_ratio(img_gt, img_test)


def psnr_batch_tensor_metric(b_gt, b_pred):
    """
    calculate PSNR for batch tensor images
    :param b_gt: tensor, shape [B, C, H, W]
    :param b_test: tensor, shape [B, C, H, W]
    :return:
    """
    assert b_gt.shape[0] == b_pred.shape[0]
    bs = b_gt.shape[0]
    psnr = 0.0
    for gt, t in zip(b_gt, b_pred):
        psnr += psnr_one_img(*(to_numpy(gt, t)))
    return psnr / bs


def ssim_one_image(img_gt, img_test, channel_axis=0):
    assert img_gt.shape == img_test.shape, (
        "image 1 and image 2 should have the same size"
    )
    return structural_similarity(
        img_gt, img_test, channel_axis=channel_axis, data_range=1.0
    )


def ssim_batch_tensor_metric(b_gt, b_pred):
    assert b_gt.shape[0] == b_pred.shape[0]
    bs = b_gt.shape[0]
    ssim = 0.0
    for gt, t in zip(b_gt, b_pred):
        ssim += ssim_one_image(*(to_numpy(gt, t)), channel_axis=0)
    return ssim / bs


class NonAnalysis(object):
    def __init__(self):
        self.acc_ave = {}  # only used as attribution

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return "NonAnalysis()"


# FIXME: this python code is not same as matlab code, you should use matlab code to get the real accuracy
# only used in training and validate
class AnalysisPanAcc(object):
    def __init__(self, ratio=4, ref=True, ergas_ratio: int = 4, **unref_factory_kwargs):
        """pansharpening metric analysis class

        Args:
            ratio (int, optional): fusion ratio. Defaults to 4.
            ref (bool, optional): reduce-resolution or full-resolution. Defaults to True.
            ergas_ratio (int, optional): previous api (may decrepated soon). Defaults to 4.
            unref_factory_kwargs(dict): sensor, default_max_value. Defaults to {'sensor': 'default', 'default_max_value': None}.

        ## Main function call
        Args:

            - ref mode (reduced-resolution)
                b_gt (torch.Tensor): [b, c, h, w]
                b_pred (torch.Tensor): [b, c, h, w]

            - unref mode (full-resolution)
                sr (torch.Tensor): [b, c, h, w]
                ms (torch.Tensor, optional): [b, c, h/ratio, w/ratio]
                lms (torch.Tensor): [b, c, h, w]
                pan (torch.Tensor): [b, c, h, w]

        """
        # ergas_ratio is decrepated
        if ratio is None:
            ratio = ergas_ratio
            warn(
                "`ergas_ratio` is deprecated, use ratio instead",
                category=DeprecationWarning,
            )
        self.ratio = ratio
        self.ref = ref

        # metric functions
        if ref:
            self.__sam_ergas_psnr_cc_one_image = partial(
                analysis_accu, ratio=ergas_ratio, choices=5
            )
            self.ssim = ssim_batch_tensor_metric
        else:
            # @sensor in ['QB', 'IKONOS', 'WV2', 'WV3', 'default']
            assert (
                "sensor" in unref_factory_kwargs
                or "default_max_value" in unref_factory_kwargs
            ), "@sensor or @default_max_value should be specified in unrefactory_kwargs"
            sensor = unref_factory_kwargs.pop("sensor", "default").upper()

            if sensor == "DEFAULT":
                warn("sensor is not specified, use default sensor type")
            self.default_max_value = unref_factory_kwargs.pop("default_max_value", None)

            if self.default_max_value is None:
                _default_max_value = {
                    "QB": 2047,
                    "IKONOS": 1023,
                    "WV2": 2047,
                    "WV3": 2047,
                    "GF2": 1023,
                    "DEFAULT": 2047,
                    "CAVE_X4": 1,
                    "CAVE_X8": 1,
                    "HARVARD_X4": 1,
                    "HARVARD_X8": 1,
                    "GF5": 1,
                    "GF2-GF5": 1,
                }
                self.default_max_value = _default_max_value.get(sensor)
                log_print(
                    f">>> `default_max_value` is not specified, set it according to `sensor`:"
                    f"{sensor, self.default_max_value}\n"
                    "-" * 20,
                    level="warning",
                )

            self.FS_metric_fn = partial(
                indexes_evaluation_FS,
                L=11,
                Qblocks_size=32,
                sensor=sensor,
                th_values=0,
                ratio=ratio,
                flagQNR=False,
            )

        # tracking accuracy
        self._acc_d = {}
        self._call_n = 0
        self.acc_ave = (
            {"SAM": 0.0, "ERGAS": 0.0, "PSNR": 0.0, "CC": 0.0, "SSIM": 0.0}
            if ref
            else {"D_S": 1.0, "D_lambda": 1.0, "HQNR": 0.0}
        )

    @property
    def empty_acc(self):
        return (
            {"SAM": 0.0, "ERGAS": 0.0, "PSNR": 0.0, "CC": 0.0, "SSIM": 0.0}
            if self.ref
            else {"D_S": 1.0, "D_lambda": 1.0, "HQNR": 0.0}
        )

    @staticmethod
    def permute_dim(*args, permute_dims=(1, 2, 0)):
        l = []
        for i in args:
            l.append(i.permute(*permute_dims))
        return l

    @staticmethod
    def _sum_acc(d_ave, d_now, n, n2=1):
        assert len(d_ave) == len(d_now)
        for k in d_ave.keys():
            v2 = d_now[k] * n2
            d_ave[k] *= n
            d_ave[k] += v2.cpu().item() if isinstance(v2, torch.Tensor) else v2
        return d_ave

    @staticmethod
    def _average_acc(d_ave, n):
        for k in d_ave.keys():
            d_ave[k] /= n
        return d_ave

    def sam_ergas_psnr_cc_batch(self, b_gt, b_pred):
        b_gt.shape[0]
        # input shape should be [B, C, H, W]
        acc_ds = {"SAM": 0.0, "ERGAS": 0.0, "PSNR": 0.0, "CC": 0.0}
        for i, (img1, img2) in enumerate(zip(b_gt, b_pred)):
            img1, img2 = self.permute_dim(img1, img2)
            acc_d = self.__sam_ergas_psnr_cc_one_image(img1, img2)
            acc_ds = self._sum_acc(acc_ds, acc_d, i)
            acc_ds = self._average_acc(acc_ds, i + 1)
        return acc_ds

    def D_lambda_D_s_HQNR_batch(self, sr=None, ms=None, lms=None, pan=None):
        assert sr is not None and lms is not None and pan is not None and ms is not None
        if ms is None:
            ms = torch.nn.functional.interpolate(
                lms, scale_factor=1 / self.ratio, mode="bilinear", align_corners=False
            )

        acc_ds = {"D_S": 1.0, "D_lambda": 1.0, "HQNR": 0.0}
        sr, ms, lms, pan = self.permute_dim(sr, ms, lms, pan, permute_dims=(0, 2, 3, 1))
        sr, ms, lms, pan = to_numpy(sr, ms, lms, pan)
        _max_value = getattr(self, "default_max_value")
        sr, ms, lms, pan = map(
            lambda x: np.clip(x * _max_value, 0, _max_value), [sr, ms, lms, pan]
        )
        for i, (sr_i, ms_i, lms_i, pan_i) in enumerate(zip(sr, ms, lms, pan)):
            QNR_index, D_lambda, D_S = self.FS_metric_fn(
                I_F=sr_i, I_MS_LR=ms_i, I_MS=lms_i, I_PAN=pan_i
            )
            acc_d = dict(HQNR=QNR_index, D_lambda=D_lambda, D_S=D_S)
            acc_ds = self._sum_acc(acc_ds, acc_d, i)
            acc_ds = self._average_acc(acc_ds, i + 1)

        return acc_ds

    def once_batch_call(self, **kwargs):
        if self.ref:
            acc_d1 = self.sam_ergas_psnr_cc_batch(**kwargs)
            acc_ssim = self.ssim(**kwargs)
            acc_d1["SSIM"] = acc_ssim
        else:
            acc_d1 = self.D_lambda_D_s_HQNR_batch(**kwargs)
        self._acc_d = acc_d1
        return acc_d1

    def _call_check_args_to_kwargs(self, *args):
        def may_np_to_tensor(d):
            for k, v in d.items():
                if not torch.is_tensor(v):
                    d[k] = torch.tensor(v, dtype=torch.float32)
                elif v.dtype != torch.float32:
                    d[k] = v.float()

            return d

        if len(args) == 2:
            assert self.ref, "ref mode should have 2 args"
            kwargs = dict(b_gt=args[0], b_pred=args[1])
            assert args[0].shape == args[1].shape
        elif len(args) == 3:
            assert not self.ref, "unref mode should have more than 2 args"
            kwargs = dict(sr=args[0], lms=args[1], pan=args[2])
            assert args[0].shape == args[1].shape == args[2].shape
        elif len(args) == 4:
            assert not self.ref, "unref mode should have more than 2 args"
            kwargs = dict(sr=args[0], ms=args[1], lms=args[2], pan=args[3])
            bs, c, h, w = args[1].shape
            assert (
                args[0].shape
                == torch.Size((bs, c, int(h * self.ratio), int(w * self.ratio)))
                == args[2].shape
                # == args[3].shape
            )
        else:
            raise ValueError("args should have 2 or 4 elements")

        return may_np_to_tensor(kwargs)

    def __call__(self, *args) -> dict:
        """
        Args:
            ref mode:
                b_gt (torch.Tensor): [b, c, h, w]
                b_pred (torch.Tensor): [b, c, h, w]

            unref mode:
                sr (torch.Tensor): [b, c, h, w]
                ms (torch.Tensor, optional): [b, c, h/ratio, w/ratio]
                lms (torch.Tensor): [b, c, h, w]
                pan (torch.Tensor): [b, c, h, w]
        """
        kwargs = self._call_check_args_to_kwargs(*args)

        n = args[0].shape[0]
        self.acc_ave = self._sum_acc(
            self.acc_ave, self.once_batch_call(**kwargs), self._call_n, n2=n
        )
        self.acc_ave = self._average_acc(self.acc_ave, self._call_n + n)
        self._call_n += n
        return self.acc_ave

    def clear_history(self, verbose=False):
        if verbose:
            log_print(">> AccAnalysis: clear history")
        self._acc_d = {}
        self._call_n = 0
        self.acc_ave = (
            {"SAM": 0.0, "ERGAS": 0.0, "PSNR": 0.0, "CC": 0.0, "SSIM": 0.0}
            if self.ref
            else {"D_S": 1.0, "D_lambda": 1.0, "HQNR": 0.0}
        )

    def print_str(self, decimals=6):
        return dict_to_str(self.acc_ave, decimals=decimals)

    def result_str(self):
        return self.print_str()

    def __repr__(self) -> str:
        repr_str = f"AnalysisPanAcc(ratio={self.ratio}, ref={self.ref}):"
        repr_str += f"\n{self.print_str()}"
        return repr_str


class PansharpeningMetrics(AnalysisPanAcc):
    def __init__(
        self,
        ratio=4,
        ref=True,
        eargas_ratio=None,  # decrepated
        device: str = "cuda",
        **unref_factory_kwargs,
    ):
        super().__init__(ratio, ref, **unref_factory_kwargs)

        # Convert to all MeanMetric
        self.acc_ave: dict[str, float]
        self.acc_ave_metrics = {
            k: MeanMetric().to(device) for k, v in self.acc_ave.items()
        }
        self._curr_acc: dict[str, Tensor] | None = None

    def _to_sync_metrics(self, acc: dict, weight: float | int = 1.0):
        for (k, update_v), name in zip(acc.items(), self.acc_ave_metrics.keys()):
            self.acc_ave_metrics[name].update(update_v, weight)
        return self.acc_ave_metrics

    def _get_acc_ave(self) -> dict[str, float]:
        acc_ave = {}
        for k, metric in self.acc_ave_metrics.items():
            # may synchronize across ranks and keep type as float
            acc_ave[k] = metric.compute().item()
        return acc_ave

    def compute(self) -> dict[str, float]:
        acc_ave = self._get_acc_ave()
        return acc_ave

    def __call__(self, *inp_tensors):
        kwargs = self._call_check_args_to_kwargs(*inp_tensors)
        n = inp_tensors[0].shape[0]
        device = inp_tensors[0].device

        acc_dict = self.once_batch_call(**kwargs)
        # to tensor
        acc_dict_th = {
            k: torch.as_tensor(v, device=device) for k, v in acc_dict.items()
        }
        self._curr_acc = acc_dict_th
        self.acc_ave_metrics = self._to_sync_metrics(acc_dict_th)
        self.acc_ave = self._get_acc_ave()
        self._call_n += n
        return self.acc_ave

    def update(self, *inp_tensors):
        return self.__call__(*inp_tensors)

    def clear_history(self, verbose=False):
        super().clear_history(verbose)
        for n, metric in self.acc_ave_metrics.items():
            metric.reset()


# * --- Test --- #


def test_pansharpening_metrics():
    """Test function for PansharpeningMetrics class"""

    # Test with reference mode (reduced-resolution)
    print("Testing with reference mode...")
    metric_ref = PansharpeningMetrics(ratio=4, ref=True)

    # Create test data
    batch_size = 2
    channels = 3
    height = 64
    width = 64

    # Generate random test tensors
    gt = torch.rand(batch_size, channels, height, width)
    pred = torch.rand(batch_size, channels, height, width)

    # Test metrics calculation
    metrics_ref = metric_ref(gt, pred)
    print(f"Reference mode metrics: {metrics_ref}")

    # Test with AnalysisPanAcc for comparison
    print("\nTesting with AnalysisPanAcc...")
    analysis_ref = AnalysisPanAcc(ratio=4, ref=True)
    metrics_analysis = analysis_ref(gt, pred)
    print(f"AnalysisPanAcc metrics: {metrics_analysis}")

    # Test batch accumulation
    print("\nTesting batch accumulation...")
    metric_ref.clear_history()

    # Process multiple batches
    for i in range(3):
        batch_gt = torch.rand(1, channels, height, width)
        batch_pred = torch.rand(1, channels, height, width)
        _ = metric_ref(batch_gt, batch_pred)
        print(f"Batch {i + 1}: {metric_ref._curr_acc}")

    print(f"\nFinal accumulated metrics: {metric_ref.acc_ave}")

    # Test clear history
    print("\nTesting clear history...")
    metric_ref.clear_history()
    print(f"After clear: {metric_ref.acc_ave}")

    # Test string representation
    print("\nTesting string representation...")
    metric_ref(gt, pred)
    print(f"String representation: {metric_ref.print_str()}")
    print(f"Repr: {repr(metric_ref)}")

    print("\nAll tests completed successfully!")


def test_pansharpening_metrics_multi_ranks(rank):
    """Test function for multi-rank (distributed) pansharpening metrics"""

    # Initialize distributed process group
    torch.distributed.init_process_group(
        backend="gloo", init_method="tcp://localhost:12355", world_size=2, rank=rank
    )

    # Set device
    device = torch.device(f"cpu")

    print(f"Rank {rank}: Starting distributed test...")

    # Create metrics with distributed support
    metric_ref = PansharpeningMetrics(ratio=4, ref=True)

    # Create test data - different data for each rank
    batch_size = 2
    channels = 3
    height = 64
    width = 64

    # Use different random seed for each rank
    torch.manual_seed(42 + rank)

    # Generate test tensors
    gt = torch.rand(batch_size, channels, height, width, device=device)
    pred = torch.rand(batch_size, channels, height, width, device=device)

    print(f"Rank {rank}: Generated test data with shape {gt.shape}")

    # Test metrics calculation
    metrics_ref = metric_ref(gt, pred)
    print(f"Rank {rank}: Reference mode metrics: {metrics_ref}")

    # Test batch accumulation across ranks
    print(f"Rank {rank}: Testing batch accumulation...")
    metric_ref.clear_history()

    # Process multiple batches
    for i in range(3):
        torch.manual_seed(42 + rank + i + 1)
        batch_gt = torch.rand(1, channels, height, width, device=device)
        batch_pred = torch.rand(1, channels, height, width, device=device)
        current_metrics = metric_ref(batch_gt, batch_pred)
        print(f"Rank {rank}: Batch {i + 1}: {current_metrics}")

    # Synchronize across ranks
    torch.distributed.barrier()

    print(f"Rank {rank}: Final accumulated metrics: {metric_ref.acc_ave}")

    # Test clear history
    print(f"Rank {rank}: Testing clear history...")
    metric_ref.clear_history()
    print(f"Rank {rank}: After clear: {metric_ref.acc_ave}")

    # Test with AnalysisPanAcc for comparison
    print(f"Rank {rank}: Testing with AnalysisPanAcc...")
    analysis_ref = AnalysisPanAcc(ratio=4, ref=True)
    metrics_analysis = analysis_ref(gt, pred)
    print(f"Rank {rank}: AnalysisPanAcc metrics: {metrics_analysis}")

    # Final barrier
    torch.distributed.barrier()

    print(f"Rank {rank}: All tests completed successfully!")

    # Clean up
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # test_pansharpening_metrics()

    # Run multi-rank test
    print("Starting multi-rank test...")
    mp.spawn(
        test_pansharpening_metrics_multi_ranks,
        args=(),
        nprocs=2,
        join=True,
    )
    print("Multi-rank test completed!")

    # sr = torch.rand(4, 3, 256, 256)
    # ms = torch.rand(4, 3, 64, 64)
    # lms = torch.rand(4, 3, 256, 256)
    # pan = torch.rand(4, 3, 256, 256)
    # gt = torch.rand(4, 3, 256, 256)

    # analysis = AnalysisPanAcc(ref=False, ratio=4, default_max_value=2047)

    # for i in range(2):
    #     analysis(sr[i : i + 2], ms[i : i + 2], lms[i : i + 2], pan[i : i + 2])
    #     print(analysis.print_str())
