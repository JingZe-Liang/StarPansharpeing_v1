import numpy as np
import torch
from litdata.streaming.writer import BinaryWriter

from src.data.codecs import npy_codec_io, tiff_codec_io
from src.data.hyperspectral_loader import get_hyperspectral_dataloaders
from src.data.utils import norm_img
from src.stage2.denoise.utils.noise_adder import (
    UniHSINoiseAdderKornia,
    get_tokenizer_trainer_noise_adder,
)


def get_writer(path):
    return BinaryWriter(path, chunk_bytes="256Mb")


def _get_hypersigma_5cases_noise_adder_reimplem(case_name: str, use_torch: bool = True):
    """
    Get noise adder configuration for specific HyperSIGMA case.

    This function creates noise adders that exactly match HyperSIGMA's Case 2-5 scenarios:
    - Case 2: Non-iid Gaussian (σ=[30,50,70,90]) + Impulse noise
    - Case 3: Non-iid Gaussian (σ=[30,50,70,90]) + Deadline noise
    - Case 4: Non-iid Gaussian (σ=[30,50,70,90]) + Stripe noise
    - Case 5: Non-iid Gaussian (σ=[30,50,70,90]) + Stripe + Deadline + Impulse (complex)

    Args:
        case_name: Name of the case ('case2', 'case3', 'case4', 'case5')
        use_torch: Whether to use torch implementation (recommended for performance)

    Returns:
        Configured UniHSINoiseAdderKornia instance
    """
    case_configs = {
        # Case 1: Pure Gaussian noise
        # Original: AddNoise(sigma) with sigmas = [10, 30, 50, 70]
        "case1": {
            "noise_type": ["noniid"],  # Blind gaussian with σ range
            "is_neg_1_1": False,  # HyperSIGMA data is in [0,1] range
            "p": 1.0,  # 100% probability for testing
            "use_torch": use_torch,
            "clip_value": False,  # Don't clip to preserve noise characteristics
        },
        # Case 2: Non-iid Gaussian + Impulse noise
        # Original: Compose([AddNoiseNoniid(sigmas), AddNoiseCase2()])
        # AddNoiseCase2 = AddNoiseMixed([_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])], num_bands=[1/3])
        "case2": {
            "noise_type": [
                "noniid",
                "impulse",
            ],  # Order matters: noniid first, then impulse
            "is_neg_1_1": False,  # HyperSIGMA data is in [0,1] range
            "p": 1.0,  # 100% probability for testing
            "use_torch": use_torch,
            "clip_value": False,  # Don't clip to preserve noise characteristics
        },
        # Case 3: Non-iid Gaussian + Deadline noise
        # Original: Compose([AddNoiseNoniid(sigmas), AddNoiseCase3()])
        # AddNoiseCase3 = AddNoiseMixed([_AddNoiseStripe, _AddNoiseDeadline], num_bands=[1/3, 1/3])
        "case3": {
            "noise_type": ["noniid", "deadline"],  # noniid first, then deadline
            "is_neg_1_1": False,
            "p": 1.0,
            "use_torch": use_torch,
            "clip_value": False,
        },
        # Case 4: Non-iid Gaussian + Stripe noise
        # Original: Compose([AddNoiseNoniid(sigmas), AddNoiseCase4()])
        # AddNoiseCase4 = AddNoiseMixed([_AddNoiseStripe, _AddNoiseImpulse], num_bands=[1/3, 1/3])
        "case4": {
            "noise_type": ["noniid", "stripe"],  # noniid first, then stripe
            "is_neg_1_1": False,
            "p": 1.0,
            "use_torch": use_torch,
            "clip_value": False,
        },
        # Case 5: Complex noise - the most challenging case
        # Original: AddNoiseComplex = AddNoiseMixed([_AddNoiseStripe, _AddNoiseDeadline, _AddNoiseImpulse], num_bands=[1/3, 1/3, 1/3])
        # Plus additional noniid gaussian noise
        "case5": {
            "noise_type": [
                "noniid",
                "stripe",
                "deadline",
                "impulse",
            ],  # Full combination
            "is_neg_1_1": False,
            "p": 1.0,
            "use_torch": use_torch,
            "clip_value": False,
        },
    }

    if case_name not in case_configs:
        available_cases = list(case_configs.keys())
        raise ValueError(
            f"Unknown case name: {case_name}. Available cases: {available_cases}"
        )

    config = case_configs[case_name]
    return UniHSINoiseAdderKornia(
        noise_type=config["noise_type"],  # str | list[str]
        is_neg_1_1=bool(config["is_neg_1_1"]),  # bool
        p=float(config["p"]),  # float
        same_on_batch=False,
        keepdim=True,
        use_torch=bool(config["use_torch"]),  # bool
        clip_value=bool(config["clip_value"]),  # bool
    )


def get_all_hypersigma_noise_adders(
    use_torch: bool = True, cases: list[str] | None = None
):
    """
    Get all HyperSIGMA noise adders in a dictionary.

    Returns:
        Dictionary with case names as keys and noise adders as values
    """
    case_names = (
        ["case1", "case2", "case3", "case4", "case5"] if cases is None else cases
    )
    return {
        case: _get_hypersigma_5cases_noise_adder_reimplem(case, use_torch)
        for case in case_names
    }


def create_denoising_dataset(
    dataset_path: str = "data/Downstreams/WDCDenoise/hyper_images/Washington_DC_mall-191_bands-px_192-0000.tar",
    save_path: str = "data/Downstreams/WDCDenoise/_cases_litdata",
):
    # Data writer
    writer = get_writer(save_path)

    # Get noise adders for all HyperSIGMA cases
    noise_adders = get_all_hypersigma_noise_adders(use_torch=True)

    # Dataloader
    _, dl = get_hyperspectral_dataloaders(
        dataset_path,
        batch_size=1,
        num_workers=0,
        shuffle_size=0,
        to_neg_1_1=False,
        quantile_img_clip=0.995,
        resample=False,
        per_channel_norm=True,
        persistent_workers=False,
        # permute=False,
    )

    total_i = 0
    for case_name, noise_adder in noise_adders.items():
        print(f"{case_name}")

        case_i = 0
        for sample in dl:
            img = sample["img"]

            img_noisy = noise_adder(img)[0]
            img_noisy = img_noisy.clip(0, 1)
            img_noisy = img_noisy.cpu().numpy()

            img_noisy_bytes = tiff_codec_io(img_noisy.transpose(1, 2, 0))
            img_clean_bytes = tiff_codec_io(img[0].cpu().numpy().transpose(1, 2, 0))

            writer.add_item(
                total_i,
                {
                    "img_clean": img_clean_bytes,
                    "img_noisy": img_noisy_bytes,
                    "__key__": f"{case_name}_{case_i}",
                },
            )
            case_i += 1
            total_i += 1

    writer.done()


def test_generate_case():
    from litdata import StreamingDataset

    ds = StreamingDataset(
        input_dir="data/Downstreams/WDCDenoise/_cases_litdata",
    )

    for i in range(len(ds)):
        item = ds[i]
        img_clean = item["img_clean"]
        img_noisy = item["img_noisy"]
        print(
            f"Batch {i} - name {item['__key__']}:  Noisy image shape: {img_noisy.shape}"
        )


if __name__ == "__main__":
    # create_denoising_dataset()
    test_generate_case()
