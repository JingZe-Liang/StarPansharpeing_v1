from __future__ import annotations

from typing import Any, Literal

import litdata as ld
import pytorch_lightning as pl

from src.stage2.cloud_removal.data.CUHK_CR import CUHK_CR_StreamingDataset


def _merge_dicts(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base) if base else {}
    if override:
        merged.update(override)
    return merged


class CUHK_CR_StreamingDataset_EMRDM(CUHK_CR_StreamingDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        img = sample["img"]
        gt = sample["gt"]
        sample["label"] = gt
        sample["cond_image"] = img
        sample["cloudy"] = img
        return sample


class CUHKCRLitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        input_dir: str,
        name: Literal["cr1", "cr2"],
        batch_size: int,
        num_workers: int,
        to_neg_1_1: bool = True,
        interp_to: int | None = None,
        rgb_nir_aug_p: float = 0.0,
        rgb_nir_aug_size: tuple[int, int] | None = None,
        train_split: Literal["train", "test"] | None = "train",
        val_split: Literal["train", "test"] | None = "test",
        test_split: Literal["train", "test"] | None = "test",
        predict_split: Literal["train", "test"] | None = "test",
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
        shuffle_predict: bool = False,
        batch_size_val: int | None = None,
        batch_size_test: int | None = None,
        batch_size_predict: int | None = None,
        loader_kwargs: dict[str, Any] | None = None,
        train_loader_kwargs: dict[str, Any] | None = None,
        val_loader_kwargs: dict[str, Any] | None = None,
        test_loader_kwargs: dict[str, Any] | None = None,
        predict_loader_kwargs: dict[str, Any] | None = None,
        train_ds_kwargs: dict[str, Any] | None = None,
        val_ds_kwargs: dict[str, Any] | None = None,
        test_ds_kwargs: dict[str, Any] | None = None,
        predict_ds_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.to_neg_1_1 = to_neg_1_1
        self.interp_to = interp_to
        self.rgb_nir_aug_p = rgb_nir_aug_p
        self.rgb_nir_aug_size = rgb_nir_aug_size

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.predict_split = predict_split

        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.shuffle_predict = shuffle_predict

        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.batch_size_predict = batch_size_predict

        self.loader_kwargs = loader_kwargs
        self.train_loader_kwargs = train_loader_kwargs
        self.val_loader_kwargs = val_loader_kwargs
        self.test_loader_kwargs = test_loader_kwargs
        self.predict_loader_kwargs = predict_loader_kwargs

        self.train_ds_kwargs = train_ds_kwargs
        self.val_ds_kwargs = val_ds_kwargs
        self.test_ds_kwargs = test_ds_kwargs
        self.predict_ds_kwargs = predict_ds_kwargs

        self.datasets: dict[str, CUHK_CR_StreamingDataset_EMRDM] = {}
        self._generator = None
        self._generator_val = None

    def set_generator(self, f, g=None) -> None:
        self._generator = f
        self._generator_val = g if g is not None else f

    def prepare_data(self) -> None:
        return

    def _build_dataset(
        self,
        *,
        split: Literal["train", "test"],
        shuffle: bool,
        extra_kwargs: dict[str, Any] | None,
    ) -> CUHK_CR_StreamingDataset_EMRDM:
        kwargs = {
            "input_dir": self.input_dir,
            "name": self.name,
            "split": split,
            "to_neg_1_1": self.to_neg_1_1,
            "interp_to": self.interp_to,
            "rgb_nir_aug_p": self.rgb_nir_aug_p,
            "rgb_nir_aug_size": self.rgb_nir_aug_size,
            "shuffle": shuffle,
        }
        kwargs = _merge_dicts(kwargs, extra_kwargs)
        return CUHK_CR_StreamingDataset_EMRDM(**kwargs)

    def _build_loader(
        self,
        *,
        dataset: CUHK_CR_StreamingDataset_EMRDM,
        batch_size: int,
        loader_kwargs: dict[str, Any] | None,
    ) -> ld.StreamingDataLoader:
        base_kwargs = {"batch_size": batch_size, "num_workers": self.num_workers, "pin_memory": True}
        merged = _merge_dicts(base_kwargs, self.loader_kwargs)
        merged = _merge_dicts(merged, loader_kwargs)
        return ld.StreamingDataLoader(dataset, **merged)  # type: ignore[arg-type]

    def setup(self, stage: str | None = None) -> None:
        self.datasets = {}
        if self.train_split is not None:
            self.datasets["train"] = self._build_dataset(
                split=self.train_split,
                shuffle=self.shuffle_train,
                extra_kwargs=self.train_ds_kwargs,
            )
        if self.val_split is not None:
            self.datasets["validation"] = self._build_dataset(
                split=self.val_split,
                shuffle=self.shuffle_val,
                extra_kwargs=self.val_ds_kwargs,
            )
        if self.test_split is not None:
            self.datasets["test"] = self._build_dataset(
                split=self.test_split,
                shuffle=self.shuffle_test,
                extra_kwargs=self.test_ds_kwargs,
            )
        if self.predict_split is not None:
            self.datasets["predict"] = self._build_dataset(
                split=self.predict_split,
                shuffle=self.shuffle_predict,
                extra_kwargs=self.predict_ds_kwargs,
            )

    def train_dataloader(self) -> ld.StreamingDataLoader:
        dataset = self.datasets["train"]
        return self._build_loader(
            dataset=dataset,
            batch_size=self.batch_size,
            loader_kwargs=self.train_loader_kwargs,
        )

    def val_dataloader(self) -> ld.StreamingDataLoader:
        dataset = self.datasets["validation"]
        batch_size = self.batch_size_val or self.batch_size
        return self._build_loader(
            dataset=dataset,
            batch_size=batch_size,
            loader_kwargs=self.val_loader_kwargs,
        )

    def test_dataloader(self) -> ld.StreamingDataLoader:
        dataset = self.datasets["test"]
        batch_size = self.batch_size_test or self.batch_size
        return self._build_loader(
            dataset=dataset,
            batch_size=batch_size,
            loader_kwargs=self.test_loader_kwargs,
        )

    def predict_dataloader(self) -> ld.StreamingDataLoader:
        dataset = self.datasets["predict"]
        batch_size = self.batch_size_predict or self.batch_size
        return self._build_loader(
            dataset=dataset,
            batch_size=batch_size,
            loader_kwargs=self.predict_loader_kwargs,
        )
