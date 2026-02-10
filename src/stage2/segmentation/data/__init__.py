from __future__ import annotations

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "five_billion": ["FiveBillionStreamingDataset", "FiveBillionLargeImageDataset"],
        "deep_globe": ["DeepGlobeRoadExtractionStreamingDataset"],
        "single_mat_loader": ["SingleMatDataset"],
        "oil_leakage": ["get_oil_leakage_dataloaders"],
        "sos_oil_leakage": [
            "SOSOilLeakageDataset",
            "SOSOilLeakageKeyTransform",
            "get_sos_oil_leakage_dataloader",
        ],
        "flood3i": ["Flood3IDataset", "get_flood3i_dataloader"],
        "atlantic_forest": ["AtlanticForestSegmentationDataset", "get_atlantic_forest_dataloader"],
        "cross_city_multimodal": [
            "CrossCityMultimodalSegmentationDataset",
            "CrossCityMultimodalPatchStreamingDataset",
            "CrossCityTrainLitDataConfig",
            "CrossCityLitDataStreamKwargs",
            "CrossCityLitDataCombinedKwargs",
            "CrossCityLitDataLoaderKwargs",
            "export_cross_city_train_patches_to_litdata",
        ],
        "data_split": [
            "single_image_split_dataset",
            "oversample_weak_classes",
            "single_image_get_seg_label",
            "single_image_get_label_mask",
            "single_image_slide_train_data",
            "single_img_slide_test_data",
            "SingleImageHyperspectralSegmentationDataset",
        ],
    },
)
