import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "hyperspectral_loader": [
            "get_fast_test_hyperspectral_data",
            "get_hyperspectral_dataloaders",
            "ms_pan_dir_paired_loader",
        ],
        "litdata_hyperloader": [
            "CombinedStreamingDataset",
            "ConditionsStreamingDataset",
            "GenerativeStreamingDataset",
            "ImageStreamingDataset",
            "IndexedCombinedStreamingDataset",
            "SingleCycleStreamingDataset",
            "_BaseStreamingDataset",
        ],
        "multimodal_loader": [
            "MultimodalityDataloader",
            "get_mm_chained_loaders",
        ],
        "window_slider": [
            "WindowSlider",
            "create_windowed_dataloader",
        ],
    },
)
