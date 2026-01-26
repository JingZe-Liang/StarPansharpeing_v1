import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "data": ["US3DDepthStreamingDataset"],
    },
)
