import torch

__default_wids_keys = [
    "__key__",
    "__index__",
    "__shard__",
    "__shardindex__",
]


# TODO: complete this collate function
def multimodal_wids_collate_fn(batch: list[dict]):
    """
    Custom collate function for multimodal WIDS dataset.
    This function handles the different modalities in the batch.
    """

    collated_batch = {
        "__key__": {},
        "__index__": {},
        "__shard__": {},
        "__shardindex__": {},
    }
    for d in batch:
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, got {type(d)}")
        for name, mm_dict in d.items():
            for key, value in mm_dict.items():
                if key in __default_wids_keys:
                    # collated_batch
                    ...
