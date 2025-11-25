import hydra

import src.utilities.config_utils

hydra.initialize(config_path="../configs/change_detection/segment_model")
cfg = hydra.compose(config_name="tokenizer_hybrid")

print(cfg)

# init the model
hydra.utils.instantiate(cfg)
