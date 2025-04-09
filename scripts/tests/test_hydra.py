import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))

import sys

sys.path.insert(0, "/Data2/ZiHanCao/exps/hyperspectral-1d-tokenizer")


@hydra.main(
    config_path="../configs/1d_tokenizer",
    config_name="flowmo_no_quant_t512p8",
    version_base=None,
)
def main(args):
    accelerator = hydra.utils.instantiate(args.accelerator)
    hydra.utils.instantiate(args.vq_loss)

    # print(args.dataset.batch_size_val)
    # print(args.vq_loss)
    # print(args.dataset)


main()
