import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))

import sys

sys.path.insert(
    0,
    "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer",
)

from src.utilities.optim import KronMars


@hydra.main(
    config_path="../configs/tokenizer_gan",
    config_name="unicosmos_lora_finetune_f8c16p4",
    version_base=None,
)
def main(args):
    accelerator = hydra.utils.instantiate(args.accelerator)
    print(accelerator.state.fsdp_plugin)
    # hydra.utils.instantiate(args.train.tokenizer_optimizer)

    # print(args.dataset.batch_size_val)
    # print(args.vq_loss)
    # print(args.dataset)


main()
