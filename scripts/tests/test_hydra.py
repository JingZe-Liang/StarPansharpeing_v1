import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))

import sys

sys.path.insert(0, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer")


@hydra.main(
    config_path="../configs/tokenizer_gan",
    config_name="cosmos_post_train_f8c16p4",
    version_base=None,
)
def main(args):
    accelerator = hydra.utils.instantiate(args.accelerator)
    hydra.utils.instantiate(args.vq_loss)
    # print(args.dataset.batch_size_val)
    # print(args.vq_loss)
    # print(args.dataset)


main()
