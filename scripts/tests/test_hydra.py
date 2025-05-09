import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))

import sys

sys.path.insert(
    0,
    "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer",
)


@hydra.main(
    config_path="../configs/pansharpening",
    config_name="cosmos_f8c16p4_fusionnet",
    version_base=None,
)
def main(args):
    # print(OmegaConf.to_yaml(args, resolve=True))
    # hydra.utils.instantiate(args.train.tokenizer_optimizer)

    # print(OmegaConf.to_yaml(args.dataset, resolve=True))
    # print(OmegaConf.to_container(args.dataset.train.wds_paths, resolve=True))
    print(args.dataset.consts.bands)

    # print(args.dataset.batch_size_val)
    # print(args.vq_loss)
    # print(args.dataset)


main()
