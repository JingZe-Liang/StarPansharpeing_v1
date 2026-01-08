import ast
from glob import glob
from pathlib import Path

import hydra
from omegaconf import ListConfig, OmegaConf

OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))
OmegaConf.register_new_resolver("class", lambda x: hydra.utils.get_class(x))
OmegaConf.register_new_resolver("list", lambda x: list(x))
OmegaConf.register_new_resolver("tuple", lambda x: tuple(x))
OmegaConf.register_new_resolver("glob", lambda x: ListConfig([str(p) for p in Path().glob(x)]))


@hydra.main(
    config_path="../configs/stereo_matching/",
    config_name="hybrid_stereo_matching",
    version_base=None,
)
def main(args):
    # print(OmegaConf.to_yaml(args, resolve=True))
    # hydra.utils.instantiate(args.train.tokenizer_optimizer)

    # print(OmegaConf.to_yaml(args.dataset, resolve=True))
    # print(OmegaConf.to_container(args.dataset.train.wds_paths, resolve=True))
    # print(type(args.dataset.channels))

    # print(args.dataset.batch_size_val)
    # print(args.vq_loss)
    # print(args.dataset)

    # _, loader = hydra.utils.instantiate(args.dataset.train_loader)
    # print(type(loader))

    # model_cls_path = "src.stage1.cosmos.cosmos_tokenizer.ContinuousImageTokenizer"
    # model_cls = hydra.utils.get_class(model_cls_path)
    # print(model_cls)

    # start_probs = args.dataset.train_loader.curriculum_kwargs.start_prob
    # print(scipy.special.softmax(start_probs))

    # print(OmegaConf.to_container(args.dataset.train_loader, resolve=True))
    print(OmegaConf.to_yaml(args.dataset, resolve=True))

    ds, loader = hydra.utils.instantiate(args.dataset.train)
    print(ds)
    for batch in loader:
        # print(batch)
        pass


def main_compose():
    hydra.initialize_config_dir(
        "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/scripts/configs/stereo_matching/dataset"
    )
    cfg = hydra.compose("us3d")
    print(cfg.train.input_dir)


# cfg_string = """
# cfg: ???
# name: test

# dataset:
#     ds_name: ${..name}

# """

# cfg = OmegaConf.create(cfg_string)
# print(cfg.dataset.ds_name)

# main()
main_compose()
