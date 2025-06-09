import ast

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))
OmegaConf.register_new_resolver("class", lambda x: hydra.utils.get_class(x))
OmegaConf.register_new_resolver("list", lambda x: list(x))
OmegaConf.register_new_resolver("tuple", lambda x: tuple(x))


@hydra.main(
    config_path="../configs/tokenizer_gan",
    config_name="unicosmos_tokenizer_f16c16p1",
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

    print(args.train.disc_optimizer.betas)


main()
