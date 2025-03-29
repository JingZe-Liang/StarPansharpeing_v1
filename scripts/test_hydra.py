import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))


@hydra.main(config_path="configs/cosmos", config_name="pos_train")
def main(args):
    print(OmegaConf.to_container(args.dataset, resolve=True))


main()
