import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="configs/1d_tokenizer", config_name="1d_tokenizer_no_quantizer")
def main(args):
    print(OmegaConf.to_container(args.dataset, resolve=True))


main()
