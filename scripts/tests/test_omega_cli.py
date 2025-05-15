import hydra
from omegaconf import OmegaConf


def _get_cli_cfg_isolate_with_hydra_cfg(exp_config_dict: dict):
    cli_args = OmegaConf.create(
        {
            "config_name": "123",
            "only_rank_zero_catch": True,
        }
    )

    # args from cli
    _req_help = False
    for index, args_in_cli in enumerate(sys.argv[1:]):
        if args_in_cli in ("help", "?", "-h", "--help"):
            _req_help = True
            break

        k_, v_ = args_in_cli.split("=")
        if k_ in cli_args.keys():
            cli_args[k_] = v_
            sys.argv.pop(index + 1)

    if _req_help:
        from rich import print

        print(
            "Help information:\n\n"
            "cli arguments:\n"
            "\t config_name: str, the name of the config file to use\n"
            "\t only_rank_zero_catch: bool, whether to only catch the exception on rank 0\n"
            "\t help(h, -h, --help): bool, whether to show the help information\n"
        )
        print("-" * 60)
        print("hydra arguments:\n \trefer to the choosen config file\n")
        print("-" * 60)
        print("default config file choices:\n")
        print(exp_config_dict)
        exit(0)

    # exp config
    hydra_cfg_name = exp_config_dict[str(cli_args.config_name)]

    return hydra_cfg_name, cli_args


if __name__ == "__main__":
    _configs_dict = {
        "123": "unicosmos_tokenizer_kl_repa_f8c16p4",
    }

    _configs, cli_args = _get_cli_cfg_isolate_with_hydra_cfg(_configs_dict)
    print(cli_args)

    @hydra.main(
        config_path="../configs/tokenizer_gan",
        config_name=_configs,  # Default config
        version_base=None,
    )
    def main(cfg):
        print(cfg.dataset.used)

    main()
