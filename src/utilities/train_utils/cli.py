import sys

from omegaconf import OmegaConf

CLI_DEFAUTL_DICT = {
    "config_name": None,
    "only_rank_zero_catch": True,
}


def get_cli_cfg_isolate_with_hydra_cfg(
    exp_config_dict: dict, cli_default_dict: dict = CLI_DEFAUTL_DICT
):
    """Parse and isolate CLI arguments from Hydra configuration, providing help information.

    This function handles two types of arguments:
    1. Custom CLI arguments (config_name, only_rank_zero_catch)
    2. Hydra configuration arguments (passed through to Hydra)

    Args:
        exp_config_dict (dict): Dictionary mapping config names to Hydra config paths.
            Example: {"exp1": "configs/exp1.yaml", "exp2": "configs/exp2.yaml"}
        cli_default_dict (dict): Default config to use if not specified via CLI.

    Returns:
        tuple: (hydra_cfg_name, cli_args) where:
            - hydra_cfg_name: The resolved Hydra config path from exp_config_dict
            - cli_args: OmegaConf object containing parsed CLI arguments

    Notes:
        - Help can be triggered with any of: help, ?, -h, --help
        - Custom CLI arguments are removed from sys.argv to prevent Hydra parsing errors
        - The function will exit if help is requested

    Usage:
        hydra_cfg_name, cli_args = get_cli_cfg_isolate_with_hydra_cfg(
            exp_config_dict={"exp1": "configs/exp1.yaml", "exp2": "configs/exp2.yaml"},
            default_config_name="exp1",
        )

    Example in CLI:
        >>> python main.py config_name=exp1 some_hydra_args=i_am_hydra_args
    """

    cli_args = OmegaConf.create(cli_default_dict.copy())

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

        print("-" * 60)
        if list(cli_default_dict.keys()) != list(CLI_DEFAUTL_DICT.keys()):
            print("\nHelp information:\n\ncli arguments:\n")
            for key, value in cli_default_dict.items():
                print(f"\t{key}: default value {value}")
            print(
                "\t help(h, -h, --help): bool, whether to show the help information\n"
            )
        else:
            print(
                "\nHelp information:\n\n"
                "cli arguments:\n"
                "\t config_name:          str, the name of the config file to use\n"
                "\t only_rank_zero_catch: bool, whether to only catch the exception on rank 0\n"
                "\t help(h, -h, --help):  bool, whether to show the help information\n"
            )
        print("\nUsage:")
        print("\t python main.py config_name=some_config_name some_hydra_args\n")
        print("-" * 60)
        print("\nHydra arguments:\n \trefer to the choosen config file\n")
        print("-" * 60)
        print("\nDefault config file choices:\n")
        print(exp_config_dict)
        exit(0)

    # exp config
    hydra_cfg_name = exp_config_dict[str(cli_args.config_name)]

    return hydra_cfg_name, cli_args
