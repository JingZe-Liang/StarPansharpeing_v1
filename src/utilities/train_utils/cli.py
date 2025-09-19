import argparse
import sys

from loguru import logger
from omegaconf import OmegaConf
from rich_argparse import RichHelpFormatter

CLI_DEFAUTL_DICT = {
    "config_name": None,
    "only_rank_zero_catch": True,
}


def remove_argparser_args(unknown_args):
    script_name = sys.argv[0]
    sys.argv = [script_name] + unknown_args


def argsparse_cli_args(
    exp_config_dict: dict[str, str],
    cli_default_dict: dict = CLI_DEFAUTL_DICT,
    parser: argparse.ArgumentParser | None = None,
):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)

    # fmt: off
    config_name_parser = parser.add_argument_group("Configuration Names", description="Determine which config to be used to launch the experiment")
    config_name_parser.add_argument("--config_name", "-c", type=str, default=None, help=f"Available config names: {list(exp_config_dict.keys())}")
    config_name_parser.add_argument("--only_rank_zero_catch", "-zc", action="store_false", default=True, help="Only catch exceptions on rank zero")
    # fmt: on

    cli_args, unknown = parser.parse_known_args()
    config_name = cli_args.config_name or cli_default_dict["config_name"]
    hydra_config_path = exp_config_dict.get(config_name)

    # refactor the sys.argv
    remove_argparser_args(unknown)

    if hydra_config_path is None:
        logger.error(f"Unknown config name: {config_name}")
        sys.exit(1)

    return hydra_config_path, cli_args


def get_cli_cfg_isolate_with_hydra_cfg(
    exp_config_dict: dict[str, str], cli_default_dict: dict = CLI_DEFAUTL_DICT
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
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()

        # 创建标题
        title = Text("🚀 Hyperspectral 1D Tokenizer CLI Help", style="bold magenta")
        console.print(Panel(title, expand=False))

        # CLI 参数表格
        cli_table = Table(
            title="📋 CLI Arguments", show_header=True, header_style="bold cyan"
        )
        cli_table.add_column("Parameter", style="green", width=20)
        cli_table.add_column("Type", style="yellow", width=10)
        cli_table.add_column("Default", style="blue", width=15)
        cli_table.add_column("Description", style="white")

        if list(cli_default_dict.keys()) != list(CLI_DEFAUTL_DICT.keys()):
            for key, value in cli_default_dict.items():
                cli_table.add_row(
                    key, type(value).__name__, str(value), "Custom CLI parameter"
                )
        else:
            cli_table.add_row(
                "config_name", "str", "None", "Name of the config file to use"
            )
            cli_table.add_row(
                "only_rank_zero_catch",
                "bool",
                "True",
                "Whether to only catch exceptions on rank 0",
            )

        cli_table.add_row(
            "help", "bool", "False", "Show this help information (h, -h, --help)"
        )

        console.print(cli_table)
        console.print()

        usage_panel = Panel(
            "[bold green]python main.py[/bold green] [cyan]config_name=some_config_name[/cyan] [yellow]some_hydra_args[/yellow]",
            title="💡 Usage Example",
            title_align="left",
        )
        console.print(usage_panel)
        console.print()

        hydra_panel = Panel(
            "Hydra arguments depend on the chosen config file.\nRefer to your specific config file for available options.",
            title="⚙️  Hydra Arguments",
            title_align="left",
        )
        console.print(hydra_panel)
        console.print()

        config_table = Table(
            title="📁 Available Config Files",
            show_header=True,
            header_style="bold cyan",
        )
        config_table.add_column("Config Name", style="green", width=20)
        config_table.add_column("File Path", style="blue")

        for config_name, config_path in exp_config_dict.items():
            config_table.add_row(config_name, config_path)

        console.print(config_table)

        exit(0)

    # exp config
    hydra_cfg_name = exp_config_dict[str(cli_args.config_name)]

    return hydra_cfg_name, cli_args


from pyfiglet import figlet_format
from rich.console import Console
from rich.text import Text


def print_colored_banner(text: str, font: str = "slant", color: str = "bold magenta"):
    ascii_art = figlet_format(text, font=font)
    console = Console()
    styled_text = Text(ascii_art, style=color)
    console.print(styled_text, width=400)
