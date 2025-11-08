import argparse
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from litdata import StreamingDataLoader, StreamingDataset
from litdata.streaming.reader import BinaryReader, ChunksConfig
from litdata.utilities import dataset_utilities as litdata_utils
from loguru import logger


def _catch_signals(handler=None):
    """
    Set up signal handlers for graceful shutdown, particularly for Ctrl+C (SIGINT).

    This function registers handlers for SIGINT and SIGTERM to ensure
    proper cleanup when the user interrupts the program.
    """

    def signal_handler_default(signum: int, frame) -> None:
        """
        Handle interrupt signals by printing a message and exiting gracefully.

        Parameters
        ----------
        signum : int
            The signal number received
        frame : object
            The current stack frame
        """
        logger.info(f"\nReceived signal {signum}. Gracefully shutting down...")
        sys.exit(0)

    # Register signal handlers
    if handler is None:
        signal.signal(signal.SIGINT, signal_handler_default)
        signal.signal(signal.SIGTERM, signal_handler_default)
    else:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    # logger.info("Signal handlers registered. Use Ctrl+C to interrupt gracefully.")


def _close_tempfile(temp_file):
    def _inner(signum, frame):
        logger.info(f"\nReceived signal {signum}. Cleaning up temporary files...")
        if temp_file:
            temp_file.close()
            logger.info("Temporary config file deleted.")
        sys.exit(0)

    return _inner


class _BaseStreamingDataset(StreamingDataset):
    def __init__(self, *args, **kwargs) -> None:
        input_dir = Path(kwargs.pop("input_dir"))
        index_file_name = None
        if input_dir.is_dir():
            pass
        elif Path(input_dir).is_file() and Path(input_dir).suffix == ".json":
            index_file_name = input_dir.stem
            input_dir = input_dir.parent
        else:
            raise ValueError(
                f"input_dir must be a directory or a index json file, got: {input_dir}"
            )

        # FIXME: I don't know why the 'index_path' args does not work
        index_file_name = kwargs.pop("index_file_name", index_file_name)
        if index_file_name is not None:
            litdata_utils._INDEX_FILENAME = f"{index_file_name}.json"

        super().__init__(input_dir=input_dir, *args, **kwargs)

        # change back
        if index_file_name is not None:
            litdata_utils._INDEX_FILENAME = "index.json"

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        # override the class

        # if any(not isinstance(d, StreamingDataset) for d in datasets):
        #     raise RuntimeError("The provided datasets should be instances of the StreamingDataset.")
        return


def _read_config_from_dir(data_dir: str, index_file_name="index.json"):
    config_path = os.path.join(data_dir, index_file_name)
    # load config
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def _modify_config_data_format(config: dict):
    # to all bytes, no decode
    df = config["config"].get("data_format", [])
    for i, k in enumerate(df):
        if k != "str":
            df[i] = "bytes"
    config["config"]["data_format"] = df
    return config


def _build_dataset(data_dir: str, index_file_name="index.json"):
    config = _read_config_from_dir(data_dir, index_file_name)
    config = _modify_config_data_format(config)

    # write to a temp file
    config_temp_file = tempfile.NamedTemporaryFile(
        delete=True,
        mode="w",
        suffix=".json",
        delete_on_close=True,
        dir=data_dir,
    )
    _catch_signals(_close_tempfile(config_temp_file))
    json.dump(config, config_temp_file)
    config_temp_file.flush()
    logger.info(f"Temporary config file created at: {config_temp_file.name}")

    ds = _BaseStreamingDataset(
        input_dir=data_dir,
        index_file_name=Path(config_temp_file.name).stem,
    )
    return ds, config_temp_file


def _read_item_key(ds: StreamingDataset, index: int):
    item = ds[index]
    return item["__key__"]


def read_keys(data_dir: str, save_file: str | None = None):
    ds, temp_file = _build_dataset(data_dir)
    # touch the save file if not None
    if save_file is not None:
        Path(save_file).touch()

    try:
        for i in range(len(ds)):
            key = _read_item_key(ds, i)
            logger.info(f"Item {i} key: {key}")
            if save_file is not None:
                with open(save_file, "a") as f:
                    f.write(f"{key}\n")
    finally:
        temp_file.close()
        logger.info("Temporary config file deleted.")


def main():
    """
    Usage:
        ## read all keys
        python litdata_index.py read --data-dir /path/to/input/dir --save-file /path/to/save/file
    """

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title="subcommands", dest="subcommand")

    read_parser = subparser.add_parser("read", help="Read keys from dataset")
    read_parser.add_argument("-d", "--data-dir", type=str, required=True)
    read_parser.add_argument("-s", "--save-file", type=str, default=None)

    args = parser.parse_args()

    if args.subcommand == "read":
        read_keys(args.data_dir, args.save_file)


if __name__ == "__main__":
    main()
