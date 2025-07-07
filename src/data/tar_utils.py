import os
import re
import tarfile
from contextlib import contextmanager

import webdataset as wds

from src.utilities.logging import log_print


@contextmanager
def tar_sink_manager():
    """
    Context manager for writing to a tar file.
    """

    total_sinks = {}

    def get_sink(name, tar_path):
        if name not in total_sinks:
            total_sinks[name] = wds.TarWriter(tar_path)
            log_print(f"Created new tar sink for {tar_path}")
        return total_sinks[tar_path]

    try:
        yield get_sink

    finally:
        for sink in total_sinks.values():
            sink.close()
            log_print(f"Closed tar sink for {sink.name}")


class TarSinkManager:
    total_sinks = {}

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def get_sink(self, name: str, tar_rel_path: str):
        tar_path = os.path.join(self.base_dir, tar_rel_path)
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)

        if name not in self.total_sinks:
            self.total_sinks[name] = wds.TarWriter(tar_path)
            log_print(f"Created new tar sink for {tar_path}")
        return self.total_sinks[name]

    def close_all(self):
        for name, sink in self.total_sinks.items():
            sink.close()
            log_print(f"Closed tar sink for {name}")
        self.total_sinks.clear()
        log_print("Closed all tar sinks.")


def remove_key_in_tar(tar_path: str, key: str):
    """
    Remove specific key(s) (supporting regex) from a tar file.
    The original tar will be replaced after filtering.
    """
    from tqdm import tqdm

    assert tar_path.endswith(".tar"), "The tar_path must end with .tar"

    # Compile the regex pattern
    pattern = re.compile(key)
    tmp_path = tar_path + ".tmp"

    # breakpoint()
    with tarfile.open(tar_path, "r") as tar, tarfile.open(tmp_path, "w") as out_tar:
        for member in (tbar := tqdm(tar, desc="Filtering tar members")):
            if not pattern.search(member.name):
                out_tar.addfile(
                    member, tar.extractfile(member) if member.isfile() else None
                )
            else:
                tbar.set_description(f"Skipping {member.name}")
    # Replace original tar file
    import os

    ans = input("Press Enter to replace original tar file [y/n]: ")
    if ans.lower() != "y":
        print("Replacement cancelled.")
        return
    else:
        os.replace(tmp_path, tar_path)


if __name__ == "__main__":
    # Example usage
    tar_path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/LoveDA/conditions/conditions/LoveDA-3_bands-px_1024-0000.tar"
    key_to_remove = r".*\.rgb.png"  # Remove all .rgb files
    remove_key_in_tar(tar_path, key_to_remove)
    print(f"Removed keys matching '{key_to_remove}' from {tar_path}.")
