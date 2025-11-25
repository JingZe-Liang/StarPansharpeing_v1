import inspect
import profile
import re
from collections.abc import Generator, Iterable
from typing import Sequence, Union, cast

import torch
import torch.nn as nn
from loguru import logger
from torch.autograd import profiler


def filter_params_by_name(params: dict[str, nn.Parameter], regex: str):
    pattern = re.compile(regex)
    return {name: param for name, param in params.items() if pattern.search(name)}


@profiler.record_function("filter_no_wds_into_optim_groups")  # type: ignore
def filter_no_wds_into_optim_groups(
    params: Iterable[tuple[str, nn.Parameter]]
    | dict[str, nn.Parameter]
    | Generator[tuple[str, nn.Parameter], None, None],
    no_wd_regex: str | Sequence[str],
):
    """Filter out parameters that should not use weight decay into a separate group."""
    # Convert to list to avoid consuming iterators
    if inspect.isgenerator(params):
        params = {name: p for name, p in params if p.requires_grad}
    params = cast(dict[str, nn.Parameter], params)

    if isinstance(no_wd_regex, Sequence) and not isinstance(no_wd_regex, str):
        no_wd_regex = r"|".join(no_wd_regex)
    no_wd_named_params = filter_params_by_name(params, no_wd_regex)
    no_wd_param_ids = {id(p) for p in no_wd_named_params.values()}
    wd_params = [p for _, p in params.items() if id(p) not in no_wd_param_ids]

    logger.debug(f"filter out no wd params: {no_wd_named_params.keys()}")
    logger.debug(f"ilter out wd params: {len(wd_params)}")

    assert len(wd_params) + len(no_wd_named_params) == len(params), "Parameter count mismatch after filtering."

    return [
        {"params": wd_params},  # with weight decay
        {
            "params": list(no_wd_named_params.values()),
            "weight_decay": 0.0,
        },  # without weight decay
    ]


if __name__ == "__main__":
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        net = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10), nn.Linear(10, 10), nn.Conv2d(10, 10, 3))

        optim_groups = filter_no_wds_into_optim_groups(net.named_parameters(), ["bias", "1"])

        for g in optim_groups:
            print(len(g["params"]), g.get("weight_decay", "default wd"))

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
