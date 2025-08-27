from copy import deepcopy
from typing import Any, Literal, Mapping

import accelerate
import torch
import torch.distributed as dist
from torchmetrics.aggregation import (
    MaxMetric,
    MeanMetric,
    MinMetric,
    RunningMean,
    RunningSum,
)
from torchmetrics.metric import Metric

from src.utilities.logging.print import log_print


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        call_kwargs = kwargs.copy()
        force_new = call_kwargs.pop("__force_new_instance__", False)

        if force_new:
            instance = super().__call__(*args, **call_kwargs)
            # overload the instance to the class
            cls._instances[cls] = instance
            return instance

        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **call_kwargs)

        return cls._instances[cls]


class MultiObjectMeta(type):
    _instances: dict[int | str | type, object] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        call_kwargs = kwargs.copy()
        force_new = call_kwargs.pop("__force_new_instance__", False)
        less_than_n_instances = call_kwargs.pop("__less_than_n_instances__", 10)
        instance_name = call_kwargs.pop("__instance_name__", None)

        if len(cls._instances) == 0:
            # if no instances, create the first instance
            instance = super().__call__(*args, **call_kwargs)
            if instance_name is not None:
                log_print(
                    f"Given {instance_name=} as the instance key, but not initialized yet, "
                    "using class as the default key.",
                    "warning",
                )
            new_instance_key = cls  # default instance uses cls as the key
            cls._instances[new_instance_key] = instance
            return instance

        if not force_new:
            if instance_name is not None:
                # give instance_name, return the instance with that name
                assert instance_name in cls._instances, (
                    f"Instance with name {instance_name} not found in {cls.__name__}."
                )
                return cls._instances[instance_name]
            else:
                # no instance_name, return the default instance (cls as the key)
                return cls._instances[cls]
        else:
            # force new and instance_name is given, use instance_name as the key to
            # create the new instance
            # else not give the instance_name, use the id of the instance as the key
            instance = super().__call__(*args, **call_kwargs)
            new_instance_key = id(instance) if instance_name is None else instance_name
            cls._instances[new_instance_key] = instance
            if len(cls._instances) > less_than_n_instances:
                raise ValueError(
                    f"Too many instances of {cls.__name__} created: {len(cls._instances)}. "
                    f"Limit is {less_than_n_instances}."
                )
            return instance


class StepsCounter(metaclass=SingletonMeta):
    _initialized = False

    def __init__(
        self,
        step_names: list[str] | None = None,
        **__meta_kwargs,  # This is to allow for future extensions without breaking the interface
    ):
        if StepsCounter._initialized:
            log_print(
                "StepsCounter is a singleton. Attempting to re-initialize. "
                "New step_names will be added to the existing instance.",
                warn_once=True,
            )
            if step_names is None:
                return

            for name in step_names:
                if name not in self.step_names:
                    self.step_names.append(name)
                    setattr(self, f"n_{name}_steps", 0)
            return
        elif step_names is None:
            raise ValueError(
                "step_names must be provided for the first initialization of StepsCounter."
            )

        assert step_names is not None, (
            "step_names cannot be None when first initializing StepsCounter."
        )
        self.step_names = list(step_names)
        for name in self.step_names:
            setattr(self, f"n_{name}_steps", 0)
        StepsCounter._initialized = True

    def __repr__(self):
        return f"StepsCounter({self.state_dict()})"

    def state_dict(self):
        return {
            f"n_{name}_steps": getattr(self, f"n_{name}_steps")
            for name in self.step_names
        }

    def load_state_dict(self, state_dict):
        for name in self.step_names:
            n_ = f"n_{name}_steps"
            assert n_ in state_dict, f"Key {n_} missing in state_dict"
            setattr(self, n_, state_dict[n_])

    def update(self, name: str, update_n: int = 1):
        n_step = self.get(name)
        setattr(self, f"n_{name}_steps", n_step + update_n)

    def get(self, name: str):
        step_name = f"n_{name}_steps"
        assert hasattr(self, step_name), f"Key {step_name} missing in state_dict"

        return getattr(self, step_name)

    def __getitem__(self, name: str):
        return self.get(name)

    def __setitem__(self, name: str, value: int):
        name_set = f"n_{name}_steps"
        assert hasattr(self, name_set), f"Key {name_set} missing in state_dict"
        setattr(self, name_set, value)


class LossMetricTracker(metaclass=MultiObjectMeta):
    def clear_all(self, clear_name: bool = False):
        if not clear_name:
            self.loss_metrics_values = {}
            self.loss_metrics_tracked = {}
        else:
            self.loss_metrics_values = {
                name: None for name in self.loss_metrics_values.keys()
            }
            self.loss_metrics_tracked = {
                name: [] for name in self.loss_metrics_tracked.keys()
            }

    def clear_values(self, name: str | list | None = None, clear_name: bool = False):
        if name is None:
            self.loss_metrics_values = (
                {} if clear_name else {n: None for n in self.loss_metrics_values.keys()}
            )
        elif isinstance(name, str):
            if name in self.loss_metrics_values:
                self.loss_metrics_values[name] = None
        elif isinstance(name, list):
            for n in name:
                if n in self.loss_metrics_values:
                    self.loss_metrics_values[n] = None

    def clear_tracked(self, name: str | list | None = None, clear_name: bool = False):
        if name is None:
            self.loss_metrics_tracked = (
                {} if clear_name else {n: [] for n in self.loss_metrics_tracked.keys()}
            )
        elif isinstance(name, str):
            if name in self.loss_metrics_tracked:
                self.loss_metrics_tracked[name] = []
        elif isinstance(name, list):
            for n in name:
                if n in self.loss_metrics_tracked:
                    self.loss_metrics_tracked[n] = []

    def __init__(
        self,
        loss_metrics_values: dict[str, float] | None = None,
        loss_metrics_tracked: dict[str, list[float] | float] | None = None,
        **__meta_kwargs,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            if not loss_metrics_values and not loss_metrics_tracked:
                return
            else:
                if loss_metrics_values is not None:
                    for name, value in loss_metrics_values.items():
                        self.add_value(name, value)

                if loss_metrics_tracked is not None:
                    for name, values in loss_metrics_tracked.items():
                        self.add_tracked(name, values)

        elif not loss_metrics_values and not loss_metrics_tracked:
            raise ValueError(
                "Loss metrics must be provided for the first initialization of LossMetricTracker."
            )

        self.loss_metrics_values: dict[str, float | None]
        self.loss_metrics_tracked: dict[str, list[float] | None]

        if not hasattr(self, "loss_metrics_values"):
            self.loss_metrics_values = {}
        if not hasattr(self, "loss_metrics_tracked"):
            self.loss_metrics_tracked = {}

        for name, value in (loss_metrics_values or {}).items():
            self.add_value(name, value)

        for name, values in (loss_metrics_tracked or {}).items():
            self.add_tracked(name, values)

        self._initialized = True

    def add_value(self, name: str, value: float, ema: float = 1.0):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            value = value.item()
        elif not isinstance(value, float):
            raise ValueError(f"Expected float for {name}, got {type(value)}")

        if (
            name in self.loss_metrics_values
            and self.loss_metrics_values[name] is not None
        ):
            self.loss_metrics_values[name] = (
                self.loss_metrics_values[name] * (1 - ema) + value * ema  # type: ignore
            )
        else:
            self.loss_metrics_values[name] = value

    def add_tracked(self, name: str, value: float | list[float]):
        if not isinstance(value, (float, list)):
            raise ValueError(f"Expected float or list for {name}, got {type(value)}")
        elif isinstance(value, float):
            value = [value]
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            assert value.ndim == 1, (
                f"{name=} with {value} should be 1-d array or tensor"
            )
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            value = value.tolist()

        if name not in self.loss_metrics_tracked:
            self.loss_metrics_tracked[name] = value
        else:
            self.loss_metrics_tracked[name].extend(value)  # type: ignore

    def round_value(self, value: float | list[float] | None, decimals: int):
        if isinstance(value, list):
            vs = []
            for v in value:
                if isinstance(v, float):
                    vs.append(round(v, decimals))
                elif v is None:
                    vs.append(None)

            return vs
        elif isinstance(value, float):
            return round(value, decimals)
        elif value is None:
            return None
        else:
            raise ValueError(
                f"Unknown type for {value} in loss_metrics: {self.loss_metrics_values}"
            )

    def round(self, names: list[str] | None = None, decimals: int | None = 4):
        if names is None:
            names = list(self.loss_metrics_values.keys()) + list(
                self.loss_metrics_tracked.keys()
            )

        loss_metrics_values = {}
        loss_metrics_tracked = {}

        for name in names:
            if name in self.loss_metrics_values:
                loss_metrics_values[name] = (
                    self.round_value(self.loss_metrics_values[name], decimals)
                    if decimals is not None
                    else self.loss_metrics_values[name]
                )
            if name in self.loss_metrics_tracked:
                loss_metrics_tracked[name] = (
                    self.round_value(self.loss_metrics_tracked[name], decimals)
                    if decimals is not None
                    else self.loss_metrics_tracked[name]
                )

        return loss_metrics_values, loss_metrics_tracked

    def track(self, name: str, value: float | list[float]):
        if not isinstance(value, list):
            value = [value]

        if name not in self.loss_metrics_values:
            self.loss_metrics_tracked[name] = value
        else:  # name in self.loss_metrics_tracked
            if isinstance(self.loss_metrics_tracked[name], list):
                self.loss_metrics_tracked[name].extend(value)  # type: ignore
            elif self.loss_metrics_tracked.get(name, None) is None:
                self.loss_metrics_tracked[name] = value
            else:
                raise ValueError(
                    f"Expected list for {name} in loss_metrics_tracked, got {type(self.loss_metrics_tracked[name])}"
                )

    def get_tracked_values_op(
        self,
        name: str | list[str] | None,
        round_decimals: int | None = 4,
        track_value_op: Literal["max", "min", "mean", "all"] = "mean",
        none_if_not_found: bool = True,
    ):
        if name is None:
            name = list(self.loss_metrics_tracked.keys())
        elif isinstance(name, str):
            name = [name]

        def op_tracked(lst: list | None, op: str):
            if lst is None or len(lst) == 0:
                return None
            if op == "max":
                return max(lst)
            elif op == "min":
                return min(lst)
            elif op == "mean":
                return sum(lst) / len(lst)
            elif op == "all":
                return lst
            else:
                raise ValueError(
                    f"Unknown operation {op} for tracked values."
                    f"Supported operations: max, min, mean, all"
                )

        tracked_values_oped = {}
        for n in name:
            if n in self.loss_metrics_tracked:
                tracked_oped = op_tracked(self.loss_metrics_tracked[n], track_value_op)
                tracked_values_oped[n] = (
                    self.round_value(tracked_oped, round_decimals)
                    if round_decimals is not None
                    else tracked_oped
                )
            elif none_if_not_found:
                log_print(f"Tracked values for {n} not found", "warning")
                tracked_values_oped[n] = None

        return tracked_values_oped

    def get_value(
        self,
        name: str | list[str] | None,
        round_decimals: int | None = None,
        none_if_not_found: bool = False,
    ):
        if isinstance(name, str):
            name = [name]
        elif name is None:
            name = list(self.loss_metrics_values.keys())

        loss_metrics_values, _ = self.round(name, round_decimals)

        loss_metrics_values_ret = {}
        for n in name:
            if n not in loss_metrics_values and none_if_not_found:
                log_print(f"Loss metric value for {n} not found", "warning")
                loss_metrics_values_ret[n] = None
            else:
                loss_metrics_values_ret[n] = loss_metrics_values[n]

        return loss_metrics_values_ret

    def get(
        self,
        name: str | list[str] | None,
        round_decimals: int | None = None,
        track_value_op: Literal["max", "min", "mean"] = "mean",
        *,
        none_if_not_found: bool = False,
    ):
        if isinstance(name, str):
            name_values = [name]
            name_tracked = [name]
        elif name is None:
            name_values = list(self.loss_metrics_values.keys())
            name_tracked = list(self.loss_metrics_tracked.keys())
        else:
            name_values = name
            name_tracked = name

        loss_metrics_values = self.get_value(
            name_values,
            round_decimals=round_decimals,
            none_if_not_found=none_if_not_found,
        )
        loss_metrics_tracked = self.get_tracked_values_op(
            name_tracked,
            round_decimals=round_decimals,
            track_value_op=track_value_op,
            none_if_not_found=none_if_not_found,
        )

        return loss_metrics_values, loss_metrics_tracked

    def get_and_clear(self):
        values_ret = self.loss_metrics_values.copy()
        tracked_ret = self.loss_metrics_tracked.copy()
        self.clear_all(clear_name=False)

        return values_ret, tracked_ret

    def to_string(
        self,
        value_name: str | list[str] | None = None,
        tracked_name: str | list[str] | None = None,
        round_decimals: int | None = None,
        track_value_op: Literal["max", "min", "mean", "all"] = "mean",
        default_val: float | None = 0.0,
        sep=" - ",
        with_color: bool = False,
    ):
        """
        Returns a string representation of the tracked values for the given name(s).
        """
        if value_name is None:
            value_name = list(self.loss_metrics_values.keys())
        elif isinstance(value_name, str):
            value_name = [value_name]
        if tracked_name is None:
            tracked_name = list(self.loss_metrics_tracked.keys())
        elif isinstance(tracked_name, str):
            tracked_name = [tracked_name]

        loss_metrics_values = self.get_value(
            name=value_name,
            round_decimals=round_decimals,
            none_if_not_found=False,
        )
        loss_metrics_tracked = self.get_tracked_values_op(
            name=tracked_name,
            round_decimals=round_decimals,
            track_value_op=track_value_op,
            none_if_not_found=False,
        )

        with_color_fn = lambda n: f"\033[1;34m{n}\033[0m" if with_color else str(n)

        values_str = []
        tracked_str = []
        for n, v in loss_metrics_values.items():
            values_str.append(
                f"{with_color_fn(n)}: {v if v is not None else default_val}"
            )
        for n, v in loss_metrics_tracked.items():
            tracked_str.append(
                f"{with_color_fn(n)}: {v if v is not None else default_val}"
            )

        return sep.join(values_str), sep.join(tracked_str)

    def __repr__(self):
        return (
            f"LossMetricTracker(loss_metrics_values={self.loss_metrics_values}, "
            f"loss_metrics_tracked={self.loss_metrics_tracked})"
        )

    def state_dict(self):
        return {
            "loss_metrics_values": self.loss_metrics_values,
            "loss_metrics_tracked": self.loss_metrics_tracked,
        }

    def load_state_dict(self, state_dict):
        self.loss_metrics_values = state_dict["loss_metrics_values"]
        self.loss_metrics_tracked = state_dict["loss_metrics_tracked"]

    @classmethod
    def sync_state(
        cls,
        instance_name: str | int | None = None,
        overwrite_instance_use_sync: bool = False,
    ):
        # Get the existing instance
        instance: LossMetricTracker = cls(
            __force_new_instance__=False, __instance_name__=instance_name
        )

        # Sync
        if hasattr(torch, "distributed") and torch.distributed.is_initialized():
            instance_cp = deepcopy(instance)
            ins_lst: list[LossMetricTracker] = [
                None
            ] * torch.distributed.get_world_size()  # type: ignore
            torch.distributed.all_gather_object(ins_lst, instance)

            # mean all values and tracked

            # for the values
            for name in instance.loss_metrics_values:
                vs = []
                for ins in ins_lst:
                    v = ins.loss_metrics_values.get(name, None)
                    assert v is not None, (
                        f"Value for {name} is None in instance {ins}, this should not when syncronizing."
                    )
                    vs.append(v)
                instance_cp.loss_metrics_values[name] = sum(vs) / len(vs)

            # for the tracked
            for name in instance.loss_metrics_tracked:
                tracked: list[list[float]] = []
                for ins in ins_lst:
                    v: list[float] | None = ins.loss_metrics_tracked.get(name, None)
                    assert v is not None, (
                        f"Tracked values for {name} is None in instance {ins}, this should not when syncronizing."
                    )
                    tracked.append(v)

                # mean the list of list of float, return the list of float
                # [
                #  rank1: [0.1, 0.2, 0.3],
                #  rank2: [0.4, 0.5, 0.6],
                # ]
                # -> return:
                #         [0.25, 0.35, 0.45]
                instance_cp.loss_metrics_tracked[name] = [
                    sum(values) / len(values) for values in zip(*tracked)
                ]

            if overwrite_instance_use_sync:
                # overwrite the instance with the synchronized instance
                cls._instances[instance_name] = instance_cp
                log_print(
                    f"LossMetricTracker instance {instance_name} synchronized and overwritten.",
                )
            return instance_cp
        else:
            log_print(
                "Torch distributed is not initialized. Skipping sync_state.", "debug"
            )
            return instance

    @classmethod
    def remove_instance(cls, instance_name: str | int | type | None = None):
        """
        Remove the instance with the given name.
        If instance_name is None, remove the default instance.
        """
        if instance_name is None:
            instance_name = cls

        if instance_name in cls._instances:
            del cls._instances[instance_name]
            log_print(f"Instance {instance_name} removed from {cls.__name__}.", "info")
        else:
            log_print(
                f"Instance {instance_name} not found in {cls.__name__}.", "warning"
            )


# * --- Utilities --- * #


def metrics_sync(
    metrics: Mapping[str, Metric | torch.Tensor | float | list[float]],
    output_tensor_dict=False,
    reduce_op: str = "AVG",
    **metric_fn_kwargs,
):
    # Create a new dictionary to store synced metrics to avoid modifying the input
    synced_metrics = {}

    for name, metric in metrics.items():
        assert isinstance(metric, (Metric, torch.Tensor)), (
            f"Expected Metric or Tensor for {name}, got {type(metric)}"
        )
        if isinstance(metric, Metric):
            synced_metrics[name] = metric
        else:
            op2metric_cls = {
                "AVG": MeanMetric,
                "MAX": MaxMetric,
                "MIN": MinMetric,
                "RUN_MEAN": RunningMean,
                "RUN_SUM": RunningSum,
            }
            device = metric.device if isinstance(metric, torch.Tensor) else "cpu"
            metric_ = op2metric_cls[reduce_op](**metric_fn_kwargs).to(device)
            metric_.update(torch.as_tensor(metric).to(device))
            synced_metrics[name] = metric_

    if not output_tensor_dict:
        return synced_metrics

    output_dict = {}
    for name, metric in synced_metrics.items():
        if isinstance(metric, Metric):
            # will be sync in compute
            output_dict[name] = metric.compute()
        else:
            raise RuntimeError(f"Expected Metric type for {name}, got {type(metric)}")

    return output_dict


def dict_tensor_sync(
    metrics: dict[str, torch.Tensor | float],
    use_reduce=True,  # be sure that all should be Tensor
    *,
    reduce_op: dist.ReduceOp | None | str = "AVG",
):
    def _reduce_syn():
        str2reduce_op = {
            "AVG": dist.ReduceOp.AVG,
            "SUM": dist.ReduceOp.SUM,
            "MAX": dist.ReduceOp.MAX,
            "MIN": dist.ReduceOp.MIN,
        }
        synced_metrics = {}
        for name, value in metrics.items():
            tensor_ = torch.as_tensor(value).clone().detach()
            # Ensure tensor is of type float for reduction operations
            if tensor_.dtype not in (
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.float16,
            ):
                tensor_ = tensor_.float()
            op = str2reduce_op[reduce_op] if isinstance(reduce_op, str) else reduce_op
            assert op is not None, "Expected a valid reduce operation"
            dist.all_reduce(tensor_, op=op)
            synced_metrics[name] = tensor_
        return synced_metrics

    def _reduce_obj_syn():
        lst_metrics: list[dict] = [None] * dist.get_world_size()  # type: ignore
        dist.all_gather_object(lst_metrics, metrics)
        if reduce_op == "AVG":
            return {
                name: sum(d[name] for d in lst_metrics) / len(lst_metrics)
                for name in lst_metrics[0]
            }
        elif reduce_op == "MAX":
            return {name: max(d[name] for d in lst_metrics) for name in lst_metrics[0]}
        elif reduce_op == "MIN":
            return {name: min(d[name] for d in lst_metrics) for name in lst_metrics[0]}
        else:
            raise ValueError(f"Unknown reduce operation: {reduce_op}")

    if dist.is_initialized():
        # Create a new dictionary to store synced metrics to avoid modifying the input
        if use_reduce:
            return _reduce_syn()
        else:
            return _reduce_obj_syn()

    return metrics


# * --- test --- #


def _test_metrics_sync(rank):
    # 0, 1 -> 0.5
    # 1, 2 -> 1.5
    metrics = {"a": torch.tensor(rank), "b": torch.tensor(rank + 1)}
    print(metrics_sync(metrics, output_tensor_dict=True))


def _test_dict_tensor_sync(rank):
    a = torch.tensor(float(rank))
    b = torch.tensor(float(rank + 1))
    metrics = {"a": a, "b": b}
    print(dict_tensor_sync(metrics, use_reduce=True))


def _test_track_mp_sync(rank: int):
    lmt = LossMetricTracker(
        loss_metrics_values={
            "loss": 1.0 + rank,
        },
        loss_metrics_tracked={
            "loss": [rank + 0.0, rank + 1.0, rank + 2.0],
        },
    )

    # rank0: [1], [loss: [0.0, 1.0, 2.0]]
    # rank1: [2], [loss: [1.0, 2.0, 3.0]]
    # sync: [1.5], [loss: [0.5, 1.5, 2.5]]

    print("rank", rank, "- LossMetricTracker state before sync:", lmt)

    lmt_sync = LossMetricTracker.sync_state(
        overwrite_instance_use_sync=True,
    )
    print(f"Rank {rank} - LossMetricTracker state after sync: {lmt_sync}")


if __name__ == "__main__":
    accelerator = accelerate.Accelerator()
    rank = accelerator.process_index
    print("Rank:", rank)

    ## > Test 1: _test_track_mp_sync, must start in __main__
    # _test_track_mp_sync(rank)

    ## > Test 2: _test_metrics_sync
    # _test_metrics_sync(rank)

    ## > Test 3: _test_dict_tensor_sync
    _test_dict_tensor_sync(rank)
