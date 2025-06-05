import warnings
from typing import Literal

from src.utilities.logging.print import log_print


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class StepsCounter(metaclass=SingletonMeta):
    def __init__(self, step_names: list[str] | None = None):
        if hasattr(self, "_initialized") and self._initialized:
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
        self._initialized = True

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


class LossMetricTracker(metaclass=SingletonMeta):
    loss_metrics_values: dict[str, float | None] = {}
    loss_metrics_tracked: dict[str, list[float] | None] = {}

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
    ):
        if hasattr(self, "_initialized") and self._initialized:
            log_print(
                "LossMetricTracker is a singleton. Attempting to re-initialize. "
                "New loss metrics will be added to the existing instance.",
                warn_once=True,
            )

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
        if not isinstance(value, float):
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
                raise ValueError(f"Unknown operation {op} for tracked values")

        tracked_values_oped = {}
        for n in name:
            if n in self.loss_metrics_tracked:
                tracked_oped = op_tracked(self.loss_metrics_tracked[n], track_value_op)
                tracked_values_oped[n] = (
                    self.round_value(tracked_oped, round_decimals)
                    if round_decimals is not None
                    else tracked_oped
                )
            else:
                log_print(f"Tracked values for {n} not found", "warning")
                tracked_values_oped[n] = None

        return tracked_values_oped

    def get_value(
        self,
        name: str | list[str] | None,
        round_decimals: int | None = None,
    ):
        if isinstance(name, str):
            name = [name]
        elif name is None:
            name = list(self.loss_metrics_values.keys())

        loss_metrics_values, _ = self.round(name, round_decimals)

        loss_metrics_values_ret = {}
        for n in name:
            if n not in loss_metrics_values:
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

        loss_metrics_values = self.get_value(name_values, round_decimals=round_decimals)
        loss_metrics_tracked = self.get_tracked_values_op(
            name_tracked,
            round_decimals=round_decimals,
            track_value_op=track_value_op,
        )

        return loss_metrics_values, loss_metrics_tracked

    def get_and_clear(self):
        values_ret = self.loss_metrics_values.copy()
        tracked_ret = self.loss_metrics_tracked.copy()
        self.clear_all(clear_name=False)

        return values_ret, tracked_ret

    def __repr__(self):
        return (
            f"LossMetricTracker(loss_metrics_values={self.loss_metrics_values}, "
            f"loss_metrics_tracked={self.loss_metrics_tracked})"
        )


if __name__ == "__main__":
    # sc = StepsCounter(["train", "val"])

    # sc.update("train", 5)
    # print(sc["train"])

    # sc2 = StepsCounter(["test"])
    # sc2.update("test", 10)
    # print(sc2["test"])

    # sc2.update("train", 3)
    # print(sc["train"])

    lmt = LossMetricTracker(
        loss_metrics_values={"loss": 0.15, "accuracy": 0.12318, "name1": 0.1},
        loss_metrics_tracked={
            "loss": [0.521351, 0.4213141, 0.3],
            "accuracy": [0.8, 0.85],
            "val": 0.2,
        },
    )
    print(
        lmt.get_tracked_values_op(
            name=["loss", "val"], track_value_op="all", round_decimals=None
        )
    )
    print(lmt.get_value(name=["loss", "accuracy"], round_decimals=2))
    # lmt.clear_tracked(clear_name=True)
    # print(lmt)

    # lmt.clear_values(clear_name=True)
    # print(lmt)

    lmt2 = LossMetricTracker()
    lmt2.add_tracked("name1", [0.1, 0.2, 0.3])
    lmt2.add_value("name2", 0.5)

    print(lmt2)

    lmt2.add_value("name2", 1.0, 0.5)
    print(lmt2.get_value(name="name1", round_decimals=2))

    print(lmt2.get_and_clear())
    print(lmt2)
