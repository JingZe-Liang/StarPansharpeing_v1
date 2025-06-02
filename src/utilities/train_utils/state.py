import warnings


class StepsCounter:
    _instance = None
    _initialized = False
    step_names = []

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(StepsCounter, cls).__new__(cls)
        return cls._instance

    def __init__(self, step_names: list[str] | None = None):
        if self._initialized:
            warnings.warn(
                "StepsCounter is a singleton. Attempting to re-initialize. "
                "New step_names will be added to the existing instance.",
                UserWarning,
            )
            if step_names is None:
                return

            for name in step_names:
                if name not in self.step_names:
                    self.step_names.append(name)
                    setattr(self, f"n_{name}_steps", 0)
            return
        elif step_names is None and len(self.step_names) == 0:
            raise ValueError(
                "step_names must be provided for the first initialization of StepsCounter."
            )

        # all set to 0
        self.step_names = list(
            step_names
        )  # Use list() to ensure it's a mutable list and a copy
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


if __name__ == "__main__":
    sc = StepsCounter(["train", "val"])

    sc.update("train", 5)
    print(sc["train"])

    sc2 = StepsCounter(["test"])
    sc2.update("test", 10)
    print(sc2["test"])

    sc2.update("train", 3)
    print(sc["train"])

    # accelerator = accelerate.Accelerator()

    # accelerator.save_model(sc, "./")
