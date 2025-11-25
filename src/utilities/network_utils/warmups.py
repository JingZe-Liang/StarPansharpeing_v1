class ValueWarmups:
    def __init__(self, start_value: float, end_value: float, steps: int):
        self.start_value = start_value
        self.end_value = end_value
        self.steps = steps
        self.current_value = self.start_value

    def __call__(self, step: int):
        if step < self.steps:
            self.current_value = self.start_value + (self.end_value - self.start_value) * (step / self.steps)
        else:
            self.current_value = self.end_value
        return self.current_value


if __name__ == "__main__":
    v = ValueWarmups(0.1, 1.0, 10)
    print(v(12))
