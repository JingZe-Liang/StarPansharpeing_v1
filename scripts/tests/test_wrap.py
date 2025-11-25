import numpy as np
import torch
from torch.compiler import wrap_numpy


@torch.compile
@wrap_numpy
def any_numpy_fn(x):
    return np.linalg.inv(x)


def test_wrap_numpy():
    x = np.random.randn(16, 16)
    y = any_numpy_fn(x)
    return x, y


if __name__ == "__main__":
    x, result = test_wrap_numpy()
    print("Wrapped numpy function result:")
    print(result)

    non_wrap_result = np.linalg.inv(x)
    print("Non-wrapped numpy function result:")
    print(non_wrap_result)

    assert np.allclose(result.cpu().numpy(), non_wrap_result), "Wrapped and non-wrapped results do not match!"
    print("Test passed: Wrapped and non-wrapped results match.")
