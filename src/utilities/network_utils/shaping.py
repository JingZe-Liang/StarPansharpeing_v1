import inspect
from functools import lru_cache, wraps
from typing import Callable

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from numpy.typing import NDArray
from torch import Tensor

type ReshapeTyped = str | list[str | None] | dict[str, str | None]
type OpTyped = str | list[str | Callable] | dict[str, str | Callable] | Callable


def get_einops_op(op: str | Callable) -> Callable:
    """
    Get the einops operation function from string or callable.

    Args:
        op: Operation name ("rearrange", "reduce", "repeat") or callable function

    Returns:
        Callable: The corresponding einops operation function

    Raises:
        ValueError: If operation name is not supported
        TypeError: If operation is neither string nor callable
    """
    if callable(op):
        return op
    elif isinstance(op, str):
        ops = {"rearrange": rearrange, "reduce": reduce, "repeat": repeat}
        if op not in ops:
            raise ValueError(f"Unsupported operation: {op}. Supported: {list(ops.keys())}")
        return ops[op]
    else:
        raise TypeError(f"Operation must be a string or callable, got {type(op)}")


# decorator
def reshape_wrapper(
    input_reshping: ReshapeTyped,
    input_op: OpTyped = "rearrange",
    output_reshping: str | list[str] | dict[str, str] | None = None,
    output_op: ReshapeTyped | None = None,
):
    """
    Reshape the input tensor according to the specified input and output shapes.
    If reverse_shape is True, it will reshape from out_shape to input_shape.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal input_reshping, output_reshping, input_op, output_op

            if isinstance(input_op, (str, Callable)):
                input_op = [input_op]
            if output_op is not None and isinstance(output_op, (str, Callable)):
                output_op = [output_op]

            # Reshaping the input
            if isinstance(input_reshping, str):
                input_reshping = [input_reshping]

            if isinstance(input_reshping, list):
                new_args = []
                for i, arg in enumerate(args):
                    if i < len(input_reshping) and input_reshping[i] is not None:
                        if isinstance(input_op, list):
                            op = input_op[i] if i < len(input_op) else input_op[0]
                        else:
                            op = input_op
                        op_func = get_einops_op(op)
                        new_args.append(op_func(arg, input_reshping[i]))
                    else:
                        new_args.append(arg)
                output = func(*new_args, **kwargs)
            elif isinstance(input_reshping, dict):
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                for inp_key, inp_reshp in input_reshping.items():
                    if inp_key in bound_args.arguments:
                        if inp_reshp is not None:
                            inp_value = bound_args.arguments[inp_key]
                            assert isinstance(inp_value, (torch.Tensor, np.ndarray)), (
                                f"Input {inp_key} must be a torch.Tensor or np.ndarray, got {type(inp_value)}"
                            )

                            if isinstance(input_op, dict):
                                op = input_op.get(inp_key, "rearrange")
                            elif isinstance(input_op, list):
                                op = input_op[0]
                            else:
                                op = input_op

                            op_func = get_einops_op(op)
                            bound_args.arguments[inp_key] = op_func(inp_value, inp_reshp)
                    else:
                        raise KeyError(f"Input key '{inp_key}' for reshaping not found in function signature or call.")
                output = func(*bound_args.args, **bound_args.kwargs)
            else:
                raise TypeError(
                    f"input_reshping must be a str, list of str, or dict of str, got {type(input_reshping)}"
                )

            # Reshaping the output
            if output_reshping is not None:
                if isinstance(output_reshping, str):
                    output_reshping = [output_reshping]

                if isinstance(output_reshping, list):
                    if isinstance(output, tuple):
                        new_output = []
                        for i, out in enumerate(output):
                            if i < len(output_reshping) and (out_reshp := output_reshping[i]) is not None:
                                if isinstance(output_op, list):
                                    op = (
                                        output_op[i]
                                        if i < len(output_op)
                                        else (output_op[0] if output_op else "rearrange")
                                    )
                                else:
                                    op = output_op if output_op else "rearrange"
                                op_func = get_einops_op(op)
                                new_output.append(op_func(out, out_reshp))
                            else:
                                new_output.append(out)
                        return tuple(new_output)
                    elif isinstance(output, (torch.Tensor, np.ndarray)):
                        op = (
                            output_op[0]
                            if isinstance(output_op, list) and output_op
                            else (output_op if output_op else "rearrange")
                        )
                        op_func = get_einops_op(op)
                        return op_func(output, output_reshping[0])
                    elif isinstance(output, dict):
                        for i, (out_reshp, (out_k, out_v)) in enumerate(zip(output_reshping, output.items())):
                            assert isinstance(out_v, (torch.Tensor, np.ndarray)), (
                                f"Output {out_k} must be a torch.Tensor or a numpy.ndarray, got {type(out_v)}"
                            )
                            if out_reshp is not None:
                                if isinstance(output_op, list):
                                    op = (
                                        output_op[i]
                                        if i < len(output_op)
                                        else (output_op[0] if output_op else "rearrange")
                                    )
                                else:
                                    op = output_op if output_op else "rearrange"
                                op_func = get_einops_op(op)
                                output[out_k] = op_func(out_v, out_reshp)
                        return output
                elif isinstance(output_reshping, dict):
                    for out_key, out_reshp in output_reshping.items():
                        if out_reshp is not None:
                            assert out_key in output, f"Output key {out_key} not found in output"
                            assert isinstance(
                                (out_value := output[out_key]),
                                (torch.Tensor, np.ndarray),
                            ), f"Output {out_key} must be a torch.Tensor or np.ndarray"

                            if isinstance(output_op, dict):
                                op = output_op.get(out_key, "rearrange")
                            elif isinstance(output_op, list):
                                op = output_op[0] if output_op else "rearrange"
                            else:
                                op = output_op if output_op else "rearrange"

                            op_func = get_einops_op(op)
                            output[out_key] = op_func(out_value, out_reshp)
                    return output
            else:
                return output

        return wrapper

    return decorator


@lru_cache(maxsize=16)
def get_reduce_einops_pattern(tensor_dim: int | list, reduce_dims: tuple[int, ...]) -> str:
    """
    Generate einops pattern for reducing specified dimensions.

    Args:
        tensor_dim: Number of dimensions (int) or sequence representing dimensions
        reduce_dims: Tuple of dimension indices to reduce (supports negative indexing)

    Returns:
        str: Einops pattern string (e.g., "a b c -> a 1 c")

    Examples:
        >>> get_reduce_einops_pattern(3, (1,))
        'a b c -> a 1 c'
        >>> get_reduce_einops_pattern(4, (-2, -1))
        'a b c d -> a b 1 1'
    """
    # tensor_dim: int or sequence; reduce_dims: tuple of dims to reduce
    dims = tensor_dim if isinstance(tensor_dim, int) else len(tensor_dim)
    # normalize negative dims and sort
    rd = sorted(d if d >= 0 else dims + d for d in reduce_dims)
    # generate a name for each axis, e.g. 'a','b','c',...
    axes = [chr(ord("a") + i) for i in range(dims)]
    # in the output pattern, replace reduced axes with '1'
    out = [("1" if i in rd else axes[i]) for i in range(dims)]
    return f"{' '.join(axes)} -> {' '.join(out)}"


@lru_cache(maxsize=16)
def get_flatten_einops_pattern(
    tensor_dim: int | list,
    flatten_dims: tuple[int, ...],
    f_dim_at: int = -1,
) -> tuple[str, dict[str, int]]:
    """
    Generate einops pattern for flattening specified dimensions.

    Args:
        tensor_dim: Number of dimensions (int) or sequence representing dimensions
        flatten_dims: Tuple of dimension indices to flatten (supports negative indexing)
        f_dim_at: Position to place the flattened dimension (supports negative indexing).
                 -1 means last position, 0 means first position, etc.

    Returns:
        tuple[str, dict[str, int]]:
            - Einops pattern string (e.g., "a b c d -> a (b c d)")
            - Dictionary mapping axis names to their dimension indices for size computation

    Examples:
        >>> get_flatten_einops_pattern(4, (1, 2, 3))
        ('a b c d -> a (b c d)', {'b': 1, 'c': 2, 'd': 3})
        >>> get_flatten_einops_pattern(4, (1, 2), f_dim_at=1)
        ('a b c d -> a (b c) d', {'b': 1, 'c': 2})
        >>> get_flatten_einops_pattern(4, (1, 2), f_dim_at=0)
        ('a b c d -> (b c) a d', {'b': 1, 'c': 2})
        >>> get_flatten_einops_pattern(3, (0, 1, 2))
        ('a b c -> (a b c)', {'a': 0, 'b': 1, 'c': 2})
    """
    # tensor_dim: int or sequence; flatten_dims: tuple of dims to flatten
    dims = tensor_dim if isinstance(tensor_dim, int) else len(tensor_dim)
    # normalize negative dims and sort
    fd = sorted(d if d >= 0 else dims + d for d in flatten_dims)
    # generate a name for each axis, e.g. 'a','b','c',...
    axes = [chr(ord("a") + i) for i in range(dims)]

    # group the axes to be flattened and the remaining axes
    flatten_axes = [axes[i] for i in fd]
    remaining_axes = [axes[i] for i in range(dims) if i not in fd]

    # construct the pattern
    if remaining_axes and flatten_axes:
        # if there are both remaining and flatten axes
        flatten_pattern = " ".join(flatten_axes)

        # determine where to place the flattened dimension
        n_remaining = len(remaining_axes)
        if f_dim_at < 0:
            f_dim_at = n_remaining + 1 + f_dim_at  # convert negative index

        # ensure f_dim_at is in valid range
        f_dim_at = max(0, min(f_dim_at, n_remaining))

        # insert flattened dimension at specified position
        output_axes = remaining_axes.copy()
        output_axes.insert(f_dim_at, f"({flatten_pattern})")

        # create dictionary mapping axis names to dimension indices
        flatten_dim_dict = {axis: i for i, axis in enumerate(axes) if axis in flatten_axes}

        return f"{' '.join(axes)} -> {' '.join(output_axes)}", flatten_dim_dict
    elif remaining_axes:
        # if only remaining axes (no flattening needed)
        return f"{' '.join(axes)} -> {' '.join(remaining_axes)}", {}
    elif flatten_axes:
        # if only flatten axes (flatten to 1D)
        flatten_dim_dict = {axis: i for i, axis in enumerate(axes) if axis in flatten_axes}
        return f"{' '.join(axes)} -> ({' '.join(flatten_axes)})", flatten_dim_dict
    else:
        # edge case: no axes specified
        return f"{' '.join(axes)} -> {' '.join(axes)}", {}


def reverse_einops_pattern(pattern: str):
    # 'a b c -> a (b c)' -> 'a (b c) -> a b c'
    a, b = pattern.split("->")
    new_p = b.strip() + " -> " + a.strip()
    return new_p


def reduce_any(reduce_op: str, x: NDArray | Tensor, op_dim: tuple[int, ...]):
    p = get_reduce_einops_pattern(x.ndim, op_dim)
    return reduce(x, p, reduction=reduce_op)


def flatten_any(x: NDArray | Tensor, op_dim: tuple[int, ...], f_dim_at: int = -1):
    p, dim_dict = get_flatten_einops_pattern(x.ndim, op_dim, f_dim_at)
    # Calculate the actual flattened dimension size
    flatten_size = 1
    for axis, dim_idx in dim_dict.items():
        flatten_size *= x.shape[dim_idx]

    # Create size dict with actual dimension sizes
    size_dict = {axis: x.shape[dim_idx] for axis, dim_idx in dim_dict.items()}
    size_dict["flatten_size"] = flatten_size

    return rearrange(x, p), size_dict


if __name__ == "__main__":
    from functools import partial

    # Test the new get_flatten_einops_pattern function
    print("=== Testing get_flatten_einops_pattern ===")

    # Test case 1: 4D tensor, flatten last 3 dimensions
    pattern, dim_dict = get_flatten_einops_pattern(4, (1, 2, 3))
    print(f"4D tensor, flatten dims (1,2,3):")
    print(f"  Pattern: {pattern}")
    print(f"  Dim dict: {dim_dict}")
    print()

    # Test case 2: 4D tensor, flatten middle 2 dimensions at position 1
    pattern, dim_dict = get_flatten_einops_pattern(4, (1, 2), f_dim_at=1)
    print(f"4D tensor, flatten dims (1,2) at position 1:")
    print(f"  Pattern: {pattern}")
    print(f"  Dim dict: {dim_dict}")
    print()

    # Test case 3: 3D tensor, flatten all dimensions
    pattern, dim_dict = get_flatten_einops_pattern(3, (0, 1, 2))
    print(f"3D tensor, flatten all dims:")
    print(f"  Pattern: {pattern}")
    print(f"  Dim dict: {dim_dict}")
    print()

    # Test case 4: Test flatten_any function with actual tensor
    import torch

    x = torch.randn(2, 3, 4, 5)  # Shape: [batch, channels, height, width]

    # Test flatten_any with spatial dimensions (height, width) -> dims 2,3
    print(f"Original tensor shape: {x.shape}")
    result, size_dict = flatten_any(x, (2, 3), f_dim_at=2)
    print(f"Flatten spatial dims (2,3) at position 2:")
    print(f"  Result shape: {result.shape}")
    print(f"  Size dict: {size_dict}")
    print(f"  Expected flatten size: 4*5 = {4 * 5}, got: {size_dict['flatten_size']}")
    print()

    # Test case 5: Flatten channels and height
    result2, size_dict2 = flatten_any(x, (1, 2), f_dim_at=1)
    print(f"Flatten channels and height (1,2) at position 1:")
    print(f"  Result shape: {result2.shape}")
    print(f"  Size dict: {size_dict2}")
    print(f"  Expected flatten size: 3*4 = {3 * 4}, got: {size_dict2['flatten_size']}")
    print()

    print("✓ All tests passed!")

    # 示例1: 使用不同的输入操作符
    # @reshape_wrapper(
    #     input_reshping=["h w c -> c h w", "h w c -> c h w"],  # 两个输入都使用相同的变换
    #     input_op=["rearrange", "rearrange"],
    #     output_reshping=["c h w -> h w c", "c h w -> h w c"],
    #     output_op=["rearrange", "rearrange"],
    # )
    # def example_function1(x, y):
    #     z = x + y
    #     return z, z

    # # 示例2: 使用字典形式指定不同操作符
    # @reshape_wrapper(
    #     input_reshping={"x": "h w c -> c h w", "y": "b c -> c b"},
    #     input_op={"x": "rearrange", "y": "rearrange"},
    #     output_reshping={"result": "c h w -> h w c"},
    #     output_op={"result": "rearrange"},
    # )
    # def example_function2(x, y):
    #     return {"result": x}

    # # 示例3: 使用reduce操作与callable
    # @reshape_wrapper(
    #     input_reshping=["h w c -> c h w"],
    #     input_op=["rearrange"],
    #     output_reshping=["c h w -> h w"],
    #     output_op=[partial(reduce, reduction="mean")],  # 使用callable来指定reduce操作
    # )
    # def example_function3(x):
    #     return x

    # # 示例4: 混合使用字符串和callable
    # @reshape_wrapper(
    #     input_reshping=["h w c -> c h w", "c h w -> (c h) w"],
    #     input_op=["rearrange", partial(rearrange)],  # 混合使用
    #     output_reshping=["(c h) w -> c h w", "c h w -> h w"],
    #     output_op=["rearrange", partial(reduce, reduction="mean")],
    # )
    # def example_function4(x, y):
    #     z = x + y
    #     return z, z

    # x = torch.randn(256, 256, 3)
    # y = torch.randn(256, 256, 3)  # 修正y的形状以匹配x

    # result1, result2 = example_function1(x, y)
    # print(f"Result1 shape: {result1.shape}, Result2 shape: {result2.shape}")

    # # 演示reduce操作
    # result3 = example_function3(x)
    # print(f"Result3 shape: {result3.shape}")  # 应该是 [256, 256]

    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        @reshape_wrapper(
            input_reshping=[None, "b h w c -> b c h w"],
            output_reshping=["b c h w -> b h w c"],
        )
        def forward(self, x):
            print(x.shape)
            return x

    net = Network()
    x = torch.randn(2, 256, 256, 3)  # 输入形状为 [batch_size, height, width, channels]
    output = net(x)
    print(output.shape)
