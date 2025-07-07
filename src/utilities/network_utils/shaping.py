import inspect
from functools import wraps
from typing import Callable

import numpy as np
import torch
from einops import rearrange, reduce, repeat


def get_einops_op(op):
    if callable(op):
        return op
    elif isinstance(op, str):
        ops = {"rearrange": rearrange, "reduce": reduce, "repeat": repeat}
        if op not in ops:
            raise ValueError(
                f"Unsupported operation: {op}. Supported: {list(ops.keys())}"
            )
        return ops[op]
    else:
        raise TypeError(f"Operation must be a string or callable, got {type(op)}")


type ReshapeTyped = str | list[str | None] | dict[str, str | None]
type OpTyped = str | list[str | Callable] | dict[str, str | Callable] | Callable


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
                            bound_args.arguments[inp_key] = op_func(
                                inp_value, inp_reshp
                            )
                    else:
                        raise KeyError(
                            f"Input key '{inp_key}' for reshaping not found in function signature or call."
                        )
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
                            if (
                                i < len(output_reshping)
                                and (out_reshp := output_reshping[i]) is not None
                            ):
                                if isinstance(output_op, list):
                                    op = (
                                        output_op[i]
                                        if i < len(output_op)
                                        else (
                                            output_op[0] if output_op else "rearrange"
                                        )
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
                        for i, (out_reshp, (out_k, out_v)) in enumerate(
                            zip(output_reshping, output.items())
                        ):
                            assert isinstance(out_v, (torch.Tensor, np.ndarray)), (
                                f"Output {out_k} must be a torch.Tensor or a numpy.ndarray, got {type(out_v)}"
                            )
                            if out_reshp is not None:
                                if isinstance(output_op, list):
                                    op = (
                                        output_op[i]
                                        if i < len(output_op)
                                        else (
                                            output_op[0] if output_op else "rearrange"
                                        )
                                    )
                                else:
                                    op = output_op if output_op else "rearrange"
                                op_func = get_einops_op(op)
                                output[out_k] = op_func(out_v, out_reshp)
                        return output
                elif isinstance(output_reshping, dict):
                    for out_key, out_reshp in output_reshping.items():
                        if out_reshp is not None:
                            assert out_key in output, (
                                f"Output key {out_key} not found in output"
                            )
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


if __name__ == "__main__":
    from functools import partial

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
