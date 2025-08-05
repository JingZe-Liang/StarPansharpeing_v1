import time
from contextlib import contextmanager

import torch
from tqdm import trange


def func_mem_wrapper(device):
    def func_wrap(func):
        def wrapper(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats(device)  # reset the peak memory stats
            initial_memory = torch.cuda.memory_allocated(device)

            ret = func(*args, **kwargs)

            allocated_memory = torch.cuda.memory_allocated(device)
            peak_memory = torch.cuda.max_memory_allocated(device)

            memory_usage = allocated_memory - initial_memory

            print(f"Initial memory allocated: {initial_memory / 1024**2:.2f} MB")
            print(
                f"Memory allocated after forward pass: {allocated_memory / 1024**2:.2f} MB"
            )
            print(f"Peak memory allocated: {peak_memory / 1024**2:.2f} MB")
            print(f"Memory usage: {memory_usage / 1024**2:.2f} MB")

            print(torch.cuda.memory_summary(device))

            return ret

        return wrapper

    return func_wrap


def func_speed_wrapper(test_num=100):
    def inner_func_wrapper(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            for _ in trange(test_num):
                ret = func(*args, **kwargs)

            end_time = time.time()
            total_time = end_time - start_time
            average_time = total_time / test_num

            print(f"Function {func.__name__} executed {test_num} times.")
            print(f"Total time: {total_time:.4f} seconds")
            print(f"Average time per execution: {average_time:.4f} seconds")

            return ret

        return wrapper

    return inner_func_wrapper


@contextmanager
def timer():
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
