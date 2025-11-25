import time
from contextlib import contextmanager

import psutil
import torch
from tqdm import trange


def get_memory_info():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print("-" * 30, "CPU Memory Info", "-" * 30)
    print(f"Used Mem: [{memory.used / (1024**3):.2f}/{memory.total / (1024**3):.2f}]GB")
    print(f"Mem usage: {memory.percent}%")
    print(f"Swap Mem: [{swap.used / (1024**3):.2f}/{swap.total / (1024**3):.2f}]GB")
    print("-" * 75)


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
            print(f"Memory allocated after forward pass: {allocated_memory / 1024**2:.2f} MB")
            print(f"Peak memory allocated: {peak_memory / 1024**2:.2f} MB")
            print(f"Memory usage: {memory_usage / 1024**2:.2f} MB")

            print(torch.cuda.memory_summary(device))

            return ret

        return wrapper

    return func_wrap


@contextmanager
def mem_context(device=None):
    """
    Context manager to monitor GPU memory usage during function execution.

    Args:
        device: The CUDA device to monitor (e.g., torch.device('cuda:0'))

    Usage:
        with memory_monitor(torch.device('cuda:0')):
            # Your code here
            result = model(input)
    """
    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device)

    try:
        yield
    finally:
        allocated_memory = torch.cuda.memory_allocated(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
        memory_usage = allocated_memory - initial_memory

        print(f"Initial memory allocated: {initial_memory / 1024**2:.2f} MB")
        print(f"Memory allocated after execution: {allocated_memory / 1024**2:.2f} MB")
        print(f"Peak memory allocated: {peak_memory / 1024**2:.2f} MB")
        print(f"Memory usage: {memory_usage / 1024**2:.2f} MB")

        print(torch.cuda.memory_summary(device))


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
def speed_test_context(test_num=100):
    """
    Context manager to test execution speed of code block.

    Args:
        test_num (int): Number of iterations to run the code block

    Usage:
        for _ in speed_test_context(100):
            # Your code to test here
            result = some_function(input)
            # Do something with result
    """
    start_time = time.time()

    class SpeedTestHelper:
        def __init__(self, num):
            self.num = num
            self.counter = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.counter < self.num:
                self.counter += 1
                return self
            else:
                raise StopIteration

    helper = SpeedTestHelper(test_num)
    yield helper

    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / test_num

    print(f"Code block executed {test_num} times.")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per execution: {average_time:.4f} seconds")


@contextmanager
def timer():
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
