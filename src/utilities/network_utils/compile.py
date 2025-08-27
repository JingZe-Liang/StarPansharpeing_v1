from functools import wraps

# * --- no compilation wrappers --- #


def null_decorator(**any_kwargs):
    def _inner_decorator(func):
        return func

    return _inner_decorator


def null_decorator_no_any_kwgs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
