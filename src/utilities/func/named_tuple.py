from collections import namedtuple
from typing import NamedTuple


def get_named_tuple(name: str, *fields) -> type:
    """Create a namedtuple class with the given name and fields.

    Args:
        name: The name of the namedtuple class
        *fields: Variable length argument list of field names

    Returns:
        A namedtuple class type

    Example:
        >>> Point = get_named_tuple('Point', 'x', 'y')
        >>> p = Point(1, 2)
        >>> print(p.x, p.y)  # Output: 1 2
    """
    return namedtuple(name, fields, defaults=(None,) * len(fields))


def create_named_tuple(name: str, **kwargs) -> NamedTuple:
    """Create a namedtuple instance with the given name and key-value pairs.

    Args:
        name: The name of the namedtuple class
        **kwargs: Keyword arguments where keys become field names and values become field values

    Returns:
        A namedtuple instance

    Example:
        >>> person = create_named_tuple('Person', name='Alice', age=25)
        >>> print(person.name, person.age)  # Output: Alice 25
    """
    named_tuple_cls = get_named_tuple(name, *kwargs.keys())
    ins = named_tuple_cls(*kwargs.values())
    return ins
