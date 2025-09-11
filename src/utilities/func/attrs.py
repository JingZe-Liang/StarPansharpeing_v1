def get_attr_iter(obj: object, attr: str):
    for at in attr.split("."):
        if not hasattr(obj, at):
            raise AttributeError(f"Object {obj} has no attribute {at}")
        obj = getattr(obj, at)
    return obj
