from einops import pack, rearrange, repeat, unpack


def pack_one(x, pattern):
    x, ps = pack([x], pattern)
    return x, ps


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]
