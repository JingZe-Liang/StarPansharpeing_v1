import torch


def split_into_patches(x, patch_size=64):
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    return patches


def combine_patches(patches, B, C, H, W, patch_size=64):
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    patches = patches.view(B, num_patches_h * num_patches_w, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.contiguous().view(B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5)
    patches = patches.contiguous().view(B, C, H, W)
    return patches
