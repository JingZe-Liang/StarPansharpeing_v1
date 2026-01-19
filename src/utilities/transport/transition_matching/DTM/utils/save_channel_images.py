import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision as tv
import numpy as np


# def save_channel_images(tensor, name_prefix, save_dir="./samples/"):
#     """
#     Save each channel of the tensor as a separate image.
#     """
#     n_channels = tensor.shape[1]  # 获取通道数
#     os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
#
#     for c in range(n_channels):
#         # 提取每个通道的图像
#         channel_img = tensor[:, c, :, :].cpu()
#
#         # 每个通道保存为独立的图像
#         for i in range(channel_img.shape[0]):
#             img = channel_img[i]  # 获取单个图像
#             save_path = os.path.join(save_dir, f"{name_prefix}_channel_{c}_img_{i}.png")
#             tv.utils.save_image(img, save_path)  # 保存为独立图像
#             print(f"Saved: {save_path}")


def save_channel_images(input_tensor, name, save_dir="./samples/"):
    """
    Save each image in the input tensor as a separate image without applying any gray scale.

    Parameters:
    - input_tensor: A tensor with shape (B, C, H, W) where B is batch size, C is number of channels,
                     H and W are height and width of the image.
    - name: The prefix for the saved image file names.
    - save_dir: Directory where the images will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over each image in the batch
    for batch_idx in range(input_tensor.shape[0]):
        # Iterate over each channel
        for channel_idx in range(input_tensor.shape[1]):
            # Extract the single channel and convert it to a (H, W) image
            channel_image = input_tensor[batch_idx, channel_idx].cpu().detach().numpy()

            # Normalize the image if necessary (between 0 and 1)
            channel_image = np.clip(channel_image, 0, 1)
            filename = os.path.join(save_dir, f"{name}_batch{batch_idx}_channel{channel_idx}.png")

            plt.imshow(channel_image)
            # plt.imshow(channel_image, cmap='gray')  # 强制灰度色图

            plt.axis("off")
            plt.savefig(filename, bbox_inches="tight", pad_inches=0)
            plt.clf()


# def save_channel_images(input_tensor, name, save_dir="./samples/"):
#     """
#     Save each image in the input tensor as a separate image without applying any normalization.
#
#     Parameters:
#     - input_tensor: A tensor with shape (B, C, H, W) where B is batch size, C is number of channels,
#                      H and W are height and width of the image.
#     - name: The prefix for the saved image file names.
#     - save_dir: Directory where the images will be saved.
#     """
#     # Ensure the save directory exists
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Iterate over each image in the batch
#     for batch_idx in range(input_tensor.shape[0]):
#         # Iterate over each channel
#         for channel_idx in range(input_tensor.shape[1]):
#             # Extract the single channel and convert it to a (H, W) image
#             channel_image = input_tensor[batch_idx, channel_idx].cpu().detach().numpy()
#
#             # If the image is multi-channel (e.g., 3-channel RGB or multi-spectral),
#             # ensure values are properly scaled or adjust accordingly.
#             # For the time being, assuming that the image's pixel values are within the correct range
#             # If the values are not in the range [0, 255], you may want to scale or adjust them.
#
#             # No normalization, just ensure values are in the valid range for visualization
#             channel_image = np.clip(channel_image, 0, 255)  # clip to [0, 255] if necessary
#
#             # Create a filename for each channel
#             filename = os.path.join(save_dir, f"{name}_batch{batch_idx}_channel{channel_idx}.png")
#
#             # Plot the image without applying any colormap (cmap=None)
#             plt.imshow(channel_image.astype(np.uint8))  # Ensure values are integers
#             plt.axis('off')  # Remove axis for better visualization
#             plt.savefig(filename, bbox_inches='tight', pad_inches=0)
#
#             # Clear the figure to avoid memory leaks
#             plt.clf()


def save_tensor_as_npy(tensor, name, save_dir="./samples/"):
    """
    Save tensor as a .npy file.

    Parameters:
    - tensor: The tensor to save.
    - name: The prefix for the saved file name.
    - save_dir: Directory where the .npy file will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Convert tensor to numpy array
    np_array = tensor.cpu().detach().numpy()

    # Save the numpy array to a .npy file
    filename = os.path.join(save_dir, f"{name}.npy")
    np.save(filename, np_array)
    print(f"Saved tensor as {filename}")
