import os
import random

import numpy as np
import torch
from PIL import Image
from skimage import transform as sk_transform
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.utils import _DiffbandsDataLoader


class Mask(object):
    def __init__(self, w=64, h=64, resize=64, sub_w_num=8, sub_h_num=8, dense_rate=0):
        self.w = w
        self.h = h
        self.sub_w_num = sub_w_num
        self.sub_h_num = sub_h_num
        self.target_num_max = sub_w_num * sub_h_num
        self.target_num_range = [1, 32]
        self.dense_rate = dense_rate
        self.resize = resize

    def single_square_shape(self, diameter):
        point_list = []
        move = diameter // 2
        for x in range(0, diameter):
            for y in range(0, diameter):
                point_list = point_list + [(x - move, y - move)]
        return point_list

    def judge_adjacent(self, img):
        adj_point_list = []
        move_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        m, n = img.shape
        for i in range(m):
            for j in range(n):
                if img[i, j] > 0:
                    for move in move_list:
                        if 0 <= i + move[0] <= m - 1 and 0 <= j + move[1] <= n - 1:
                            if img[i + move[0], j + move[1]] == 0:
                                if (i + move[0], j + move[1]) not in adj_point_list:
                                    adj_point_list = adj_point_list + [
                                        (i + move[0], j + move[1])
                                    ]
        return adj_point_list

    def single_random_shape(self, area):
        img = np.zeros([19, 19])
        point_num = 1
        img[9, 9] = 1
        point_list = [(0, 0)]
        while point_num < area:
            adj_point_list = self.judge_adjacent(img)
            if len(adj_point_list) > 0:
                for point in adj_point_list:
                    if random.random() < 0.5 and point_num < area:
                        img[point] = 1
                        point_list = point_list + [(point[0] - 9, point[1] - 9)]
                        point_num += 1
        return point_list

    def single_mask(self, target_num=None):
        if target_num is None:
            # self.target_num = random.randint(1, self.target_num_max)
            self.target_num = random.randint(
                self.target_num_range[0], self.target_num_range[1]
            )
        else:
            self.target_num = target_num
        self.dense = True if random.random() < self.dense_rate else False

        pos_list = []

        pos_id_list = list(range(self.sub_w_num * self.sub_h_num))
        random.shuffle(pos_id_list)
        for i in range(self.target_num):
            w_id, h_id = divmod(pos_id_list[i], self.sub_w_num)
            x_pos = (
                random.randint(0, int(self.w / self.sub_w_num - 1))
                + w_id * self.w / self.sub_w_num
            )
            y_pos = (
                random.randint(0, int(self.h / self.sub_h_num - 1))
                + h_id * self.h / self.sub_h_num
            )
            pos_list = pos_list + [(x_pos, y_pos)]
        self.pos_list = np.array(pos_list)

        mask_image = np.ones([self.w, self.h])
        max_area = 20
        min_area = 3
        for i in range(self.target_num):
            area = random.randint(min_area, max_area)
            single_target_shape = self.single_random_shape(area)
            single_target_shape = np.array(single_target_shape)
            single_target_shape[:, 0] = single_target_shape[:, 0] + self.pos_list[i, 0]
            single_target_shape[:, 1] = single_target_shape[:, 1] + self.pos_list[i, 1]
            for j in range(single_target_shape.shape[0]):
                if (
                    -1 < single_target_shape[j, 0] < self.w
                    and -1 < single_target_shape[j, 1] < self.h
                ):
                    mask_image[single_target_shape[j, 0], single_target_shape[j, 1]] = 0
        mask_image = sk_transform.resize(
            mask_image, (self.resize, self.resize), order=0
        )
        return mask_image

    def __call__(self, n, target_num=None, dense=None):
        Ms = []
        for i in range(n):
            mask = self.single_mask(target_num=target_num)
            mask = mask.astype("int64")
            Ms.append(mask)
        return Ms


class HADDataset(Dataset):
    def __init__(
        self,
        dataset_path="data/Downstreams/HAD/HAD100",
        sensor="aviris_ng",
        mask_class="zero",
        resize=64,
        start_channel=0,
        channel=50,
        train_ratio=1,
        norm_type: str = "img",
        scale: float = 1.0,  # 0.1,
        to_neg_1_1=False,
        ret_dict=False,
    ):
        self.dataset_path = dataset_path
        self.mask_class = mask_class
        self.resize = resize
        self.start_channel = start_channel
        self.channel = channel
        self.sensor = sensor
        self.train_ratio = train_ratio
        self.mask_generator = Mask(
            resize=self.resize,
        )
        # load dataset
        self.train_img, self.paste_img = self.load_dataset_folder()
        # set transforms
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.norm_type = norm_type
        self.scale = scale
        self.to_neg_1_1 = to_neg_1_1
        self.ret_dict = ret_dict

    def normalize(self, x: np.ndarray):
        if self.norm_type == "img":
            x = (x - x.min()) / (x.max() - x.min())
        else:  # per-channel
            x = (x - x.min(axis=(0, 1), keepdims=True)) / (
                x.max(axis=(0, 1), keepdims=True) - x.min(axis=(0, 1), keepdims=True)
            )

        if self.to_neg_1_1:
            x = x * 2 - 1
        x = x * self.scale

        return x

    def __getitem__(self, idx):
        # load image
        img_path = self.train_img[idx]
        x = np.load(img_path)
        x = x[:, :, self.start_channel : (self.channel + self.start_channel)]
        # x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1  # normalize to [-1, 1]
        # x = x * 0.1  # scale to [-0.1, 0.1]
        x = self.normalize(x)

        x = sk_transform.rotate(x, random.choice([0, 90, 180, 270]))
        if random.random() > 0.5:
            x = x[:, ::-1, ...].copy()
        if random.random() > 0.5:
            x = x[::-1, ...].copy()
        x = self.transform(x)
        # x = x.type(torch.FloatTensor)

        # flip
        # add mask
        mask = self.mask_generator(1)[0]
        mask = np.expand_dims(mask, axis=2)
        # breakpoint()
        # x = self.transform(x)
        x = x.type(torch.FloatTensor)
        mask = self.transform(mask)
        mask = mask.type(torch.FloatTensor)
        # mask_out = (0.2 * np.random.rand(x.shape[0], x.shape[1], x.shape[2]) - 0.5)
        # mask_out = 0*np.ones([x.shape[0], x.shape[1], x.shape[2]])
        if self.mask_class == "no":
            x_m = x
        elif self.mask_class == "zero":
            x_m = mask * x
        elif self.mask_class == "other_sensor":
            paste = np.load(random.choice(self.paste_img))
            paste = paste[:, :, : self.channel]
            # paste = (paste - np.min(paste)) / (np.max(paste) - np.min(paste)) * 2 - 1
            # paste = paste * 0.1
            paste = self.normalize(paste)
            paste = self.transform(paste)
            x_m = mask * x + (1 - mask) * paste
        elif self.mask_class == "image":
            HSI_name = img_path.split("/")[-1].split("_")[0]
            while 1:
                paste_path = random.choice(self.train_img)
                paste_HSI_name = paste_path.split("/")[-1].split("_")[0]
                if paste_HSI_name != HSI_name:
                    break
            paste = np.load(paste_path)
            paste = paste[
                :, :, self.start_channel : (self.channel + self.start_channel)
            ]
            # paste = (paste - np.min(paste)) / (np.max(paste) - np.min(paste)) * 2 - 1
            # paste = paste * 0.1
            paste = self.normalize(paste)
            paste = self.transform(paste)
            x_m = mask * x + (1 - mask) * paste
        elif self.mask_class == "random":
            paste = np.random.rand(x.shape[0], x.shape[1], x.shape[2])
            paste = (paste * 2 - 1) * self.scale
            paste = self.transform(paste)
            x_m = mask * x + (1 - mask) * paste
        elif self.mask_class == "sin":
            sin_head = np.pi * np.random.rand(1)
            sin = np.sin(np.linspace(sin_head, sin_head + 2 * np.pi, self.channel))
            paste = (
                sin.reshape(self.channel, 1, 1)
                .repeat(self.resize, axis=1)
                .repeat(self.resize, axis=2)
            )
            paste = paste * self.scale
            x_m = mask * x + (1 - mask) * paste
        elif self.mask_class == "invimage":
            HSI_name = img_path.split("/")[-1].split("_")[0]
            _n = 0
            while 1:
                paste_path = random.choice(self.train_img)
                paste_HSI_name = paste_path.split("/")[-1].split("_")[0]
                if paste_HSI_name != HSI_name or _n > 200:
                    if _n > 200:
                        raise RuntimeError("no other image")
                    break
                _n += 1
            paste = np.load(paste_path)
            paste = paste[
                :, :, self.start_channel : (self.channel + self.start_channel)
            ]
            paste = paste[:, :, ::-1]
            # paste = (paste - np.min(paste)) / (np.max(paste) - np.min(paste)) * 2 - 1
            # paste = paste * 0.1
            paste = self.normalize(paste)
            paste = self.transform(paste)
            x_m = mask * x + (1 - mask) * paste
        else:
            raise Exception("this mode is not defined")
        # x_m_o = mask * x + (1 - mask) * mask_out
        x_m = x_m.type(torch.FloatTensor)
        # x_m_o = x_m_o.type(torch.FloatTensor)

        # print(f"Loaded image shape: {x.shape}, masked image shape: {x_m.shape}")

        if self.ret_dict:
            return {"img": x, "masked_img": x_m}
        else:
            return x, x_m

    def __len__(self):
        return len(self.train_img)

    def load_dataset_folder(self):
        print(f"Loading dataset from: {self.dataset_path}")
        print(f"Sensor: {self.sensor}")

        if self.sensor == "aviris_ng" or self.sensor == "all":
            train_img_dir = os.path.join(self.dataset_path, "train", "aviris_ng")
            paste_img_dir = os.path.join(self.dataset_path, "train", "aviris")
        elif self.sensor == "aviris":
            train_img_dir = os.path.join(self.dataset_path, "train", "aviris")
            paste_img_dir = os.path.join(self.dataset_path, "train", "aviris_ng")
        else:
            train_img_dir = os.path.join(self.dataset_path, "test", "aviris_ng")
            paste_img_dir = os.path.join(self.dataset_path, "train", "aviris")

        print(f"Train image directory: {train_img_dir}")
        print(f"Paste image directory: {paste_img_dir}")

        # Check if directories exist
        if not os.path.exists(train_img_dir):
            print(f"Warning: Train image directory does not exist: {train_img_dir}")
            return [], []

        if not os.path.exists(paste_img_dir):
            print(f"Warning: Paste image directory does not exist: {paste_img_dir}")
            return [], []

        train_list = sorted(
            [
                os.path.join(train_img_dir, f)
                for f in os.listdir(train_img_dir)
                if f.endswith(".npy")
            ]
        )

        print(f"Found {len(train_list)} training files")
        # train_list = sorted(
        #     [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.npy') and
        #                                       'ang20170821t183707' in f])
        train_list = train_list[: int(len(train_list) * self.train_ratio)]
        paste_list = sorted(
            [
                os.path.join(paste_img_dir, f)
                for f in os.listdir(paste_img_dir)
                if f.endswith(".npy")
            ]
        )
        if self.sensor == "all":
            train_list = train_list + paste_list

        return train_list, paste_list

    @classmethod
    def create_dataloader(
        cls,
        dataset_path="data/Downstreams/HAD/HAD100",
        sensor="aviris_ng",
        mask_class="zero",
        resize=64,
        start_channel=0,
        channel=50,
        train_ratio=1,
        norm_type: str = "img",
        scale: float = 1.0,  # 0.1,
        to_neg_1_1=False,
        ret_dict=False,
        **loader_kwargs,
    ):
        """Create a dataloader for loading data.
        Two modes:
            1. HAD mode, loading orignal hyperspectral and masked images.
            2. Pure image mode, loading only images.
        """
        # Set default DataLoader parameters if not provided
        default_loader_kwargs = {
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
        }

        # Update with user-provided kwargs
        default_loader_kwargs.update(loader_kwargs)

        # Create dataset instance with provided parameters
        dataset = cls(
            dataset_path=dataset_path,
            sensor=sensor,
            mask_class=mask_class,
            resize=resize,
            start_channel=start_channel,
            channel=channel,
            train_ratio=train_ratio,
            norm_type=norm_type,
            scale=scale,
            to_neg_1_1=to_neg_1_1,
            ret_dict=ret_dict,
        )

        # Create dataloader
        ensure_same_bands = loader_kwargs.pop("ensure_same_bands", False)
        if not ensure_same_bands:
            dataloader = torch.utils.data.DataLoader(dataset, **default_loader_kwargs)
        else:
            dataloader = _DiffbandsDataLoader(dataset, **default_loader_kwargs)

        return dataset, dataloader


class HADTestDataset(Dataset):
    def __init__(self, dataset_path="./", resize=64, start_channel=0, channel=100):
        self.dataset_path = dataset_path
        self.resize = resize
        self.start_channel = start_channel
        self.channel = channel
        self.mask_generator = Mask(resize=self.resize)
        self.sensor = "aviris_ng"

        # load dataset
        self.test_img, self.gt_img = self.load_dataset_folder()

        # set transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        x, gt = self.test_img[idx], self.gt_img[idx]
        # load test image
        x = np.load(x)
        x = x[:, :, self.start_channel : (self.channel + self.start_channel)]
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
        x = x * 0.1
        x = self.transform(x)
        x = x.type(torch.FloatTensor)

        # load gt
        gt = Image.open(gt)
        gt = np.array(gt)
        gt = gt[:, :, 1]
        gt = Image.fromarray(gt)
        gt = self.transform(gt)
        return x, gt

    def __len__(self):
        return len(self.test_img)

    def load_dataset_folder(self):
        test_img_dir = os.path.join(self.dataset_path, "test", self.sensor)
        gt_dir = os.path.join(self.dataset_path, "ground_truth", self.sensor)
        test_img = sorted(
            [
                os.path.join(test_img_dir, f)
                for f in os.listdir(test_img_dir)
                if f.endswith(".npy")
            ]
        )
        img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in test_img]
        gt_img = [os.path.join(gt_dir, img_name + ".png") for img_name in img_name_list]
        assert len(test_img) == len(gt_img), "number of test img and gt should be same"
        return test_img, gt_img

    @classmethod
    def create_loader(
        cls, dataset_path="./", resize=64, start_channel=0, channel=100, **loader_kwargs
    ):
        """
        Create a complete dataset and dataloader for HAD100 anomaly detection testing.

        Parameters
        ----------
        dataset_path : str
            Path to the HAD100 dataset
        resize : int
            Resize images to this size
        start_channel : int
            Starting channel index
        channel : int
            Number of spectral channels to use
        **loader_kwargs
            Additional arguments for DataLoader (batch_size, shuffle, num_workers, etc.)

        Returns
        -------
        dataset : HADTestDataset
            The created dataset instance
        dataloader : torch.utils.data.DataLoader
            The created dataloader instance
        """
        # Set default DataLoader parameters if not provided (based on train.py defaults)
        default_loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
        }

        # Update with user-provided kwargs
        default_loader_kwargs.update(loader_kwargs)

        # Create dataset instance with provided parameters
        dataset = cls(
            dataset_path=dataset_path,
            resize=resize,
            start_channel=start_channel,
            channel=channel,
        )

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset, **default_loader_kwargs)

        return dataset, dataloader


def test_loader():
    """Test function to visualize HAD100 dataset mask effects"""
    import os

    import matplotlib.pyplot as plt

    # Create output directory
    output_dir = "tmp/had100_visualization"
    os.makedirs(output_dir, exist_ok=True)

    # Test different mask types
    mask_types = ["no", "zero", "other_sensor", "image", "random", "sin", "invimage"]

    # Initialize dataset with different mask types
    print("Testing HAD100 dataset visualization...")

    for mask_type in mask_types:
        print(f"Testing mask type: {mask_type}")

        # Create dataset with current mask type
        dataset = HADDataset(
            dataset_path="data/Downstreams/HAD/HAD100/HAD100Dataset",
            sensor="aviris_ng",
            mask_class=mask_type,
            resize=64,
            start_channel=0,
            channel=50,
            train_ratio=0.1,  # Use smaller subset for testing
        )

        # Get a sample
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            original, masked = dataset[0]
            print(f"Original tensor shape: {original.shape}")
            print(f"Masked tensor shape: {masked.shape}")

            # Convert tensors to numpy
            original_np = original.numpy()
            masked_np = masked.numpy()
            print(f"Original numpy shape: {original_np.shape}")
            print(f"Masked numpy shape: {masked_np.shape}")

            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f"HAD100 Dataset - Mask Type: {mask_type}", fontsize=16)

            # Show original image (first 3 channels as RGB)
            if original_np.shape[0] >= 3:
                rgb_original = np.transpose(original_np[:3], (1, 2, 0))
                rgb_original = (rgb_original - rgb_original.min()) / (
                    rgb_original.max() - rgb_original.min()
                )
                axes[0, 0].imshow(rgb_original)
                axes[0, 0].set_title("Original (RGB)")
            else:
                axes[0, 0].imshow(original_np[0], cmap="viridis")
                axes[0, 0].set_title("Original (Channel 1)")

            # Show masked image (first 3 channels as RGB)
            if masked_np.shape[0] >= 3:
                rgb_masked = np.transpose(masked_np[:3], (1, 2, 0))
                rgb_masked = (rgb_masked - rgb_masked.min()) / (
                    rgb_masked.max() - rgb_masked.min()
                )
                axes[0, 1].imshow(rgb_masked)
                axes[0, 1].set_title("Masked (RGB)")
            else:
                axes[0, 1].imshow(masked_np[0], cmap="viridis")
                axes[0, 1].set_title("Masked (Channel 1)")

            # Show difference
            diff = np.abs(original_np - masked_np)
            axes[0, 2].imshow(diff[0], cmap="hot")
            axes[0, 2].set_title("Difference (Channel 1)")

            # Show mask generator example
            mask_gen = Mask(resize=64)
            mask_example = mask_gen(1)[0]
            axes[0, 3].imshow(mask_example, cmap="gray")
            axes[0, 3].set_title("Mask Pattern")

            # Show spectral profiles
            print(f"Original data shape: {original_np.shape}")
            print(f"Masked data shape: {masked_np.shape}")

            # Use center coordinates dynamically
            h, w = original_np.shape[1], original_np.shape[2]
            center_h, center_w = h // 2, w // 2
            print(f"Image dimensions: {h}x{w}, center: ({center_h}, {center_w})")

            center_pixel = original_np[:, center_h, center_w]
            center_pixel_masked = masked_np[:, center_h, center_w]
            axes[1, 0].plot(center_pixel, "b-", label="Original")
            axes[1, 0].plot(center_pixel_masked, "r-", label="Masked")
            axes[1, 0].set_title("Spectral Profile (Center)")
            axes[1, 0].legend()
            axes[1, 0].set_xlabel("Channel")
            axes[1, 0].set_ylabel("Value")

            # Show histogram
            axes[1, 1].hist(original_np.flatten(), bins=50, alpha=0.5, label="Original")
            axes[1, 1].hist(masked_np.flatten(), bins=50, alpha=0.5, label="Masked")
            axes[1, 1].set_title("Value Distribution")
            axes[1, 1].legend()

            # Show multiple mask examples
            for i in range(2):  # 只显示2个例子以避免越界
                mask_example = mask_gen(1)[0]
                axes[1, 2 + i].imshow(mask_example, cmap="gray")
                axes[1, 2 + i].set_title(f"Mask Example {i + 1}")

            # Remove axis ticks for better visualization
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()

            # Save the plot
            output_path = os.path.join(
                output_dir, f"had100_{mask_type}_visualization.png"
            )
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Saved visualization to: {output_path}")

    # Create mask generator demonstration
    print("Creating mask generator demonstration...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Mask Generator Examples", fontsize=16)

    mask_gen = Mask(resize=64, dense_rate=0.3)

    # Show different mask configurations
    for i in range(8):
        row = i // 4
        col = i % 4

        # Generate mask with different target numbers
        if i < 4:
            mask = mask_gen(1, target_num=i + 1)[0]
            axes[row, col].imshow(mask, cmap="gray")
            axes[row, col].set_title(f"{i + 1} Target(s)")
        else:
            # Show dense masks
            mask = mask_gen(1)[0]
            axes[row, col].imshow(mask, cmap="gray")
            axes[row, col].set_title(f"Dense Mask {i - 3}")

        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    plt.tight_layout()
    mask_demo_path = os.path.join(output_dir, "mask_generator_demo.png")
    plt.savefig(mask_demo_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved mask generator demo to: {mask_demo_path}")

    # Create summary statistics
    print("Creating summary statistics...")
    with open(os.path.join(output_dir, "test_summary.txt"), "w") as f:
        f.write("HAD100 Dataset Test Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tested mask types: {', '.join(mask_types)}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Image size: 64x64\n")
        f.write(f"Channels: 50\n")
        f.write(f"Sensor: aviris_ng\n\n")

        for mask_type in mask_types:
            f.write(f"Mask type: {mask_type}\n")
            f.write(f"  - Visualization: had100_{mask_type}_visualization.png\n")
        f.write(f"\nMask generator demo: mask_generator_demo.png\n")

    print(f"Test summary saved to: {os.path.join(output_dir, 'test_summary.txt')}")
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Test completed successfully!")


def __test_loading_images():
    # Create dataset with current mask type
    dataset = HADDataset(
        dataset_path="data/Downstreams/HAD/HAD100/HAD100Dataset",
        sensor="all",
        mask_class="no",
        resize=64,
        start_channel=0,
        channel=276,
        train_ratio=1,  # Use smaller subset for testing
        to_neg_1_1=True,
        ret_dict=True,
    )
    print(f"Dataset length: {len(dataset)}")

    # img, img_m = dataset[0]
    # print(f"Image shape: {img.shape}, min: {img.min()}, max: {img.max()}")
    # print(f"Mask shape: {img_m.shape}")

    dl = _DiffbandsDataLoader(dataset, batch_size=16, num_workers=1)
    for sample in dl:
        print(sample["img"].shape)


if __name__ == "__main__":
    # test_loader()
    __test_loading_images()
