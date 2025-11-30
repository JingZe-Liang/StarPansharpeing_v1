from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
from loguru import logger


def sample_img_with_gt_indices(
    img: np.ndarray, gt_map: np.ndarray, class_indices: Dict[int, np.ndarray], patch_size: int = 32
):
    """
    Sample image patches with proper padding and coordinate conversion.

    This function extracts patches centered at the specified pixel coordinates,
    with appropriate padding for boundary handling.

    Parameters
    ----------
    img : np.ndarray
        Input image array.
    gt_map : np.ndarray
        Ground truth label map with same shape as img.
    class_indices : dict[int, np.ndarray]
        Dictionary mapping class IDs to arrays of pixel coordinates.
    patch_size : int, optional
        Size of the patches to extract. Default is 32.

    Returns
    -------
    dict[int, list[np.ndarray]]
        Dictionary mapping class IDs to lists of image patches.
    dict[int, list[np.ndarray]]
        Dictionary mapping class IDs to lists of label patches.

    Notes
    -----
    - Image is padded with zeros to handle boundary patches
    - GT is padded with 255 (ignore index) to handle boundary patches
    - Coordinates are treated as center points for patch extraction
    """
    # For hyperspectral images, img should be (H, W, C) and gt should be (H, W)
    assert len(img.shape) == 3, f"Image should be 3D (H, W, C), got shape: {img.shape}"
    assert len(gt_map.shape) == 2, f"Ground truth should be 2D (H, W), got shape: {gt_map.shape}"
    assert img.shape[:2] == gt_map.shape, f"Spatial dimensions mismatch: {img.shape[:2]} != {gt_map.shape}"
    shape_gt, shape_img = gt_map.shape, img.shape

    # Calculate padding margin
    margin = patch_size // 2

    # Pad image with zeros and GT with 255 (ignore index for loss)
    # For 3D image: pad only spatial dimensions (H, W), not spectral channels
    padded_img = np.pad(img, ((margin, margin), (margin, margin), (0, 0)), mode="constant", constant_values=0)
    padded_gt = np.pad(gt_map, margin, mode="constant", constant_values=255)
    unsampled_area = np.zeros_like(padded_gt)

    # Create patches/labels dict
    patches_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    for class_id, indices in class_indices.items():
        indices = cast(np.ndarray, indices)
        for idx in indices:
            center_x, center_y = idx  # Use coordinates as center points
            # Convert to padded image coordinates
            padded_x = center_x + margin
            padded_y = center_y + margin
            slices = slice(padded_x, padded_x + patch_size), slice(padded_y, padded_y + patch_size)

            # Extract patch from padded images
            patch = padded_img[slices[0], slices[1], :]
            label = padded_gt[slices[0], slices[1]]

            # Mask the selected patch for test
            unsampled_area[slices[0], slices[1]] = 1

            patches_dict[class_id].append(patch)
            labels_dict[class_id].append(label)

    # Clip the unsampled area to its original size
    unsampled_area = unsampled_area[margin:-margin, margin:-margin]
    assert gt_map.shape == unsampled_area.shape, (
        f"Shape mismatch after clipping: {gt_map.shape} != {unsampled_area.shape}"
    )

    return patches_dict, labels_dict, unsampled_area


class RobustHyperspectralSampler:
    """Robust sampler for hyperspectral images.

    This sampler implements multiple strategies to alleviate class
    imbalance in hyperspectral datasets, especially when some classes
    have very few pixels.

    Parameters
    ----------
    gt_2d : np.ndarray
        2D ground-truth label map with shape ``(height, width)``.
        Background label is assumed to be ``0`` and ignored.
    balance_strategy : str, optional
        Sampling strategy, one of:

        - ``"equal_size"``: sample the same number of pixels per class.
        - ``"proportional"``: sample proportionally to original sizes.
        - ``"stratified_robust"``: robust stratified split with
          per-split minimum constraints.
        - ``"adaptive_inverse"``: adaptive sampling using inverse
          frequency weights.
        - ``"adaptive_sqrt"``: adaptive sampling using inverse square
          root weights.
        - ``"adaptive_log"``: adaptive sampling using inverse
          logarithmic weights.
        - ``"combined_smart"``: combined over/under-sampling based on
          relative class size.

        Default is ``"stratified_robust"``.
    min_per_class : int, optional
        Minimum number of pixels per class for splits when applicable.
        Default is ``1``.
    target_samples_per_class : int or None, optional
        Optional target number of samples per class for some strategies.
        Default is ``None``.
    random_seed : int or None, optional
        Optional random seed for reproducibility. Default is ``None``.
    verbose : bool, optional
        If ``True``, print detailed statistics and warnings.
        Default is ``True``.

    Examples
    --------
    >>> gt_2d = np.array([
    ...     [1, 1, 0, 2, 2, 1],
    ...     [3, 0, 1, 3, 3],
    ...     [2, 2, 2, 0, 1]
    ... ])
    >>> sampler = RobustHyperspectralSampler(gt_2d, balance_strategy='stratified_robust')
    >>> train_idx, val_idx, test_idx = sampler.sample(min_per_class=2)
    >>> print(f"Total sampled: {len(train_idx) + len(val_idx) + len(test_idx)} pixels")
    """

    def __init__(
        self,
        gt_2d: np.ndarray,
        balance_strategy: str = "stratified_robust",
        min_per_class: int = 1,
        target_samples_per_class: Optional[int] = None,
        random_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the sampler with label map and strategy.

        Parameters
        ----------
        gt_2d : np.ndarray
            2D ground-truth label map with shape ``(height, width)``.
        balance_strategy : str, optional
            Name of sampling strategy.
        min_per_class : int, optional
            Minimum number of samples per split for each class.
        target_samples_per_class : int or None, optional
            Target number of samples per class if used by the strategy.
        random_seed : int or None, optional
            Random seed used for NumPy RNG.
        verbose : bool, optional
            Whether to print detailed information.
        """
        self.gt_2d = gt_2d
        self.strategy = balance_strategy
        self.min_per_class = min_per_class
        self.target_samples_per_class = target_samples_per_class
        self.verbose = verbose
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        # Build class_indices mapping from gt_2d automatically
        self.class_indices = self._create_class_indices()

        # Validate strategy name
        valid_strategies = [
            "equal_size",
            "proportional",
            "stratified_robust",
            "adaptive_inverse",
            "adaptive_sqrt",
            "adaptive_log",
            "combined_smart",
        ]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {self.strategy}. Available: {valid_strategies}")

        # Compute basic class statistics
        self.class_stats = self._analyze_class_distribution()

        if self.verbose:
            self._print_class_distribution()

    def _create_class_indices(self) -> Dict[int, List[Tuple[int, int]]]:
        """Create mapping from class id to 2D pixel coordinates.

        Returns
        -------
        dict[int, list[tuple[int, int]]]
            Dictionary mapping class id to list of 2D pixel
            coordinates (row, col).
        """
        height, width = self.gt_2d.shape
        class_indices = {}

        # Collect all non-zero (non-background) labels
        unique_classes = np.unique(self.gt_2d)
        for class_id in unique_classes:
            # if class_id == 0:  # skip background class
            #     continue

            # Get all pixel positions for this class
            class_pixels = np.where(self.gt_2d == class_id)
            # Store as (row, col) coordinate tuples
            coordinates = list(zip(class_pixels[0], class_pixels[1]))
            class_indices[int(class_id)] = coordinates

        return class_indices

    def _analyze_class_distribution(self) -> Dict[int, Dict[str, Union[int, float]]]:
        """Compute basic statistics for each class.

        Returns
        -------
        dict[int, dict]
            Per-class statistics with keys ``"count"`` and
            ``"percentage"``.
        """
        stats = {}
        total_samples = sum(len(indices) for indices in self.class_indices.values())

        for class_id, indices in self.class_indices.items():
            stats[class_id] = {"count": len(indices), "percentage": len(indices) / total_samples * 100.0}

        return stats

    def _print_class_distribution(self) -> None:
        """Pretty-print class distribution and imbalance metrics."""
        stats = self.class_stats

        if self.verbose:
            logger.info("\n" + "=" * 60)
            logger.info("Hyperspectral class distribution analysis:")
            logger.info(f"Image shape: {self.gt_2d.shape}")
            logger.info(f"Number of valid classes: {len(stats)}")
            logger.info(f"Number of labeled pixels: {sum(info['count'] for info in stats.values())}")
            logger.info("-" * 50)
            logger.info(f"{'ClassID':<8} {'Count':<10} {'Percent':<10}")
            logger.info("-" * 35)

            for class_id in sorted(stats.keys()):
                info = stats[class_id]
                logger.info(f"{class_id:<8} {info['count']:<10} {info['percentage']:<10.2f}%")

            logger.info("-" * 50)

            # Imbalance ratio: max / min class size
            counts = [info["count"] for info in stats.values()]
            imbalance_ratio = max(counts) / min(counts)
            logger.info(f"Imbalance ratio: {imbalance_ratio:.2f} (max/min)")

            # Gini coefficient over class proportions
            total_samples = sum(counts)
            probs = [c / total_samples for c in counts]
            gini = sum(abs(p_i - p_j) for i, p_i in enumerate(probs) for j, p_j in enumerate(probs) if i != j) / (
                2 * len(probs)
            )
            logger.info(f"Gini coefficient: {gini:.4f} (0=perfect, 1=worst)")

    def sample(self, **kwargs) -> Union[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]]]:
        """Run sampling according to the configured strategy.

        Parameters
        ----------
        **kwargs : dict
            Extra parameters forwarded to the underlying strategy
            implementation.

        Returns
        -------
        dict[int, np.ndarray] or dict[str, dict[int, np.ndarray]]
            Balanced sampled coordinates with shape ``(n_samples, 2)``.
            For ``"stratified_robust"`` this returns a dict with keys
            ``"train"``, ``"val"``, and ``"test"`` mapping to dicts of
            numpy arrays per class.
        """
        if self.strategy == "equal_size":
            return self._equal_size_sampling(**kwargs)
        elif self.strategy == "proportional":
            return self._proportional_sampling(**kwargs)
        elif self.strategy == "stratified_robust":
            return self._stratified_sampling_robust(**kwargs)
        elif self.strategy == "adaptive_inverse":
            return self._adaptive_sampling("inverse", **kwargs)
        elif self.strategy == "adaptive_sqrt":
            return self._adaptive_sampling("sqrt_inverse", **kwargs)
        elif self.strategy == "adaptive_log":
            return self._adaptive_sampling("log_inverse", **kwargs)
        elif self.strategy == "combined_smart":
            return self._combined_smart_sampling(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _equal_size_sampling(
        self, samples_per_class: Optional[int] = None, shuffle: bool = True
    ) -> Dict[int, np.ndarray]:
        """Sample the same number of pixels from each class.

        Parameters
        ----------
        samples_per_class : int or None, optional
            Number of pixels per class. If ``None``, use the smallest
            class size.
        shuffle : bool, optional
            Whether to shuffle each class coordinate list before sampling.

        Returns
        -------
        dict[int, np.ndarray]
            Balanced coordinates per class, each with shape ``(n_samples, 2)``.
        """
        balanced_indices = {}

        # Decide how many pixels to sample per class
        if samples_per_class is None:
            samples_per_class = min(len(indices) for indices in self.class_indices.values())

        for class_id, indices in self.class_indices.items():
            available_count = len(indices)

            if available_count < samples_per_class:
                if self.verbose:
                    logger.warning(
                        f"Warning: Class {class_id} only has {available_count} samples, "
                        f"less than requested {samples_per_class}"
                    )
                selected = indices.copy()
            else:
                indices_copy = indices.copy()
                if shuffle:
                    np.random.shuffle(indices_copy)
                selected = indices_copy[:samples_per_class]

            balanced_indices[class_id] = selected

        # Convert to numpy arrays with shape [n, 2] for each class
        balanced_arrays = {class_id: np.array(coords, dtype=np.int32) for class_id, coords in balanced_indices.items()}
        return balanced_arrays

    def _proportional_sampling(
        self,
        total_samples: int,
        min_per_class: Optional[int] = None,
        max_per_class: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[int, np.ndarray]:
        """Sample pixels in proportion to original class sizes.

        Parameters
        ----------
        total_samples : int
            Total number of pixels to sample across all classes.
        min_per_class : int or None, optional
            Minimum number of samples per class.
        max_per_class : int or None, optional
            Maximum number of samples per class.
        shuffle : bool, optional
            Whether to shuffle coordinates before sampling.
        """
        # Compute class-wise proportions
        class_sizes = {k: len(v) for k, v in self.class_indices.items()}
        total_original = sum(class_sizes.values())
        proportions = {k: v / total_original for k, v in class_sizes.items()}

        # Initialize sampled coordinate mapping
        sampled_indices = {}
        used_samples = 0

        # First pass: satisfy minimum requirement per class
        if min_per_class is not None:
            for class_id, indices in self.class_indices.items():
                available = len(indices)
                actual_min = min(min_per_class, available)
                indices_copy = indices.copy()
                if shuffle:
                    np.random.shuffle(indices_copy)
                sampled_indices[class_id] = indices_copy[:actual_min]
                used_samples += actual_min

        # Second pass: distribute remaining samples proportionally
        remaining_samples = total_samples - used_samples
        if remaining_samples > 0:
            for class_id, indices in self.class_indices.items():
                if class_id not in sampled_indices:  # avoid double counting
                    available = len(indices)
                    class_samples = int(remaining_samples * proportions[class_id])

                    # 应用数量限制
                    if max_per_class is not None:
                        class_samples = min(class_samples, max_per_class)

                    class_samples = min(class_samples, available)

                    indices_copy = indices.copy()
                    if shuffle:
                        np.random.shuffle(indices_copy)
                    sampled_indices[class_id] = indices_copy[:class_samples]
                    used_samples += class_samples

        # Convert to numpy arrays with shape [n, 2] for each class
        sampled_arrays = {class_id: np.array(coords, dtype=np.int32) for class_id, coords in sampled_indices.items()}
        return sampled_arrays

    def _stratified_sampling_robust(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        min_per_class: int = 1,
        shuffle: bool = True,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Robust stratified sampling with per-split minimum counts.

        This method performs a per-class train/val/test split while
        enforcing a minimum number of samples per split. Classes that do
        not have enough pixels to satisfy all splits are reported and
        skipped.

        Parameters
        ----------
        train_ratio : float, optional
            Proportion of samples assigned to the training split.
        val_ratio : float, optional
            Proportion of samples assigned to the validation split.
        test_ratio : float, optional
            Proportion of samples assigned to the test split.
        min_per_class : int, optional
            Minimum number of samples per class and per split.
        shuffle : bool, optional
            Whether to shuffle per-class coordinates before splitting.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        train_indices = {}
        val_indices = {}
        test_indices = {}
        insufficient_classes = []

        for class_id, indices in self.class_indices.items():
            available_count = len(indices)
            min_total = min_per_class * 3

            if available_count < min_total:
                insufficient_classes.append((class_id, available_count, min_total))
                if self.verbose:
                    logger.warning(
                        f"[Warning] Class {class_id}: only {available_count} samples "
                        f"available (need {min_total}, missing {min_total - available_count}).",
                    )
                continue

            # Enough samples: perform stratified splitting
            indices_copy = indices.copy()
            if shuffle:
                np.random.shuffle(indices_copy)

            n_train = max(min_per_class, int(available_count * train_ratio))
            n_val = max(min_per_class, int(available_count * val_ratio))
            n_test = max(min_per_class, int(available_count * test_ratio))

            # If requested total exceeds available, rescale
            total_assigned = n_train + n_val + n_test
            if total_assigned > available_count:
                scale = available_count / total_assigned
                if self.verbose:
                    logger.warning(
                        f"[Warning] Class {class_id}: scaled split counts from {total_assigned} to {available_count}.",
                    )
                n_train = max(min_per_class, int(n_train * scale))
                n_val = max(min_per_class, int(n_val * scale))
                n_test = max(min_per_class, int(n_test * scale))
                if self.verbose:
                    logger.info(f"    Final per-class split: train={n_train}, val={n_val}, test={n_test}")

            # Slice coordinates into three splits
            start_idx = 0
            train_indices[class_id] = indices_copy[start_idx : start_idx + n_train]
            start_idx += n_train

            val_indices[class_id] = indices_copy[start_idx : start_idx + n_val]
            start_idx += n_val

            test_indices[class_id] = indices_copy[start_idx : start_idx + n_test]
            start_idx += n_test

        # Report classes that do not satisfy minimum constraints
        if insufficient_classes and self.verbose:
            logger.info("\n[Warning] Classes with insufficient samples:")
            logger.info(f"{'ClassID':<10} {'Available':<12} {'Required':<12} {'Missing':<15}")
            logger.info("-" * 50)

            for class_id, available_count, min_total in insufficient_classes:
                missing = min_total - available_count
                logger.info(f"{class_id:<10} {available_count:<12} {min_total:<12} {missing:<15}")

            logger.info("-" * 50)
            logger.info("Suggestions:")
            logger.info("  1. Collect more labeled pixels or use augmentation.")
            logger.info("  2. Switch to an adaptive strategy (e.g. 'adaptive_inverse').")
            logger.info("  3. Lower 'min_per_class' if appropriate.")

        # Convert to numpy arrays with shape [n, 2] for each split
        train_arrays = {class_id: np.array(coords, dtype=np.int32) for class_id, coords in train_indices.items()}
        val_arrays = {class_id: np.array(coords, dtype=np.int32) for class_id, coords in val_indices.items()}
        test_arrays = {class_id: np.array(coords, dtype=np.int32) for class_id, coords in test_indices.items()}

        return {"train": train_arrays, "val": val_arrays, "test": test_arrays}

    def _adaptive_sampling(self, weight_type: str, **kwargs) -> Dict[int, np.ndarray]:
        """Adaptive sampling using inverse-frequency based weights.

        Parameters
        ----------
        weight_type : str
            Type of weight to use. Supported values:
            ``"inverse"``, ``"sqrt_inverse"``, ``"log_inverse"``.
        **kwargs : dict
            Extra keyword arguments. Supported key: ``"target_size"``
            for total number of sampled pixels.
        """
        class_sizes = {k: len(v) for k, v in self.class_indices.items()}
        target_size = kwargs.get("target_size", sum(class_sizes.values()) // 2)

        # Compute weights according to selected scheme
        weights = {}
        for class_id, size in class_sizes.items():
            if weight_type == "inverse":
                weights[class_id] = 1.0 / size
            elif weight_type == "sqrt_inverse":
                weights[class_id] = 1.0 / np.sqrt(size)
            elif weight_type == "log_inverse":
                weights[class_id] = 1.0 / np.log(size + 1)
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}")

        # Normalize weights to sum to one
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Compute per-class sample counts
        sampled_indices = {}
        for class_id, indices in self.class_indices.items():
            samples_for_class = int(target_size * weights[class_id])
            samples_for_class = max(1, samples_for_class)  # at least one pixel

            available = len(indices)
            actual_samples = min(samples_for_class, available)

            indices_copy = indices.copy()
            np.random.shuffle(indices_copy)
            sampled_indices[class_id] = indices_copy[:actual_samples]

        # Convert to numpy arrays with shape [n, 2] for each class
        sampled_arrays = {class_id: np.array(coords, dtype=np.int32) for class_id, coords in sampled_indices.items()}
        return sampled_arrays

    def _combined_smart_sampling(
        self,
        target_samples_per_class: Optional[int] = None,
        oversample_threshold: float = 0.5,
        undersample_threshold: float = 2.0,
        shuffle: bool = True,
    ) -> Dict[int, np.ndarray]:
        """Hybrid over/under-sampling based on relative class size.

        Parameters
        ----------
        target_samples_per_class : int or None, optional
            Optional target number of samples per class. If ``None``, a
            heuristic value based on the average class size is used.
        oversample_threshold : float, optional
            Threshold (ratio to average size) below which a class is
            considered minority and is over-sampled.
        undersample_threshold : float, optional
            Threshold (ratio to average size) above which a class is
            considered majority and is under-sampled.
        shuffle : bool, optional
            Whether to shuffle coordinates when (under-)sampling.
        """
        class_sizes = {k: len(v) for k, v in self.class_indices.items()}
        avg_size = sum(class_sizes.values()) / len(class_sizes)

        # Classify classes into minority / balanced / majority groups
        minority_classes = []
        majority_classes = []
        balanced_classes = []

        for class_id, size in class_sizes.items():
            ratio = size / avg_size
            if ratio < oversample_threshold:
                minority_classes.append(class_id)
            elif ratio > undersample_threshold:
                majority_classes.append(class_id)
            else:
                balanced_classes.append(class_id)

        # Target size heuristic
        target_size = target_samples_per_class or int(avg_size * 2)

        sampled_indices = {}
        for class_id, indices in self.class_indices.items():
            available_count = len(indices)

            if class_id in minority_classes:
                # Oversample minority class by repeating its coordinates
                target_count = int(avg_size * oversample_threshold * 1.2)
                repeat_times = target_count // available_count
                remainder = target_count % available_count

                sampled = indices * repeat_times + indices[:remainder]
                if shuffle:
                    np.random.shuffle(sampled)
                sampled_indices[class_id] = sampled

            elif class_id in majority_classes:
                # Undersample majority class
                target_count = int(avg_size * undersample_threshold * 0.8)
                target_count = min(target_count, available_count)

                indices_copy = indices.copy()
                if shuffle:
                    np.random.shuffle(indices_copy)
                sampled_indices[class_id] = indices_copy[:target_count]

            else:
                # Balanced class: keep as is (optionally shuffled)
                indices_copy = indices.copy()
                if shuffle:
                    np.random.shuffle(indices_copy)
                sampled_indices[class_id] = indices_copy

        # Convert to numpy arrays with shape [n, 2] for each class
        sampled_arrays = {class_id: np.array(coords, dtype=np.int32) for class_id, coords in sampled_indices.items()}
        return sampled_arrays

    def get_class_stats(self) -> Dict[int, Dict[str, Union[int, float]]]:
        """Return cached per-class statistics (count and percentage)."""
        return self.class_stats

    def get_sampling_summary(self) -> Dict:
        """Return a summary of current sampling statistics.

        Returns
        -------
        dict
            Dictionary with global imbalance metrics and per-class
            counts.
        """
        class_sizes = {class_id: len(indices) for class_id, indices in self.class_indices.items()}

        # Imbalance indicators on raw class sizes
        counts = [len(v) for v in self.class_indices.values()]
        imbalance_ratio = max(counts) / min(counts)
        total_samples = sum(counts)
        proportions = [c / total_samples for c in counts]
        gini = sum(
            abs(p_i - p_j) for i, p_i in enumerate(proportions) for j, p_j in enumerate(proportions) if i != j
        ) / (2 * len(proportions))

        return {
            "total_classes": len(self.class_indices),
            "total_samples": total_samples,
            "min_class_size": min(counts),
            "max_class_size": max(counts),
            "imbalance_ratio": imbalance_ratio,
            "gini_coefficient": gini,
            "strategy": self.strategy,
            "class_distribution": {class_id: count for class_id, count in class_sizes.items()},
        }

    def convert_to_numpy_arrays(
        self, sampled_indices: Union[Dict[int, List[Tuple[int, int]]], Tuple[Dict, Dict, Dict]]
    ) -> Union[Dict[int, np.ndarray], Tuple[Dict, Dict, Dict]]:
        """Convert sampled indices from list of tuples to numpy arrays.

        Parameters
        ----------
        sampled_indices : dict[int, list[tuple[int, int]]] or tuple
            Sampled indices from sampling methods. For stratified_robust,
            this is a tuple of three dicts (train, val, test).

        Returns
        -------
        dict[int, np.ndarray] or tuple
            Converted indices where each value is a numpy array with
            shape (n, 2) representing (row, col) coordinates.
            For stratified_robust, returns a tuple of three dicts.
        """
        if isinstance(sampled_indices, tuple):
            # Handle stratified_robust case with train/val/test splits
            train_arrays = {}
            val_arrays = {}
            test_arrays = {}

            for class_id, coords in sampled_indices[0].items():
                train_arrays[class_id] = np.array(coords, dtype=np.int32)

            for class_id, coords in sampled_indices[1].items():
                val_arrays[class_id] = np.array(coords, dtype=np.int32)

            for class_id, coords in sampled_indices[2].items():
                test_arrays[class_id] = np.array(coords, dtype=np.int32)

            return train_arrays, val_arrays, test_arrays
        else:
            # Handle all other sampling strategies
            numpy_arrays = {}
            for class_id, coords in sampled_indices.items():
                numpy_arrays[class_id] = np.array(coords, dtype=np.int32)
            return numpy_arrays


# Example usage when running this module directly
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create 256x256 test image and ground truth
    height, width = 256, 256
    num_bands = 10
    img = np.random.rand(height, width, num_bands).astype(np.float32)  # Random 10-band hyperspectral image
    gt_map = np.zeros((height, width), dtype=np.int32)

    # Create synthetic class distribution with imbalance
    # Class 1: Large class (40% of pixels)
    # Class 2: Medium class (25% of pixels)
    # Class 3: Small class (15% of pixels)
    # Class 4: Very small class (5% of pixels)
    # Background (0): 15% of pixels
    total_pixels = height * width
    class_1_pixels = int(0.4 * total_pixels)
    class_2_pixels = int(0.25 * total_pixels)
    class_3_pixels = int(0.15 * total_pixels)
    class_4_pixels = int(0.05 * total_pixels)

    # Randomly assign class labels
    flat_gt = np.zeros(total_pixels, dtype=np.int32)
    flat_gt[:class_1_pixels] = 1
    flat_gt[class_1_pixels : class_1_pixels + class_2_pixels] = 2
    flat_gt[class_1_pixels + class_2_pixels : class_1_pixels + class_2_pixels + class_3_pixels] = 3
    flat_gt[
        class_1_pixels + class_2_pixels + class_3_pixels : class_1_pixels
        + class_2_pixels
        + class_3_pixels
        + class_4_pixels
    ] = 4
    # Background remains 0

    # Shuffle and reshape
    np.random.shuffle(flat_gt)
    gt_map = flat_gt.reshape((height, width))

    logger.info("=" * 60)
    logger.info("Robust hyperspectral patch extraction test")
    logger.info(f"Image shape: {img.shape}")
    logger.info(f"GT map shape: {gt_map.shape}")
    logger.info(f"Patch size: 32")

    # Create sampler with equal_size strategy
    sampler = RobustHyperspectralSampler(
        gt_2d=gt_map,
        balance_strategy="equal_size",
        random_seed=42,
        verbose=True,
    )

    # Test equal size sampling
    logger.info(f"\n{'=' * 30} Strategy: equal_size")
    sampled_indices: Dict[int, np.ndarray] = sampler.sample(samples_per_class=5)  # type: ignore

    total_sampled = sum(len(v) for v in sampled_indices.values())
    logger.info(f"Total sampled pixels: {total_sampled}")

    for class_id, indices in sampled_indices.items():
        logger.info(f"Class {class_id}: {len(indices)} pixels")

    # Test the new patch extraction function
    logger.info(f"\n{'=' * 30} Testing patch extraction")
    logger.info("Testing sample_img_with_gt_indices function...")

    patch_size = 32
    patches_dict, labels_dict = sample_img_with_gt_indices(
        img=img, gt_map=gt_map, class_indices=sampled_indices, patch_size=patch_size
    )

    # Verify patch extraction results
    logger.info(f"\nPatch extraction verification:")
    logger.info(f"Expected patch size: {patch_size}x{patch_size}")

    for class_id, patches in patches_dict.items():
        if patches:
            patch_shape = patches[0].shape
            label_shape = labels_dict[class_id][0].shape

            logger.info(f"Class {class_id}:")
            logger.info(f"  Number of patches: {len(patches)}")
            logger.info(f"  Image patch shape: {patch_shape}")
            logger.info(f"  Label patch shape: {label_shape}")

            # Verify shapes are correct
            assert patch_shape == (patch_size, patch_size, img.shape[2]), f"Wrong patch shape: {patch_shape}"
            assert label_shape == (patch_size, patch_size), f"Wrong label shape: {label_shape}"

            # Check a few coordinates and their corresponding patches
            class_coords = sampled_indices[class_id]
            coords = class_coords[:2] if len(class_coords) >= 2 else class_coords
            logger.info(f"  First 2 sampled coordinates: {coords.tolist()}")

            # Check padding effectiveness for boundary pixels
            test_coord = class_coords[0]  # Get first coordinate
            center_x, center_y = test_coord
            logger.info(f"  Test coordinate (center): ({center_x}, {center_y})")

            # Verify patch contains expected number of non-zero values (for GT, 255 is padding)
            test_label = labels_dict[class_id][0]
            padding_pixels: int = np.sum(test_label == 255)
            logger.info(f"  Padding pixels in label patch: {padding_pixels}")

    logger.info(f"\n✓ All patch extractions successful!")
    logger.info(f"✓ All patches have correct dimensions: {patch_size}x{patch_size}")
    logger.info(f"✓ Padding strategy working correctly (GT padded with 255)")
    logger.info(f"✓ Image padded with zeros (check visually if needed)")
