"""
Tests for stage implementations in src.stage2.layers.stages module.

This module uses pytest fixtures and parametrization to test different stage
implementations with various configurations.
"""

import pytest
import torch

# Import all stage classes from the module
from src.stage2.layers.stages import (
    MbConvSequentialCond,
    MbConvStagesCond,
    Spatial2DNatStage,
    ResBlockStage,
)


@pytest.fixture
def device() -> torch.device:
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_data(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Provide sample input and condition tensors for testing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing (input_tensor, condition_tensor)
    """
    # Common test parameters
    batch_size = 2
    in_channels = 64
    height, width = 32, 32

    # Test input tensors
    x = torch.randn(batch_size, in_channels, height, width, device=device)
    cond = torch.randn(
        batch_size, 128, height // 2, width // 2, device=device
    )  # Different spatial size

    return x, cond


@pytest.fixture(params=[16, 32, 64])
def spatial_size(request) -> tuple[int, int]:
    """Parametrize different spatial sizes for testing."""
    size = request.param
    return size, size


@pytest.fixture
def minimal_sample_data(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Provide minimal sample data for edge case testing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing (minimal_input_tensor, minimal_condition_tensor)
    """
    minimal_x = torch.randn(1, 16, 8, 8, device=device)
    minimal_cond = torch.randn(1, 64, 16, 16, device=device)
    return minimal_x, minimal_cond


class TestMbConvSequentialCond:
    """Test cases for MbConvSequentialCond stage."""

    @pytest.mark.parametrize(
        "config",
        [
            {
                "in_chans": 64,
                "embed_dim": [64, 128],
                "depths": [2, 2],
                "cond_width": None,
                "out_chans": 64,
                "name": "no_condition",
            },
            {
                "in_chans": 64,
                "embed_dim": [64, 128],
                "depths": [2, 2],
                "cond_width": 128,
                "out_chans": 64,
                "name": "with_condition",
            },
        ],
        ids=lambda cfg: cfg["name"],
    )
    def test_mbconv_sequential_cond_forward(
        self,
        config: dict,
        sample_data: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        """
        Test forward pass of MbConvSequentialCond with different configurations.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the stage
        sample_data : tuple[torch.Tensor, torch.Tensor]
            Sample input and condition tensors
        device : torch.device
            Device to run the test on
        """
        x, cond = sample_data
        stage = MbConvSequentialCond(**{k: v for k, v in config.items() if k != "name"})
        stage = stage.to(device)

        # Test in eval mode
        stage.eval()
        with torch.no_grad():
            if config.get("cond_width") is not None:
                output = stage(x, cond)
            else:
                output = stage(x)

        # Validate output
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == x.shape[0]  # Batch size preserved
        if config["out_chans"]:
            assert output.shape[1] == config["out_chans"]

    def test_mbconv_sequential_cond_gradient_flow(
        self, sample_data: tuple[torch.Tensor, torch.Tensor], device: torch.device
    ) -> None:
        """Test that gradients flow properly through the stage."""
        x, cond = sample_data
        stage = MbConvSequentialCond(
            in_chans=64, embed_dim=[64], depths=[1], cond_width=128, out_chans=64
        )
        stage = stage.to(device)
        stage.train()

        output = stage(x, cond)
        loss = output.mean()
        loss.backward()

        # Check that some parameters have gradients
        param_with_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters()
        )
        assert param_with_grad, "Some parameters should have gradients"

    def test_mbconv_sequential_cond_gradient_checkpointing(
        self, sample_data: tuple[torch.Tensor, torch.Tensor], device: torch.device
    ) -> None:
        """Test gradient checkpointing functionality."""
        x, cond = sample_data
        stage = MbConvSequentialCond(
            in_chans=64, embed_dim=[64], depths=[1], cond_width=128, out_chans=64
        )
        stage = stage.to(device)
        stage.set_grad_checkpointing(True)
        stage.train()

        output = stage(x, cond)
        loss = output.mean()
        loss.backward()

        # Should not raise any exceptions
        assert True


class TestMbConvStagesCond:
    """Test cases for MbConvStagesCond stage."""

    def test_mbconv_stages_cond_forward(
        self, sample_data: tuple[torch.Tensor, torch.Tensor], device: torch.device
    ) -> None:
        """Test forward pass of MbConvStagesCond."""
        x, cond = sample_data
        stage = MbConvStagesCond(
            in_chans=64,
            stem_width=64,
            embed_dim=[64, 128],
            depths=[2, 2],
            cond_width=64,
            stride=1,
        )
        stage = stage.to(device)
        stage.eval()

        with torch.no_grad():
            output = stage(x, cond)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == x.shape[0]  # Batch size preserved


class TestSpatial2DNatStage:
    """Test cases for Spatial2DNatStage."""

    @pytest.mark.parametrize(
        "config",
        [
            {
                "cond_width": None,
                "name": "no_condition",
            },
            {
                "cond_width": 128,
                "name": "with_condition",
            },
        ],
        ids=lambda cfg: cfg["name"],
    )
    def test_spatial2d_nat_stage_forward(
        self,
        config: dict,
        sample_data: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        """
        Test forward pass of Spatial2DNatStage with different configurations.

        Parameters
        ----------
        config : dict
            Configuration for conditional/unconditional mode
        sample_data : tuple[torch.Tensor, torch.Tensor]
            Sample input and condition tensors
        device : torch.device
            Device to run the test on
        """
        x, cond = sample_data
        stage = Spatial2DNatStage(
            in_chans=64,
            embed_dim=[64, 128],
            depths=[1, 1],  # Use fewer layers for faster testing
            out_chans=64,
            k_size=4,
            n_heads=4,
            cond_width=config["cond_width"],
        )
        stage = stage.to(device)
        stage.eval()

        with torch.no_grad():
            if config.get("cond_width") is not None:
                output = stage(x, cond)
            else:
                output = stage(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == x.shape[0]


class TestResBlockStage:
    """Test cases for ResBlockStage."""

    @pytest.mark.parametrize("use_condition", [True, False])
    def test_resblock_stage_forward(
        self,
        use_condition: bool,
        sample_data: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Test forward pass of ResBlockStage with and without conditions."""
        x, cond = sample_data
        stage = ResBlockStage(
            in_chans=64,
            embed_dim=[64, 128],
            depths=[2, 2],
            cond_width=128 if use_condition else None,
            out_chans=64,
        )
        stage = stage.to(device)
        stage.eval()

        with torch.no_grad():
            if use_condition:
                output = stage(x, cond)
            else:
                output = stage(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == x.shape[0]


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_minimal_configuration(
        self,
        minimal_sample_data: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Test stages with minimal configuration."""
        minimal_x, minimal_cond = minimal_sample_data
        stage = MbConvSequentialCond(
            in_chans=16,
            embed_dim=[32],
            depths=[1],
            cond_width=None,
            out_chans=32,
        )
        stage = stage.to(device)
        stage.eval()

        with torch.no_grad():
            output = stage(minimal_x)

        assert output.shape == (1, 32, 8, 8)

    def test_condition_interpolation(self, device: torch.device) -> None:
        """Test condition interpolation with different spatial sizes."""
        stage = MbConvSequentialCond(
            in_chans=16,
            embed_dim=[32],
            depths=[1],
            cond_width=64,
            out_chans=32,
        )
        stage = stage.to(device)

        small_x = torch.randn(1, 16, 16, 16, device=device)
        large_cond = torch.randn(
            1, 64, 32, 32, device=device
        )  # Condition larger than input

        stage.eval()
        with torch.no_grad():
            output = stage(small_x, large_cond)

        assert output.shape == (1, 32, 16, 16)

    def test_different_spatial_sizes(
        self, spatial_size: tuple[int, int], device: torch.device
    ) -> None:
        """Test stages with different input spatial sizes."""
        h, w = spatial_size
        x = torch.randn(1, 32, h, w, device=device)
        stage = MbConvSequentialCond(
            in_chans=32,
            embed_dim=[64],
            depths=[1],
            cond_width=None,
            out_chans=64,
        )
        stage = stage.to(device)
        stage.eval()

        with torch.no_grad():
            output = stage(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 64
        assert output.shape[2] == h
        assert output.shape[3] == w


class TestStageProperties:
    """Test stage properties and utilities."""

    def test_grad_checkpointing_toggle(self, device: torch.device) -> None:
        """Test that gradient checkpointing can be toggled on/off."""
        stage = MbConvSequentialCond(
            in_chans=64,
            embed_dim=[64],
            depths=[1],
            cond_width=None,
            out_chans=64,
        )
        stage = stage.to(device)

        # Initially should be False
        assert stage.grad_checkpointing is False

        # Enable checkpointing
        stage.set_grad_checkpointing(True)
        assert stage.grad_checkpointing is True

        # Disable checkpointing
        stage.set_grad_checkpointing(False)
        assert stage.grad_checkpointing is False

    @pytest.mark.parametrize("stage_class", [MbConvSequentialCond, ResBlockStage])
    def test_stage_num_stages_property(self, stage_class, device: torch.device) -> None:
        """Test that num_stages property is correctly set."""
        embed_dim = [32, 64, 128]  # 3 stages
        stage = stage_class(
            in_chans=16,
            embed_dim=embed_dim,
            depths=[1, 1, 1],
            cond_width=None,
            out_chans=128,
        )
        stage = stage.to(device)

        assert stage.num_stages == len(embed_dim)
