import unittest
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.cosmos.modules.blocks import AdaptiveInputConvLayer, AdaptiveOutputConvLayer
from src.stage1.cosmos.modules.blocks import AdaptiveInputLinearLayer, AdaptiveOutputLinearLayer


class TestAdaptiveConvMix(unittest.TestCase):
    def test_adaptive_input_conv_mix_static_shape(self) -> None:
        layer = AdaptiveInputConvLayer(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=True,
            mode="mix",
        )
        x = torch.randn(2, 5, 16, 16)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 4, 16, 16))

    def test_adaptive_input_conv_mix_dynamic_router_coeff_changes(self) -> None:
        layer = AdaptiveInputConvLayer(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=True,
            mode="mix",
            router_condition="per_channel_mean",
            router_hidden_dim=0,
            always_use_router=True,
        )

        self.assertTrue(hasattr(layer, "in_router"))
        proj = layer.in_router.proj
        self.assertIsInstance(proj, torch.nn.Linear)

        with torch.no_grad():
            proj.weight.zero_()
            proj.bias.zero_()
            # 2 input features: [coord, mean]; let logits depend on mean only.
            proj.weight[:, 1] = torch.linspace(-1.0, 1.0, proj.weight.shape[0])

        x0 = torch.zeros(2, 5, 16, 16)
        x1 = torch.ones(2, 5, 16, 16)

        coords = torch.linspace(0.0, 1.0, x0.shape[1])
        coeff0 = layer.in_router(coords, channel_mean=x0.mean(dim=(2, 3)))
        coeff1 = layer.in_router(coords, channel_mean=x1.mean(dim=(2, 3)))
        self.assertFalse(torch.allclose(coeff0, coeff1))

        y0 = layer(x0)
        y1 = layer(x1)
        self.assertEqual(tuple(y0.shape), (2, 4, 16, 16))
        self.assertEqual(tuple(y1.shape), (2, 4, 16, 16))

    def test_adaptive_input_conv_mix_dynamic_depthwise_pool_changes(self) -> None:
        layer = AdaptiveInputConvLayer(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=True,
            mode="mix",
            router_condition="per_channel_mean_dw_pool",
            router_hidden_dim=0,
            always_use_router=True,
            router_dw_kernel_size=3,
        )

        self.assertTrue(hasattr(layer, "router_dw_kernel"))
        with torch.no_grad():
            # A simple Laplacian-like kernel to make constant regions produce near-zero response.
            layer.router_dw_kernel.copy_(
                torch.tensor(
                    [
                        [0.0, -1.0, 0.0],
                        [-1.0, 4.0, -1.0],
                        [0.0, -1.0, 0.0],
                    ]
                )
            )

        proj = layer.in_router.proj
        self.assertIsInstance(proj, torch.nn.Linear)
        with torch.no_grad():
            proj.weight.zero_()
            proj.bias.zero_()
            # 3 input features: [coord, mean, dw_pool]; let logits depend on dw_pool only.
            proj.weight[:, 2] = torch.linspace(-1.0, 1.0, proj.weight.shape[0])

        x_const = torch.ones(2, 5, 16, 16)
        x_checker = torch.ones(2, 5, 16, 16)
        x_checker[:, :, ::2, ::2] = 0.0
        x_checker[:, :, 1::2, 1::2] = 2.0
        self.assertTrue(torch.allclose(x_const.mean(dim=(2, 3)), x_checker.mean(dim=(2, 3))))

        coords = torch.linspace(0.0, 1.0, x_const.shape[1])
        mean_const = x_const.mean(dim=(2, 3))
        mean_checker = x_checker.mean(dim=(2, 3))

        k = int(layer.router_dw_kernel.shape[0])
        w_dw = layer.router_dw_kernel[None, None].repeat(x_const.shape[1], 1, 1, 1)
        dw_const = torch.nn.functional.conv2d(
            x_const, w_dw, bias=None, stride=1, padding=k // 2, groups=x_const.shape[1]
        )
        dw_checker = torch.nn.functional.conv2d(
            x_checker, w_dw, bias=None, stride=1, padding=k // 2, groups=x_checker.shape[1]
        )
        pool_const = dw_const.abs().mean(dim=(2, 3))
        pool_checker = dw_checker.abs().mean(dim=(2, 3))
        self.assertFalse(torch.allclose(pool_const, pool_checker))

        coeff_const = layer.in_router(coords, channel_mean=mean_const, channel_dw_pool=pool_const)
        coeff_checker = layer.in_router(coords, channel_mean=mean_checker, channel_dw_pool=pool_checker)
        self.assertFalse(torch.allclose(coeff_const, coeff_checker))

        y0 = layer(x_const)
        y1 = layer(x_checker)
        self.assertEqual(tuple(y0.shape), (2, 4, 16, 16))
        self.assertEqual(tuple(y1.shape), (2, 4, 16, 16))

    def test_adaptive_output_conv_mix_shape(self) -> None:
        layer = AdaptiveOutputConvLayer(
            in_channels=3,
            out_channels=8,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=True,
            mode="mix",
            router_hidden_dim=0,
        )
        x = torch.randn(2, 3, 8, 8)
        y = layer(x, out_channels=5)
        self.assertEqual(tuple(y.shape), (2, 5, 8, 8))
        y2 = layer(x, out_channels=10)
        self.assertEqual(tuple(y2.shape), (2, 10, 8, 8))

    def test_adaptive_output_conv_mix_dynamic_depthwise_pool_changes(self) -> None:
        layer = AdaptiveOutputConvLayer(
            in_channels=4,
            out_channels=8,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=True,
            mode="mix",
            router_condition="per_channel_dw_pool",
            router_hidden_dim=0,
            router_dw_kernel_size=3,
        )

        self.assertTrue(hasattr(layer, "router_dw_kernel"))
        with torch.no_grad():
            layer.router_dw_kernel.copy_(
                torch.tensor(
                    [
                        [0.0, -1.0, 0.0],
                        [-1.0, 4.0, -1.0],
                        [0.0, -1.0, 0.0],
                    ]
                )
            )

        proj = layer.out_router.proj
        self.assertIsInstance(proj, torch.nn.Linear)
        with torch.no_grad():
            proj.weight.zero_()
            proj.bias.zero_()
            proj.weight[:, 1] = torch.linspace(-1.0, 1.0, proj.weight.shape[0])

        x_const = torch.ones(2, 4, 16, 16)
        x_checker = torch.ones(2, 4, 16, 16)
        x_checker[:, :, ::2, ::2] = 0.0
        x_checker[:, :, 1::2, 1::2] = 2.0

        y0 = layer(x_const, out_channels=5)
        y1 = layer(x_checker, out_channels=5)
        self.assertEqual(tuple(y0.shape), (2, 5, 16, 16))
        self.assertEqual(tuple(y1.shape), (2, 5, 16, 16))
        self.assertFalse(torch.allclose(y0, y1))

    def test_adaptive_input_conv_sitok_shapes_and_channel_embed(self) -> None:
        layer = AdaptiveInputConvLayer(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            mode="sitok",
            sitok_reduce="none",
            sitok_embed_scale=1.0,
        )
        x = torch.ones(2, 5, 8, 8)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 5 * 4, 8, 8))

        # With shared per-channel conv, the only difference across channel chunks should come from channel embedding.
        y5d = y.reshape(2, 5, 4, 8, 8)
        self.assertFalse(torch.allclose(y5d[:, 0], y5d[:, 1]))

        layer2 = AdaptiveInputConvLayer(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            mode="sitok",
            sitok_reduce="mean",
            sitok_embed_scale=1.0,
        )
        y2 = layer2(x)
        self.assertEqual(tuple(y2.shape), (2, 4, 8, 8))

        layer3 = AdaptiveInputConvLayer(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            mode="sitok",
            sitok_reduce="pointwise",
            sitok_embed_scale=1.0,
        )
        assert isinstance(layer3.sitok_reduce_head, torch.nn.Sequential)
        with torch.no_grad():
            lin0 = layer3.sitok_reduce_head[0]
            lin1 = layer3.sitok_reduce_head[2]
            assert isinstance(lin0, torch.nn.Linear)
            assert isinstance(lin1, torch.nn.Linear)
            lin0.weight.zero_()
            lin0.bias.zero_()
            lin0.weight.copy_(torch.eye(lin0.out_features))
            lin1.weight.zero_()
            lin1.bias.zero_()
            lin1.weight[0, 0] = 1.0

        x0 = torch.zeros(2, 5, 8, 8)
        x0[:, 0] = 1.0
        x1 = torch.zeros(2, 5, 8, 8)
        x1[:, 1] = 1.0
        y3 = layer3(x0)
        y4 = layer3(x1)
        self.assertEqual(tuple(y3.shape), (2, 4, 8, 8))
        self.assertFalse(torch.allclose(y3, y4))

    def test_adaptive_output_conv_sitok_film_shape(self) -> None:
        layer = AdaptiveOutputConvLayer(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=True,
            mode="sitok_film",
            sitok_embed_dim=8,
            sitok_hidden_dim=0,
        )
        with torch.no_grad():
            layer.sitok_head.weight.zero_()
            layer.sitok_head.bias.zero_()
            layer.sitok_head.bias[0] = 1.0  # gamma base
            layer.sitok_head.weight[0, 0] = 1.0  # gamma depends on embedding dim 0

        x = torch.randn(2, 8, 8, 8)
        y = layer(x, out_channels=5)
        self.assertEqual(tuple(y.shape), (2, 5, 8, 8))
        # Different output channels should not be identical (FiLM is index-dependent).
        self.assertFalse(torch.allclose(y[:, 0], y[:, 1]))

    def test_adaptive_output_conv_sitok_pointwise_shape(self) -> None:
        layer = AdaptiveOutputConvLayer(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=True,
            mode="sitok_pointwise",
            sitok_embed_dim=8,
            sitok_hidden_dim=0,
            sitok_basis_dim=4,
        )
        with torch.no_grad():
            layer.sitok_head.weight.zero_()
            layer.sitok_head.bias.zero_()
            layer.sitok_head.weight[0, 0] = 1.0  # basis weight 0 depends on embedding dim 0

        x = torch.randn(2, 8, 8, 8)
        y = layer(x, out_channels=5)
        self.assertEqual(tuple(y.shape), (2, 5, 8, 8))
        self.assertFalse(torch.allclose(y[:, 0], y[:, 1]))

    def test_adaptive_input_linear_sitok_pointwise_shape(self) -> None:
        layer = AdaptiveInputLinearLayer(
            in_features=12,
            out_features=6,
            bias=True,
            mode="sitok",
            sitok_group_size=4,
            sitok_reduce="pointwise",
            sitok_embed_scale=1.0,
        )
        assert isinstance(layer.sitok_reduce_head, torch.nn.Linear)
        with torch.no_grad():
            layer.sitok_reduce_head.weight.zero_()
            layer.sitok_reduce_head.bias.zero_()
            layer.sitok_reduce_head.weight[0, 0] = 1.0

        x0 = torch.zeros(2, 3 * 4)
        x0[:, 0:4] = 1.0
        x1 = torch.zeros(2, 3 * 4)
        x1[:, 4:8] = 1.0
        y0 = layer(x0)
        y1 = layer(x1)
        self.assertEqual(tuple(y0.shape), (2, 6))
        self.assertFalse(torch.allclose(y0, y1))

    def test_adaptive_output_linear_sitok_film_shape(self) -> None:
        layer = AdaptiveOutputLinearLayer(
            in_features=5,
            out_features=12,
            bias=True,
            mode="sitok_film",
            sitok_group_size=3,
            sitok_embed_dim=8,
            sitok_hidden_dim=0,
        )
        with torch.no_grad():
            layer.linear.weight.fill_(0.1)
            if layer.linear.bias is not None:
                layer.linear.bias.zero_()
            assert isinstance(layer.sitok_head, torch.nn.Linear)
            layer.sitok_head.weight.zero_()
            layer.sitok_head.bias.zero_()
            layer.sitok_head.bias[:3] = 1.0
            layer.sitok_head.weight[0, 0] = 1.0

        x = torch.ones(2, 5)
        y = layer(x, out_features=2 * 3)
        self.assertEqual(tuple(y.shape), (2, 6))
        y2 = y.reshape(2, 2, 3)
        self.assertFalse(torch.allclose(y2[:, 0], y2[:, 1]))

    def test_adaptive_output_linear_sitok_pointwise_shape(self) -> None:
        layer = AdaptiveOutputLinearLayer(
            in_features=5,
            out_features=12,
            bias=True,
            mode="sitok_pointwise",
            sitok_group_size=3,
            sitok_embed_dim=8,
            sitok_hidden_dim=0,
            sitok_basis_dim=4,
        )
        with torch.no_grad():
            layer.linear.weight.zero_()
            if layer.linear.bias is not None:
                layer.linear.bias.zero_()
                layer.linear.bias[0] = 1.0
            assert isinstance(layer.sitok_head, torch.nn.Linear)
            layer.sitok_head.weight.zero_()
            layer.sitok_head.bias.zero_()
            layer.sitok_head.weight[0, 0] = 1.0

        x = torch.ones(2, 5)
        y = layer(x, out_features=2 * 3)
        self.assertEqual(tuple(y.shape), (2, 6))
        y2 = y.reshape(2, 2, 3)
        self.assertFalse(torch.allclose(y2[:, 0], y2[:, 1]))


if __name__ == "__main__":
    unittest.main()
