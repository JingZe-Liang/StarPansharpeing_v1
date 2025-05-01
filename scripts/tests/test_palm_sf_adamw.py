import heavyball
import torch
from heavyball import PaLMSFAdamW
from timm.models.resnet import resnet101

heavyball.utils.compile_mode = None
model = resnet101(pretrained=False).cuda()
optimizer = PaLMSFAdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    mars=True,
    betas=(0.95, 0.99),
    eps=1e-8,
    palm=True,
)
print(optimizer.__class__.__module__)
# optimizer = ForeachSOLP(
#     model.parameters(),
#     lr=1e-3,
#     weight_decay=1e-4,
#     betas=(0.95, 0.99),
#     eps=1e-8,
#     palm=True,
#     storage_dtype="bfloat16",
#     stochastic_schedule=True,
#     precondition_frequency=5,
# )
# optimizer = PaLMForeachSOAP(
#     model.parameters(),
#     lr=1e-3,
#     weight_decay=1e-4,
#     betas=(0.95, 0.99),
#     eps=1e-8,
#     palm=True,
#     storage_dtype="bfloat16",
#     stochastic_schedule=True,
#     precondition_frequency=5,
# )

print("Optimizer initialized with parameters")


# Test the optimizer with a dummy input
def test_optimizer():
    # Create a dummy input tensor
    input_tensor = torch.randn(4, 3, 224, 224).cuda()
    target_tensor = torch.randn(4, 1000).cuda()

    # Forward pass
    output = model(input_tensor)
    loss = torch.nn.functional.mse_loss(output, target_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check if the parameters have been updated
    for param in model.parameters():
        assert param.grad is not None, "Gradient is None"

    # Update parameters
    optimizer.step()


if __name__ == "__main__":
    test_optimizer()
    print("Optimizer test passed.")
