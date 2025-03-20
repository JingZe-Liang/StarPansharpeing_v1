import torch
import torch.nn.functional as F
from torchvision.models import vgg16

# Load the pretrained VGG16 model and set it to evaluation mode
vgg = vgg16(pretrained=True).features.eval()


# Function to extract features from specified layers
def extract_features(model, x, layers):
    """
    Extract feature maps from specific layers of a model.

    Args:
        model: The PyTorch model (e.g., vgg.features).
        x: Input tensor (e.g., an image).
        layers: List of layer indices to extract features from.

    Returns:
        Dictionary mapping layer indices to their feature maps.
    """
    features = {}
    for idx, layer in enumerate(model):
        x = layer(x)
        if idx in layers:
            features[idx] = x
    return features


# Function to compute the Gram matrix
def gram_matrix(feature_maps):
    """
    Compute the Gram matrix for a batch of feature maps.

    Args:
        feature_maps: Tensor of shape (batch_size, channels, height, width).

    Returns:
        Gram matrix tensor.
    """
    batch_size, num_channels, height, width = feature_maps.size()
    features = feature_maps.view(batch_size, num_channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))  # Batch matrix multiplication
    return gram / (num_channels * height * width)  # Normalize


# Updated Gram loss function
def gram_loss(gen_img, orig_img, vgg_layers):
    """
    Compute the Gram matrix loss between generated and original images.

    Args:
        gen_img: Generated image tensor (batch_size, 3, height, width).
        orig_img: Original image tensor (batch_size, 3, height, width).
        vgg_layers: List of VGG layer indices to compute the loss on.

    Returns:
        Total Gram matrix loss.
    """
    # Extract features from both images
    gen_features = extract_features(vgg, gen_img, vgg_layers)
    orig_features = extract_features(vgg, orig_img, vgg_layers)

    # Initialize loss
    loss = 0.0

    # Compute loss for each specified layer
    for layer_idx in vgg_layers:
        gen_gram = gram_matrix(gen_features[layer_idx])
        orig_gram = gram_matrix(orig_features[layer_idx])
        loss += F.mse_loss(gen_gram, orig_gram)

    return loss


# Example usage
if __name__ == "__main__":
    # Dummy input images (batch_size=1, channels=3, height=224, width=224)
    gen_img = torch.randn(1, 3, 224, 224)
    orig_img = torch.randn(1, 3, 224, 224)

    # Specify VGG layers to use (e.g., conv layers at indices 3, 8, 15)
    vgg_layers = [3, 8, 15]

    # Compute the Gram loss
    loss = gram_loss(gen_img, orig_img, vgg_layers)
    print(f"Gram Matrix Loss: {loss.item()}")
