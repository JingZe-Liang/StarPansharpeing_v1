from .transformer import Transformer
from .vitamin_conv import ConvCfg, VitaminCfg, VitaminModel

# Model
"""
Low-level architecture:

Latent ----> Transformers (heavy) ----> De-tokenizer ---> image
                ⬇️ (conditioning)
Image -------> VitaminModel ----> image


High-level architecture:
Latent ------------|  (conditioning) ----|
Image ------> VitaminStages -----> VisionTransformer
        |           |------------------|
        ----> DinoV3 Stages            |
                    |                  |
                      All Stage features
                            |
                MultiScale Decoder (scales context?)
                            |
                         Unet Decoder
                            |----> Segmentation / classification map?
"""
