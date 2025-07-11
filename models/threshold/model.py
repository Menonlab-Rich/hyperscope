import math  # For sqrt to reshape features
import warnings

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SwinForImageClassification


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.out_channels = out_channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # The input to DoubleConv is the sum of channels from the previous 
            # decoder layer and the skip connection from the encoder.
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is from the previous decoder layer, x2 is the skip connection
        x1 = self.up(x1)
        
        # Pad x1 to match the spatial dimensions of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
# In threshold/model.py

class Decoder(nn.Module):
    def __init__(self, in_shape, encoder_hidden_sizes, num_classes=1, bilinear=True):
        super().__init__()
        
        self.in_height, self.in_width = in_shape
        self.encoder_hidden_sizes = encoder_hidden_sizes
        
        # Define the number of channels for each decoder stage

        # Up1: Processes encoder_fm[3] (1024 ch) and combines with encoder_fm[2] (512 ch)
        self.up1 = Up(encoder_hidden_sizes[3], encoder_hidden_sizes[2], encoder_hidden_sizes[2], bilinear)
        
        # Up2: Processes output from up1 and combines with encoder_fm[1]
        self.up2 = Up(encoder_hidden_sizes[2], encoder_hidden_sizes[1], encoder_hidden_sizes[1], bilinear)

        # Up3: Processes output from up2 and combines with encoder_fm[0]
        self.up3 = Up(encoder_hidden_sizes[1], encoder_hidden_sizes[0], encoder_hidden_sizes[0], bilinear)
        
        # Further upsampling to reach the original input resolution
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(encoder_hidden_sizes[0], encoder_hidden_sizes[0] // 2)
        )
        
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(encoder_hidden_sizes[0] // 2, 64)
        )
        
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, encoder_features):
        encoder_fm = []
        downsample_factors = [8, 16, 32, 32]

        for i in range(len(downsample_factors)):
            h_state = encoder_features[i + 1]
            b, n, c = h_state.shape
            h_feat = self.in_height // downsample_factors[i]
            w_feat = self.in_width // downsample_factors[i]
            
            if h_feat * w_feat != n:
                warnings.warn(f"Feature map at stage {i} has {n} patches, which does not match the expected size of {h_feat}x{w_feat}. Reshaping based on 'n'.")
                h_feat = int(math.sqrt(n))
                w_feat = h_feat

            encoder_fm.append(h_state.permute(0, 2, 1).reshape(b, c, h_feat, w_feat))

        # Decoder path
        x = self.up1(encoder_fm[3], encoder_fm[2])
        x = self.up2(x, encoder_fm[1])
        x = self.up3(x, encoder_fm[0])
        x = self.up4(x)
        x = self.up5(x)

        contrastive_features = x
        return self.out_conv(x), contrastive_features

class Threshold(nn.Module):
    def __init__(self, 
                 encoder_name: str = "microsoft/swin-base-patch4-window7-224-in22k",
                 num_classes: int = 1, # For binary thresholding
                 out_shape: tuple = (224, 224), # Default for Swin-base-224
                 bilinear: bool = True):
        super().__init__()
        
        self.out_shape = tuple(out_shape)
        
        # Encoder
        self.encoder = SwinForImageClassification.from_pretrained(encoder_name, ignore_mismatched_sizes = True)
        
        # Configure encoder to always return hidden states
        self.encoder.config.output_hidden_states = True
        base_hidden_size = self.encoder.config.hidden_size // 8  # This is the base channel dimension, e.g., 128
        num_encoder_stages = len(self.encoder.config.depths) # The number of stages, e.g., 4
        encoder_hidden_sizes = [256, 512, 1024, 1024]

        # Pass the desired input shape to the decoder for dynamic feature map reshaping
        self.decoder = Decoder(out_shape, encoder_hidden_sizes, num_classes, bilinear) # Using out_shape as in_shape for decoder


    def forward(self, x):
        # x is the input image tensor (batch_size, channels, height, width)
        
        encoder_outputs = self.encoder(x, output_hidden_states=True) 
        
        # `encoder_outputs.hidden_states` is a tuple/list of tensors.
        # It includes the initial patch embedding output (index 0) and then the
        # output of each of the 4 stages (indices 1 to 4).
        # We pass the entire list to the decoder, and the decoder selects the relevant ones.
        # (The decoder's `forward` now correctly picks `encoder_features[i+1]` to get stage outputs)
        
        logits, contrastive_features = self.decoder(encoder_outputs.hidden_states)
        

        # Resize to the specified out_shape if different from the decoder's output resolution (if needed)
        # The decoder is designed to output at `in_shape` (passed during initialization)
        if logits.shape[2:] != self.out_shape:
            logits = F.interpolate(logits, size=self.out_shape, mode='bilinear', align_corners=False)
            
        return logits, contrastive_features

# --- Example Usage ---
if __name__ == "__main__":
    from PIL import Image

    # 1. Load Image and Process
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
    
    # Process the image to get the pixel values tensor.
    # The processor resizes to 224x224 by default for this model.
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values # Shape: (1, 3, 224, 224)

    print(f"Input image shape: {pixel_values.shape}")

    # 2. Instantiate Model
    # Pass the input image's height and width as out_shape
    input_h, input_w = pixel_values.shape[2:]
    model = Threshold(out_shape=(input_h, input_w))
    model.eval() 

    # 3. Forward Pass
    with torch.no_grad():
        output_probability_map = model(pixel_values)

    # 4. Inspect Output
    print(f"Output probability map shape: {output_probability_map.shape}")
    
    # Assert that the output matches the specified out_shape (which is input_h, input_w here)
    assert output_probability_map.shape[2:] == model.out_shape

    binary_mask = (output_probability_map > 0.5).float()
    print(f"Binary mask shape: {binary_mask.shape}")

    # Test with a different input size (e.g., 256x256)
    print("\n--- Testing with 256x256 input ---")
    
    # Note: Swin models are typically designed for fixed input sizes like 224 or 384.
    # If you feed a different size, the internal patch embeddings and number of patches
    # will change, but the model's architecture (number of attention heads, layers, etc.)
    # remains fixed. The `AutoImageProcessor` *will* resize your input to the model's
    # expected size (e.g., 224x224) by default, so you rarely need to handle
    # arbitrary input sizes dynamically at the model level for HuggingFace pre-trained models.
    # However, if you explicitly disable resizing in the processor or use a different model,
    # this dynamic calculation becomes more critical.

    # For demonstration, let's create a dummy tensor of 256x256 and process it.
    # In a real scenario, you'd configure the processor to handle 256x256.
    dummy_input_256 = torch.randn(1, 3, 256, 256)
    
    # Instantiate processor for 256x256 for a more realistic test
    # (Note: Swin base 224 model is optimized for 224. Running it on 256 might not be optimal
    # without fine-tuning or picking a Swin variant trained on larger images).
    processor_256 = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k", size={"height": 256, "width": 256})
    inputs_256 = processor_256(images=image, return_tensors="pt")
    pixel_values_256 = inputs_256.pixel_values # Shape: (1, 3, 256, 256)

    print(f"Input image shape (256): {pixel_values_256.shape}")
    
    model_256 = Threshold(out_shape=(pixel_values_256.shape[2], pixel_values_256.shape[3]))
    model_256.eval()

    with torch.no_grad():
        output_probability_map_256 = model_256(pixel_values_256)
    
    print(f"Output probability map shape (256): {output_probability_map_256.shape}")
    assert output_probability_map_256.shape[2:] == model_256.out_shape
