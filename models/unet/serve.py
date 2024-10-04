from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import torch
import torch.nn.functional as F
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
from albumentations.pytorch import ToTensorV2
import albumentations as A
from model import UNetLightning  # Import your model
from image_processing.image_processor import ImageProcessor
import logging

logging.basicConfig(level=logging.INFO)
global logger; 
logger = logging.getLogger(__name__) # Create a logger

processor = ImageProcessor(superpixel_size=32, patch_size=128)

app = FastAPI()

# Load the model once during startup
model = UNetLightning.load_from_checkpoint(
    r"D:\CZI_scope\code\ml_models\unet_2\unet-UN-200-epoch=07-val_dice=0.89.ckpt")
model.eval()  # Set the model to evaluation mode

# Define the validation transformations
val_transform = A.Compose([
    A.ToFloat(always_apply=True),
    A.Resize(128, 128),
    ToTensorV2(),
])

# Define a request model for input image
class ImageRequest(BaseModel):
    image: bytes

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global logger
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure the image is in the correct mode (16-bit grayscale)
        if image.mode != 'I;16':
            image = image.convert('I;16')

        image = np.array(image)
        
        # Process the image to get patches and coordinates
        processed_image, mask = processor.process_image(image)
        if processed_image is None or mask is None:
            JSONResponse(status_code=422, content={"message": "Failed to process the image."})
        
        rmin, rmax, cmin, cmax = processor.find_bounding_box(mask)
        rmin, rmax, cmin, cmax = processor.pad_to_patch_size(rmin, rmax, cmin, cmax)
        patches, _, coords = processor.extract_patches(processed_image, mask, rmin, rmax, cmin, cmax)
        # Transform and batch the patches
        batch_tensors = []
        for patch in patches:
            transformed = val_transform(image=patch, mask=np.zeros_like(patch))
            image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
            batch_tensors.append(image_tensor)
            

        batch_tensors = torch.cat(batch_tensors, dim=0)  # Stack into a single batch tensor

        # Perform inference on the batch
        with torch.no_grad():
            outputs = model(batch_tensors)
        
        # Process the outputs
        results = F.softmax(outputs, dim=1).argmax(dim=1).cpu().numpy()

        dtype = np.dtype([('result', np.float64, (128, 128)), ('x', np.int32), ('y', np.int32)])
        enriched_results = np.empty(len(results), dtype=dtype)

        # Populate the structured array
        for i in range(len(results)):
            enriched_results[i] = (results[i], coords[i][0], coords[i][1])

                
        # Save the numpy array to a bytes buffer
        buffer = io.BytesIO()
        np.save(buffer, enriched_results)
        buffer.seek(0)

        return StreamingResponse(buffer, media_type='application/octet-stream', headers={"Content-Disposition": "attachment; filename=result.npy"})
    except Exception as e:
        import traceback
        logger.error(f"An error occurred during prediction: {e}.")
        logger.error(f"The traceback is: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": "An error occurred during prediction."})

@app.on_event("startup")
async def startup_event():
    use_uvicorn_logger()
    
def use_uvicorn_logger():
    global logger
    logger = logging.getLogger("uvicorn")
    
if __name__ == "__main__":
    import uvicorn
    use_uvicorn_logger()
    uvicorn.run(app, host="0.0.0.0", port=8000)
