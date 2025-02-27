import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from memory_profiling import *
from dataset import *
from model import SimplifiedSAMLightningModule
import os
import torch

def train_sam_memory_efficient(config):
    """
    Memory-efficient training function for SAM
    
    Args:
        config: Configuration object with paths to data and checkpoint
    """
    # Set PyTorch memory management options
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Print initial GPU memory state
    print_gpu_memory("Initial")
    
    # Configure logger
    logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/SAM",
        tags=["finetuning", "memory-optimized"],
    )
    
    # Initialize SAM transform
    sam_transform = ResizeLongestSide(1024)  # Standard SAM input size
    
    # Create dataset and dataloader
    dataset = OptimizedSAMDataset(
        image_dir=config.img_dir,
        mask_dir=config.mask_dir,
        transform=SimpleTransform(sam_transform),
    )
    
    # Create very small batch dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Use batch size of 1
        shuffle=True,
        num_workers=7,  
    )
    
    print_gpu_memory("After dataset/dataloader creation")
    
    # Create model
    model = SimplifiedSAMLightningModule(config.checkpoint)
    
    # Configure trainer with memory optimizations
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        precision=32,  # Use mixed precision
        accumulate_grad_batches=4,  # Accumulate gradients
        gradient_clip_val=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="train_loss",
                dirpath="checkpoints",
                filename="sam-memopt-{epoch:02d}-{train_loss:.2f}",
                save_top_k=1,  # Save fewer checkpoints
                mode="min",
            ),
            pl.callbacks.EarlyStopping(
                monitor="train_loss", 
                patience=3,
                mode="min"
            ),
        ],
        logger=logger,
        log_every_n_steps=10,  # Log less frequently
        enable_checkpointing=True,
        # Additional memory optimizations
        enable_model_summary=False,
        inference_mode=False,  # More aggressive memory handling
    )
    
    # Print memory before training
    print_gpu_memory("Before training")
    
    # Start training
    trainer.fit(model, dataloader)
    
    # Clean up and print final memory usage
    free_memory()
    print_gpu_memory("After training")

# Example usage
if __name__ == "__main__":
    from config import config
    train_sam_memory_efficient(config)
