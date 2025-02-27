import torch
import gc

def print_gpu_memory(label="Current"):
    """Print GPU memory usage at a specific point"""
    return # comment this line for debugging
    torch.cuda.synchronize()
    print(f"\n[{label}] Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"[{label}] Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"[{label}] Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def free_memory():
    """Attempt to free GPU memory"""
    return # comment this line if we are running out of memory often
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
