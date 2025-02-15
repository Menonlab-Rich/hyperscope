def wait_for_memory(threshold_mb=400, check_interval=1, device='CPU'):
    import psutil
    from time import sleep
    
    if device == 'CPU':
        while True:
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            if available_memory_mb >= threshold_mb:
                break
            sleep(check_interval)
    elif device == 'CUDA':
        import torch
        while True:
            # Get total and reserved memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            allocated_memory = torch.cuda.memory_allocated(0)
            
            # Calculate free memory in MB
            free_memory_mb = (total_memory - reserved_memory - allocated_memory) / (1024 * 1024)
            
            if free_memory_mb >= threshold_mb:
                break
            
            # Force garbage collection if memory is tight
            if free_memory_mb < threshold_mb / 2:
                torch.cuda.empty_cache()
            
            sleep(check_interval)
