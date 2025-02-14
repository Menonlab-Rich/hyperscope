def wait_for_memory(threshold_mb=400, check_interval=1, device='CPU'):
    import psutil
    from time import sleep
    """
    Pause the current thread until available memory is above the threshold.
    
    :param threshold_mb: Minimum available memory in MB to continue execution.
    :param check_interval: Interval in seconds to check memory availability.
    """
    if device == 'CPU':
        while True:
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            if available_memory_mb >= threshold_mb:
                break
            sleep(check_interval)
    elif device == 'CUDA':
        from torch import cuda
        while True:
            alloc = cuda.memory_allocated()
            reserved = cuda.memory_allocated()
            available = alloc - reserved
            if available >= threshold_mb:
                break
            sleep(check_interval)
