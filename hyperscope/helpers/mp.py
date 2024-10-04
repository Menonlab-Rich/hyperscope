def wait_for_memory(threshold_mb=400, check_interval=1):
    import psutil
    from time import sleep
    """
    Pause the current thread until available memory is above the threshold.
    
    :param threshold_mb: Minimum available memory in MB to continue execution.
    :param check_interval: Interval in seconds to check memory availability.
    """
    while True:
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        if available_memory_mb >= threshold_mb:
            break
        sleep(check_interval)
