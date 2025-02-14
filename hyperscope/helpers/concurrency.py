from multiprocessing.pool import AsyncResult
from tqdm import tqdm
import time


class ConcurrentTqdm(tqdm):
    def __init__(self, *args, sequential=False, safe=True, callback=None, **kwargs):
        self.callback = self._callback if callback is None else callback
        self._strategy = "sequential" if sequential else "asynchronous"
        self._safe = safe
        
        super().__init__(*args, **kwargs)

    def unsafe(self):
        """Set the class to unsafe mode where exceptions are not checked."""
        self._safe = False
        return self
    
    def update(self, *args, res=(), **kwargs):
        super().update(*args, **kwargs)
        self.callback(*res, self)
    
    def _callback(self, *args):
        return None

    # Check if the iterator contains future objects
    def _is_future(self, iterator):
        return all(isinstance(i, AsyncResult) for i in iterator)

    def __iter__(self):
        if self._is_future(self.iterable):
            if self._strategy == "sequential":
                return (
                    self._iter_future_sequential_safe()
                    if self._safe
                    else self._iter_future_sequential_unsafe()
                )
            else:
                return (
                    self._iter_future_asynchronous_safe()
                    if self._safe
                    else self._iter_future_asynchronous_unsafe()
                )
        return super().__iter__()

    # Safe version of sequential iteration
    def _iter_future_sequential_safe(self):
        with self:
            for future in self.iterable:
                try:
                    result = future.get()
                    if result is not None:
                        to_yield = (True, result)
                except Exception as e:
                    to_yield = (False, e)
                yield to_yield
                self.update(res=to_yield)

    # Unsafe version of sequential iteration
    def _iter_future_sequential_unsafe(self):
        with self:
            for future in self.iterable:
                result = future.get()
                if result is not None:
                    to_yield = (result)
                yield to_yield
                self.update(res=to_yield)

    # Safe version of asynchronous iteration
    def _iter_future_asynchronous_safe(self):
        futures = list(self.iterable)
        with self:
            while futures:
                for future in futures[:]:
                    if future.ready():
                        try:
                            result = future.get()
                            to_yield = (True, result)
                        except Exception as e:
                            to_yield = (False, e)
                        futures.remove(future)
                        yield to_yield
                        self.update(res=to_yield)
                time.sleep(0.01)  # Short sleep to prevent high CPU usage

    # Unsafe version of asynchronous iteration
    def _iter_future_asynchronous_unsafe(self):
        futures = list(self.iterable)
        with self:
            while futures:
                for future in futures[:]:
                    if future.ready():
                        result = future.get()
                        if result is not None:
                            to_yield = (result)
                        futures.remove(future)
                        yield to_yield
                        self.update(res=to_yield)
                time.sleep(0.01)  # Short sleep to prevent high CPU usage
