


import cupy as cp


class GPUStateSnapshot:
    def __init__(self):
        self.gpu_count = 0
        self.gpu_name = "N/A"
        self.gpu_memory_total = 0
        self.gpu_memory_available = 0
        self.gpu_memory_used_percent = 0.0

        self._gather_gpu_info()

    def _gather_gpu_info(self):
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                props = cp.cuda.runtime.getDeviceProperties(0)
                self.gpu_count = device_count
                self.gpu_name = props['name'].decode('utf-8')
                self.gpu_memory_total = props['totalGlobalMem'] // (1024**3)

                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                used_gb = used_bytes // (1024**3)
                self.gpu_memory_available = self.gpu_memory_total - used_gb
                self.gpu_memory_used_percent = (used_gb / self.gpu_memory_total * 100)

        except Exception:
            # GPU info unavailable, keep default values
            pass

    def provide_snapshot(self) -> dict:
        return {
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
            "gpu_memory_total": self.gpu_memory_total,
            "gpu_memory_available": self.gpu_memory_available,
            "gpu_memory_used_percent": self.gpu_memory_used_percent,
        }