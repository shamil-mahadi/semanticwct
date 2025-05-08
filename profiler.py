#!/usr/bin/env python3
"""
Comprehensive profiling script for neural style transfer project
"""

import os
import time
import psutil
import torch
import gc
import threading
import sys
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple
import tracemalloc
import cProfile
import pstats
import io
from memory_profiler import profile
import py3nvml.py3nvml as nvml

# Import your project modules
from pipeline import TransferPipeline

class PerformanceProfiler:
    """Comprehensive performance profiler for the style transfer pipeline"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.stats = {}
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            try:
                nvml.nvmlInit()
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_available = False
    
    @contextmanager
    def profile_memory(self, name: str):
        """Context manager for memory profiling"""
        tracemalloc.start()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        if self.gpu_available:
            torch.cuda.empty_cache()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        yield
        
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if self.gpu_available:
            end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_used = end_gpu_memory - start_gpu_memory
        else:
            gpu_memory_used = 0
        
        self.stats[name] = {
            'memory_used_mb': end_memory - start_memory,
            'peak_memory_mb': peak / 1024 / 1024,
            'gpu_memory_used_mb': gpu_memory_used,
            'total_memory_mb': end_memory
        }
    
    @contextmanager
    def profile_time(self, name: str):
        """Context manager for time profiling"""
        start_time = time.time()
        start_cpu_time = time.process_time()
        
        yield
        
        end_time = time.time()
        end_cpu_time = time.process_time()
        
        if name not in self.stats:
            self.stats[name] = {}
        
        self.stats[name].update({
            'wall_time_s': end_time - start_time,
            'cpu_time_s': end_cpu_time - start_cpu_time,
            'cpu_usage_percent': psutil.cpu_percent(interval=None)
        })
    
    def monitor_system_resources(self, duration: int = 60, interval: float = 1.0):
        """Monitor system resources over time"""
        timestamps = []
        cpu_usage = []
        memory_usage = []
        gpu_usage = []
        gpu_memory = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            timestamps.append(time.time() - start_time)
            cpu_usage.append(psutil.cpu_percent(interval=None))
            memory_usage.append(self.process.memory_info().rss / 1024 / 1024)
            
            if self.gpu_available:
                try:
                    gpu_info = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_usage.append(gpu_info.gpu)
                    gpu_memory.append(memory_info.used / 1024 / 1024)
                except:
                    gpu_usage.append(0)
                    gpu_memory.append(0)
            else:
                gpu_usage.append(0)
                gpu_memory.append(0)
            
            time.sleep(interval)
        
        return {
            'timestamps': timestamps,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage,
            'gpu_memory': gpu_memory
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024,
            })
        
        return info
    
    def profile_function_calls(self, func, *args, **kwargs):
        """Profile function calls with cProfile"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        return result, s.getvalue()
    
    def print_stats(self):
        """Print comprehensive statistics"""
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING RESULTS")
        print("="*80)
        
        # System info
        print("\nSYSTEM INFORMATION:")
        print("-" * 40)
        sys_info = self.get_system_info()
        for key, value in sys_info.items():
            print(f"{key}: {value}")
        
        # Profiling stats
        print("\nPROFILING STATISTICS:")
        print("-" * 40)
        for name, stats in self.stats.items():
            print(f"\n{name}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")


@profile  # This decorator requires memory_profiler
def profile_pipeline_execution():
    """Profile the main pipeline execution"""
    
    profiler = PerformanceProfiler()
    
    # Initialize pipeline
    print("Initializing pipeline...")
    with profiler.profile_memory("initialization"):
        with profiler.profile_time("initialization"):
            pipeline = TransferPipeline()
    
    # Load test images
    content_path = "content/1.jpg"
    style_path = "styles/1.jpg"
    
    if not (os.path.exists(content_path) and os.path.exists(style_path)):
        print("Warning: Test images not found. Creating dummy data...")
        # Create dummy images for testing
        import numpy as np
        import cv2
        
        dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        os.makedirs("content", exist_ok=True)
        os.makedirs("styles", exist_ok=True)
        cv2.imwrite(content_path, dummy_img)
        cv2.imwrite(style_path, dummy_img)
    
    with open(content_path, "rb") as f:
        content_bytes = f.read()
    with open(style_path, "rb") as f:
        style_bytes = f.read()
    
    # Profile different configurations
    configs = [
        {"do_segmentation": False, "strength": 1.0, "name": "no_segmentation"},
        {"do_segmentation": True, "strength": 1.0, "grounding_prompt": "building. sky.", "name": "with_segmentation"},
        {"do_segmentation": False, "strength": 1.5, "name": "high_strength"},
    ]
    
    for config in configs:
        name = config.pop("name")
        print(f"\nProfiling configuration: {name}")
        
        with profiler.profile_memory(f"transfer_{name}"):
            with profiler.profile_time(f"transfer_{name}"):
                try:
                    result = pipeline.transfer(content_bytes, style_bytes, **config)
                    print(f"Transfer completed successfully for {name}")
                except Exception as e:
                    print(f"Error in {name}: {e}")
    
    # Profile individual components
    print("\nProfiling individual components...")
    
    # Test image preprocessing
    import numpy as np
    import cv2
    test_img = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    
    with profiler.profile_memory("preprocessing"):
        with profiler.profile_time("preprocessing"):
            for _ in range(10):
                processed = pipeline._preprocess_img(test_img, size=512)
    
    # Test tensor conversion
    with profiler.profile_memory("tensor_conversion"):
        with profiler.profile_time("tensor_conversion"):
            for _ in range(10):
                tensor = pipeline._cv2_to_tensor(processed)
    
    # Print results
    profiler.print_stats()


def profile_with_line_profiler():
    """Profile line-by-line execution (requires line_profiler)"""
    print("\nRunning line-by-line profiling...")
    print("Note: Install line_profiler with: pip install line_profiler")
    print("Then run: kernprof -l -v profiling_script.py")


def benchmark_different_sizes():
    """Benchmark performance with different image sizes"""
    print("\nBenchmarking different image sizes...")
    
    pipeline = TransferPipeline()
    profiler = PerformanceProfiler()
    
    # Create test images of different sizes
    sizes = [256, 512, 768, 1024]
    
    for size in sizes:
        print(f"\nTesting size: {size}x{size}")
        
        # Create dummy images
        import numpy as np
        import cv2
        
        dummy_img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        _, content_bytes = cv2.imencode('.jpg', dummy_img)
        _, style_bytes = cv2.imencode('.jpg', dummy_img)
        
        content_bytes = content_bytes.tobytes()
        style_bytes = style_bytes.tobytes()
        
        with profiler.profile_memory(f"size_{size}"):
            with profiler.profile_time(f"size_{size}"):
                try:
                    result = pipeline.transfer(content_bytes, style_bytes, do_segmentation=False)
                    print(f"✓ Size {size} completed")
                except Exception as e:
                    print(f"✗ Size {size} failed: {e}")
    
    profiler.print_stats()


def monitor_real_time_usage():
    """Monitor real-time system usage during execution"""
    print("\nMonitoring real-time system usage...")
    
    profiler = PerformanceProfiler()
    
    # Start monitoring in a separate thread
    monitoring_data = {}
    
    def monitor():
        monitoring_data.update(profiler.monitor_system_resources(duration=30, interval=0.5))
    
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    
    # Run some operations while monitoring
    pipeline = TransferPipeline()
    
    # Create dummy data
    import numpy as np
    import cv2
    
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    _, content_bytes = cv2.imencode('.jpg', dummy_img)
    _, style_bytes = cv2.imencode('.jpg', dummy_img)
    
    content_bytes = content_bytes.tobytes()
    style_bytes = style_bytes.tobytes()
    
    # Run multiple transfers
    for i in range(3):
        print(f"Running transfer {i+1}/3...")
        try:
            result = pipeline.transfer(content_bytes, style_bytes, do_segmentation=False)
            time.sleep(2)  # Brief pause between transfers
        except Exception as e:
            print(f"Transfer {i+1} failed: {e}")
    
    monitor_thread.join()
    
    # Print monitoring results
    if monitoring_data:
        print(f"\nMonitoring Results:")
        print(f"Peak CPU usage: {max(monitoring_data['cpu_usage']):.1f}%")
        print(f"Peak memory usage: {max(monitoring_data['memory_usage']):.1f} MB")
        if profiler.gpu_available:
            print(f"Peak GPU usage: {max(monitoring_data['gpu_usage']):.1f}%")
            print(f"Peak GPU memory: {max(monitoring_data['gpu_memory']):.1f} MB")


if __name__ == "__main__":
    print("Starting comprehensive profiling...")
    
    # Main profiling
    profile_pipeline_execution()
    
    # Additional benchmarks
    benchmark_different_sizes()
    
    # Real-time monitoring
    monitor_real_time_usage()
    
    print("\nProfiling complete!")
    print("\nAdditional profiling tools you can use:")
    print("1. PyTorch Profiler: Use torch.profiler.profile()")
    print("2. line_profiler: Run 'kernprof -l -v profiling_script.py'")
    print("3. py-spy: Run 'py-spy record -o profile.svg -- python your_script.py'")
    print("4. TensorBoard: For detailed PyTorch profiling")