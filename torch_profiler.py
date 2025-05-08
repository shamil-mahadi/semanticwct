#!/usr/bin/env python3
"""
PyTorch-specific profiling for neural style transfer
"""

import torch
import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import cv2
import os
import time
from typing import Dict, Any, List
import json

from pipeline import TransferPipeline

class PyTorchProfiler:
    """PyTorch-specific performance profiler"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
    
    def profile_model_loading(self):
        """Profile model loading and compilation"""
        print("Profiling model loading...")
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_loading"):
                pipeline = TransferPipeline()
        
        # Save profiling results
        prof.export_chrome_trace("model_loading_trace.json")
        
        # Get memory statistics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_stats = torch.cuda.memory_stats()
            self.results['model_loading'] = {
                'allocated_memory_gb': memory_stats.get('allocated_bytes.all.peak', 0) / 1024**3,
                'reserved_memory_gb': memory_stats.get('reserved_bytes.all.peak', 0) / 1024**3,
                'active_memory_gb': memory_stats.get('active_bytes.all.peak', 0) / 1024**3,
            }
        
        return pipeline, prof
    
    def profile_inference(self, pipeline: TransferPipeline, content_bytes: bytes, style_bytes: bytes):
        """Profile inference with detailed breakdowns"""
        print("Profiling inference...")
        
        # Warm up
        print("Warming up...")
        for _ in range(2):
            try:
                _ = pipeline.transfer(content_bytes, style_bytes, do_segmentation=False)
            except:
                pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Profile inference
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        ) as prof:
            with record_function("full_inference"):
                result = pipeline.transfer(content_bytes, style_bytes, do_segmentation=False)
        
        # Export results
        prof.export_chrome_trace("inference_trace.json")
        
        # Get detailed statistics
        print("\nTop 10 CPU operations by total time:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        if torch.cuda.is_available():
            print("\nTop 10 CUDA operations by total time:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
            # Memory statistics
            memory_stats = torch.cuda.memory_stats()
            self.results['inference'] = {
                'allocated_memory_gb': memory_stats.get('allocated_bytes.all.peak', 0) / 1024**3,
                'reserved_memory_gb': memory_stats.get('reserved_bytes.all.peak', 0) / 1024**3,
                'active_memory_gb': memory_stats.get('active_bytes.all.peak', 0) / 1024**3,
            }
        
        return result, prof
    
    def profile_segmentation(self, pipeline: TransferPipeline, content_bytes: bytes, style_bytes: bytes):
        """Profile segmentation components"""
        print("Profiling segmentation...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("segmentation_inference"):
                result = pipeline.transfer(
                    content_bytes, 
                    style_bytes, 
                    do_segmentation=True,
                    grounding_prompt="building. sky. human."
                )
        
        prof.export_chrome_trace("segmentation_trace.json")
        
        if torch.cuda.is_available():
            memory_stats = torch.cuda.memory_stats()
            self.results['segmentation'] = {
                'allocated_memory_gb': memory_stats.get('allocated_bytes.all.peak', 0) / 1024**3,
                'reserved_memory_gb': memory_stats.get('reserved_bytes.all.peak', 0) / 1024**3,
                'active_memory_gb': memory_stats.get('active_bytes.all.peak', 0) / 1024**3,
            }
        
        return result, prof
    
    def profile_individual_components(self, pipeline: TransferPipeline):
        """Profile individual pipeline components"""
        print("Profiling individual components...")
        
        # Create dummy data
        dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        dummy_tensor = torch.randn(1, 3, 512, 512).to(self.device)
        
        components = {}
        
        # Profile encoder
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            with record_function("encoder_forward"):
                with torch.inference_mode():
                    encoded = pipeline.encoder(dummy_tensor)
        
        components['encoder'] = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
        
        # Profile decoder
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            with record_function("decoder_forward"):
                with torch.inference_mode():
                    decoded = pipeline.decoder(encoded)
        
        components['decoder'] = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
        
        # Profile WCT operation
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            with record_function("wct_operation"):
                from wct import perform_wct
                with torch.inference_mode():
                    wct_result = perform_wct(dummy_tensor, dummy_tensor)
        
        components['wct'] = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
        
        return components
    
    def benchmark_batch_sizes(self, pipeline: TransferPipeline):
        """Benchmark different batch sizes (if applicable)"""
        print("Benchmarking batch processing...")
        
        batch_results = {}
        
        for batch_size in [1, 2, 4]:
            print(f"Testing batch size: {batch_size}")
            
            # Create batch data
            dummy_imgs = []
            for _ in range(batch_size):
                dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                _, encoded = cv2.imencode('.jpg', dummy_img)
                dummy_imgs.append(encoded.tobytes())
            
            start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # Process batch
            for i in range(batch_size):
                try:
                    result = pipeline.transfer(dummy_imgs[i], dummy_imgs[0], do_segmentation=False)
                except Exception as e:
                    print(f"Error with batch size {batch_size}: {e}")
                    continue
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_results[batch_size] = {
                'total_time': end_time - start_time,
                'time_per_image': (end_time - start_time) / batch_size,
                'throughput': batch_size / (end_time - start_time)
            }
            
            if torch.cuda.is_available():
                memory_stats = torch.cuda.memory_stats()
                batch_results[batch_size]['memory_gb'] = memory_stats.get('allocated_bytes.all.peak', 0) / 1024**3
        
        return batch_results
    
    def analyze_model_efficiency(self, pipeline: TransferPipeline):
        """Analyze model efficiency and bottlenecks"""
        print("Analyzing model efficiency...")
        
        # Create dummy input
        dummy_tensor = torch.randn(1, 3, 512, 512).to(self.device)
        
        # Profile with different torch settings
        efficiency_results = {}
        
        # Test with different autocast settings
        if torch.cuda.is_available():
            print("Testing with autocast...")
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                start_time = time.time()
                with torch.inference_mode():
                    _ = pipeline.encoder(dummy_tensor)
                autocast_time = time.time() - start_time
            
            efficiency_results['autocast_speedup'] = autocast_time
        
        # Test memory efficiency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.inference_mode():
                _ = pipeline.encoder(dummy_tensor)
            
            memory_stats = torch.cuda.memory_stats()
            efficiency_results['encoder_memory_gb'] = memory_stats.get('allocated_bytes.all.peak', 0) / 1024**3
        
        return efficiency_results
    
    def generate_report(self):
        """Generate comprehensive profiling report"""
        print("\n" + "="*80)
        print("PYTORCH PROFILING REPORT")
        print("="*80)
        
        # System info
        print(f"\nDevice: {self.device}")
        print(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Results
        print("\nMemory Usage Results:")
        print("-" * 40)
        for component, stats in self.results.items():
            print(f"{component}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value:.3f}")
        
        print("\nProfiler traces saved:")
        print("- model_loading_trace.json")
        print("- inference_trace.json")
        print("- segmentation_trace.json")
        print("\nOpen these files in Chrome://tracing for detailed analysis")


def main():
    """Main profiling function"""
    profiler = PyTorchProfiler()
    
    # Create dummy test data
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    _, content_bytes = cv2.imencode('.jpg', dummy_img)
    _, style_bytes = cv2.imencode('.jpg', dummy_img)
    
    content_bytes = content_bytes.tobytes()
    style_bytes = style_bytes.tobytes()
    
    try:
        # Profile model loading
        pipeline, load_prof = profiler.profile_model_loading()
        
        # Profile inference
        result, inference_prof = profiler.profile_inference(pipeline, content_bytes, style_bytes)
        
        # Profile segmentation
        seg_result, seg_prof = profiler.profile_segmentation(pipeline, content_bytes, style_bytes)
        
        # Profile individual components
        components = profiler.profile_individual_components(pipeline)
        
        # Benchmark batch sizes
        batch_results = profiler.benchmark_batch_sizes(pipeline)
        
        # Analyze efficiency
        efficiency_results = profiler.analyze_model_efficiency(pipeline)
        
        # Generate report
        profiler.generate_report()
        
        print("\nBatch Processing Results:")
        print("-" * 40)
        for batch_size, results in batch_results.items():
            print(f"Batch size {batch_size}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.3f}")
        
        print("\nEfficiency Analysis:")
        print("-" * 40)
        for metric, value in efficiency_results.items():
            print(f"{metric}: {value:.3f}")
        
    except Exception as e:
        print(f"Profiling error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()