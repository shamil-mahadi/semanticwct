import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import statistics
from dataclasses import dataclass
import random

# Assume DEVICE is defined globally
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resize_mask_fast(mask: np.ndarray, target_size: torch.Size) -> torch.Tensor:
    """
    A faster implementation of nearest-neighbor resizing that matches
    torch.nn.functional.interpolate(..., mode='nearest', align_corners=False).
    """
    h, w = mask.shape
    th, tw = target_size

    # Convert mask to a tensor on the correct device
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=DEVICE)

    # Calculate source coordinates for each target pixel
    # This mimics the behavior of interpolate with align_corners=False
    y_source_indices = (torch.arange(th, device=DEVICE) * (h / th)).long()
    x_source_indices = (torch.arange(tw, device=DEVICE) * (w / tw)).long()

    # Use advanced indexing with broadcasting to create the resized mask
    # y_source_indices.unsqueeze(1) has shape [th, 1]
    # x_source_indices has shape [tw]
    # Broadcasting creates a [th, tw] grid of indices
    resized_mask = mask_tensor[y_source_indices.unsqueeze(1), x_source_indices]

    return resized_mask.flatten()

def _resize_mask(mask: np.ndarray, target_size: torch.Size) -> torch.Tensor:
    """Standard resize using interpolation"""
    # Convert to float for interpolation, then back to bool
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=DEVICE)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.nn.functional.interpolate(
        mask_tensor, target_size, mode="nearest"
    )
    mask_tensor = mask_tensor.squeeze().flatten()
    return mask_tensor.bool()  # Convert back to boolean

@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run"""
    function_name: str
    input_shape: Tuple[int, int]
    target_shape: Tuple[int, int]
    execution_time: float
    memory_allocated: float
    memory_cached: float

@dataclass
class VGG19Dimensions:
    """Real VGG19 layer dimensions for different input sizes"""
    input_name: str
    conv_layers: List[Tuple[int, int]]  # (height, width) for each conv layer
    common_target_sizes: List[Tuple[int, int]]  # Common resize targets

def get_vgg19_dimensions() -> List[VGG19Dimensions]:
    """Get realistic VGG19 feature map dimensions for different input sizes"""
    return [
        VGG19Dimensions(
            input_name="224x224_input",
            conv_layers=[
                (224, 224),  # input
                (224, 224),  # conv1_1, conv1_2
                (112, 112),  # pool1
                (112, 112),  # conv2_1, conv2_2
                (56, 56),    # pool2
                (56, 56),    # conv3_1, conv3_2, conv3_3, conv3_4
                (28, 28),    # pool3
                (28, 28),    # conv4_1, conv4_2, conv4_3, conv4_4
                (14, 14),    # pool4
                (14, 14),    # conv5_1, conv5_2, conv5_3, conv5_4
                (7, 7),      # pool5
            ],
            common_target_sizes=[
                (224, 224), (112, 112), (56, 56), (28, 28), (14, 14), (7, 7),
                (32, 32), (64, 64), (128, 128), (256, 256)
            ]
        ),
        VGG19Dimensions(
            input_name="512x512_input",
            conv_layers=[
                (512, 512),  # input
                (512, 512),  # conv1_1, conv1_2
                (256, 256),  # pool1
                (256, 256),  # conv2_1, conv2_2
                (128, 128),  # pool2
                (128, 128),  # conv3_1, conv3_2, conv3_3, conv3_4
                (64, 64),    # pool3
                (64, 64),    # conv4_1, conv4_2, conv4_3, conv4_4
                (32, 32),    # pool4
                (32, 32),    # conv5_1, conv5_2, conv5_3, conv5_4
                (16, 16),    # pool5
            ],
            common_target_sizes=[
                (512, 512), (256, 256), (128, 128), (64, 64), (32, 32), (16, 16),
                (224, 224), (112, 112), (56, 56), (28, 28), (14, 14)
            ]
        ),
        VGG19Dimensions(
            input_name="1024x1024_input",
            conv_layers=[
                (1024, 1024),  # input
                (1024, 1024),  # conv1_1, conv1_2
                (512, 512),    # pool1
                (512, 512),    # conv2_1, conv2_2
                (256, 256),    # pool2
                (256, 256),    # conv3_1, conv3_2, conv3_3, conv3_4
                (128, 128),    # pool3
                (128, 128),    # conv4_1, conv4_2, conv4_3, conv4_4
                (64, 64),      # pool4
                (64, 64),      # conv5_1, conv5_2, conv5_3, conv5_4
                (32, 32),      # pool5
            ],
            common_target_sizes=[
                (1024, 1024), (512, 512), (256, 256), (128, 128), (64, 64), (32, 32),
                (224, 224), (112, 112), (56, 56), (28, 28)
            ]
        )
    ]

def generate_realistic_mask(height: int, width: int, complexity: str = "medium") -> np.ndarray:
    """Generate realistic boolean masks that might come from feature maps"""
    # Use a different seed for each size to get variety
    np.random.seed(42 + hash((height, width)) % 1000)
    
    min_dim = min(height, width)
    max_dim = max(height, width)
    
    if complexity == "simple":
        # Simple geometric shapes
        mask = np.zeros((height, width), dtype=bool)
        center_h, center_w = height // 2, width // 2
        radius = max(1, min_dim // 4)
        y, x = np.ogrid[:height, :width]
        circle_mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= radius ** 2
        mask[circle_mask] = True
        
    elif complexity == "medium":
        # Multiple objects with varying sizes
        mask = np.zeros((height, width), dtype=bool)
        num_objects = max(2, min(15, (height * width) // 8000))
        
        for _ in range(num_objects):
            # Random circular objects with safe bounds
            center_h = np.random.randint(max(1, height // 8), max(2, 7 * height // 8))
            center_w = np.random.randint(max(1, width // 8), max(2, 7 * width // 8))
            
            # Safe radius calculation
            min_radius = max(1, min_dim // 50)
            max_radius = max(2, min_dim // 6)
            if max_radius <= min_radius:
                max_radius = min_radius + 1
            
            radius = np.random.randint(min_radius, max_radius)
            
            y, x = np.ogrid[:height, :width]
            circle_mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= radius ** 2
            mask[circle_mask] = True
            
    elif complexity == "complex":
        # Complex pattern without scipy dependency
        mask = np.zeros((height, width), dtype=bool)
        
        # Add base random noise
        base_noise = np.random.rand(height, width) > 0.75
        mask = mask | base_noise
        
        # Add multiple overlapping circular regions
        num_circles = max(3, min(20, (height * width) // 3000))
        for _ in range(num_circles):
            center_h = np.random.randint(0, height)
            center_w = np.random.randint(0, width)
            
            # Safe radius calculation
            min_radius = max(1, min_dim // 100)
            max_radius = max(2, min_dim // 8)
            if max_radius <= min_radius:
                max_radius = min_radius + 1
                
            radius = np.random.randint(min_radius, max_radius)
            
            y, x = np.ogrid[:height, :width]
            circle_mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= radius ** 2
            
            if np.random.rand() > 0.5:
                mask[circle_mask] = True
            else:
                mask[circle_mask] = False
        
    return mask.astype(bool)

def benchmark_function(func, mask: np.ndarray, target_size: torch.Size, 
                      num_runs: int = 10, warmup_runs: int = 3) -> BenchmarkResult:
    """Benchmark a single function with memory tracking"""
    
    # Warmup runs
    for _ in range(warmup_runs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _ = func(mask, target_size)
    
    # Clear cache before measurement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Measure execution time
    times = []
    memory_allocated = []
    memory_cached = []
    
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        result = func(mask, target_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        if torch.cuda.is_available():
            memory_allocated.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            memory_cached.append(torch.cuda.max_memory_cached() / 1024**2)  # MB
        else:
            memory_allocated.append(0)
            memory_cached.append(0)
    
    return BenchmarkResult(
        function_name=func.__name__,
        input_shape=mask.shape,
        target_shape=tuple(target_size),
        execution_time=statistics.median(times),
        memory_allocated=statistics.median(memory_allocated),
        memory_cached=statistics.median(memory_cached)
    )

def compare_outputs(mask: np.ndarray, target_size: torch.Size, 
                   tolerance: float = 1e-6) -> Dict[str, any]:
    """Compare outputs of both functions"""
    
    result_fast = resize_mask_fast(mask, target_size)
    result_standard = _resize_mask(mask, target_size)
    
    # Convert to same device for comparison
    result_fast_cpu = result_fast.cpu().numpy()
    result_standard_cpu = result_standard.cpu().numpy()
    
    # Check shapes
    shapes_match = result_fast_cpu.shape == result_standard_cpu.shape
    
    # Check values (for boolean, should be exact match)
    values_match = np.array_equal(result_fast_cpu, result_standard_cpu)
    
    # Calculate difference ratio
    if not values_match:
        diff_count = np.sum(result_fast_cpu != result_standard_cpu)
        diff_ratio = diff_count / result_fast_cpu.size
    else:
        diff_ratio = 0.0
    
    return {
        "shapes_match": shapes_match,
        "values_match": values_match,
        "difference_ratio": diff_ratio,
        "fast_shape": result_fast_cpu.shape,
        "standard_shape": result_standard_cpu.shape,
        "fast_true_count": np.sum(result_fast_cpu),
        "standard_true_count": np.sum(result_standard_cpu)
    }

def run_comprehensive_benchmark():
    """Run comprehensive benchmarking across all VGG19 dimensions"""
    
    print(f"Running benchmarks on {DEVICE}")
    print("=" * 80)
    
    vgg_dimensions = get_vgg19_dimensions()
    all_results = []
    comparison_results = []
    
    total_tests = 0
    for vgg_dim in vgg_dimensions:
        total_tests += len(vgg_dim.conv_layers) * len(vgg_dim.common_target_sizes)
    
    current_test = 0
    
    for vgg_dim in vgg_dimensions:
        print(f"\nTesting {vgg_dim.input_name}")
        print("-" * 40)
        
        for conv_layer_size in vgg_dim.conv_layers:
            # Generate realistic masks of different complexities
            masks = {
                "simple": generate_realistic_mask(*conv_layer_size, "simple"),
                "medium": generate_realistic_mask(*conv_layer_size, "medium"),
                "complex": generate_realistic_mask(*conv_layer_size, "complex")
            }
            
            for target_size in vgg_dim.common_target_sizes:
                current_test += 1
                print(f"Progress: {current_test}/{total_tests} - "
                      f"Input: {conv_layer_size} -> Target: {target_size}")
                
                # Test with medium complexity mask (most representative)
                mask = masks["medium"]
                target_torch_size = torch.Size(target_size)
                
                # Compare outputs first
                comparison = compare_outputs(mask, target_torch_size)
                comparison["input_shape"] = conv_layer_size
                comparison["target_shape"] = target_size
                comparison["input_size_name"] = vgg_dim.input_name
                comparison_results.append(comparison)
                
                if not comparison["values_match"]:
                    print(f"  ⚠️  OUTPUT MISMATCH: {comparison['difference_ratio']:.4f} difference ratio")
                
                # Benchmark both functions
                try:
                    result_fast = benchmark_function(resize_mask_fast, mask, target_torch_size)
                    result_standard = benchmark_function(_resize_mask, mask, target_torch_size)
                    
                    result_fast.input_size_name = vgg_dim.input_name
                    result_standard.input_size_name = vgg_dim.input_name
                    
                    all_results.extend([result_fast, result_standard])
                    
                    speedup = result_standard.execution_time / result_fast.execution_time
                    memory_diff = result_standard.memory_allocated - result_fast.memory_allocated
                    
                    print(f"  Fast: {result_fast.execution_time*1000:.2f}ms, "
                          f"Standard: {result_standard.execution_time*1000:.2f}ms, "
                          f"Speedup: {speedup:.2f}x, "
                          f"Memory diff: {memory_diff:.1f}MB")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
    
    return all_results, comparison_results

def analyze_results(results: List[BenchmarkResult], comparisons: List[Dict]):
    """Analyze and visualize benchmark results"""
    
    print("\n" + "=" * 80)
    print("BENCHMARK ANALYSIS")
    print("=" * 80)
    
    # Output comparison summary
    print("\nOUTPUT COMPARISON SUMMARY:")
    print("-" * 40)
    
    total_comparisons = len(comparisons)
    matching_outputs = sum(1 for c in comparisons if c["values_match"])
    
    print(f"Total test cases: {total_comparisons}")
    print(f"Matching outputs: {matching_outputs}/{total_comparisons} ({matching_outputs/total_comparisons*100:.1f}%)")
    
    if matching_outputs < total_comparisons:
        print("\n❌ CRITICAL: Functions produce different outputs!")
        print("Mismatched cases:")
        for comp in comparisons:
            if not comp["values_match"]:
                print(f"  {comp['input_shape']} -> {comp['target_shape']}: "
                      f"{comp['difference_ratio']:.4f} difference ratio")
    else:
        print("✅ All outputs match perfectly!")
    
    # Performance analysis
    print("\nPERFORMANCE ANALYSIS:")
    print("-" * 40)
    
    fast_results = [r for r in results if r.function_name == "resize_mask_fast"]
    standard_results = [r for r in results if r.function_name == "_resize_mask"]
    
    if len(fast_results) != len(standard_results):
        print("⚠️  Unequal number of results for comparison")
        return
    
    # Calculate statistics
    speedups = []
    memory_diffs = []
    
    for fast, standard in zip(fast_results, standard_results):
        speedup = standard.execution_time / fast.execution_time
        memory_diff = standard.memory_allocated - fast.memory_allocated
        speedups.append(speedup)
        memory_diffs.append(memory_diff)
    
    print(f"Average speedup: {statistics.mean(speedups):.2f}x")
    print(f"Median speedup: {statistics.median(speedups):.2f}x")
    print(f"Min speedup: {min(speedups):.2f}x")
    print(f"Max speedup: {max(speedups):.2f}x")
    
    print(f"\nAverage memory difference: {statistics.mean(memory_diffs):.1f}MB")
    print(f"Median memory difference: {statistics.median(memory_diffs):.1f}MB")
    
    # Performance by input size
    print("\nPERFORMANCE BY INPUT SIZE:")
    print("-" * 40)
    
    size_groups = {}
    for fast, standard in zip(fast_results, standard_results):
        input_size = fast.input_shape[0] * fast.input_shape[1]
        if input_size not in size_groups:
            size_groups[input_size] = []
        
        speedup = standard.execution_time / fast.execution_time
        size_groups[input_size].append(speedup)
    
    for size in sorted(size_groups.keys()):
        avg_speedup = statistics.mean(size_groups[size])
        print(f"Input size ~{size:,} pixels: {avg_speedup:.2f}x average speedup")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    
    avg_speedup = statistics.mean(speedups)
    avg_memory_diff = statistics.mean(memory_diffs)
    
    if matching_outputs == total_comparisons:
        if avg_speedup > 1.5:
            print("✅ RECOMMENDED: Use resize_mask_fast")
            print(f"   - Significantly faster ({avg_speedup:.2f}x speedup)")
            print(f"   - Memory efficient ({avg_memory_diff:.1f}MB less)")
            print("   - Identical outputs to standard method")
        elif avg_speedup > 1.1:
            print("✅ RECOMMENDED: Use resize_mask_fast for performance-critical code")
            print(f"   - Moderately faster ({avg_speedup:.2f}x speedup)")
            print("   - Consider _resize_mask for better readability if performance isn't critical")
        else:
            print("⚖️  NEUTRAL: Both methods perform similarly")
            print("   - Choose _resize_mask for readability")
            print("   - Choose resize_mask_fast for slight performance edge")
    else:
        print("❌ NOT RECOMMENDED: Use resize_mask_fast")
        print("   - Functions produce different outputs")
        print("   - Verify correctness before using in production")

if __name__ == "__main__":
    print("VGG19 Mask Resize Benchmark")
    print("Testing with realistic feature map dimensions and masks")
    print(f"Device: {DEVICE}")
    
    # Run comprehensive benchmark
    results, comparisons = run_comprehensive_benchmark()
    
    # Analyze results
    analyze_results(results, comparisons)
    
    print("\n" + "=" * 80)
    print("Benchmark completed!")