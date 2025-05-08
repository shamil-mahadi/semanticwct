#!/usr/bin/env python
import ijson
import json
import argparse
from collections import defaultdict
from operator import itemgetter

def analyze_trace(file_path: str, limit: int = 25):
    """
    Analyzes a large Chrome trace JSON file using streaming to keep memory usage low.

    Args:
        file_path: Path to the JSON trace file.
        limit: The number of top results to return, sorted by total duration.

    Returns:
        A list of dictionaries summarizing the top N most time-consuming operations.
    """
    print(f"Starting analysis of {file_path}. This may take a few minutes...")

    # Use a defaultdict to easily aggregate stats for each operation name
    op_stats = defaultdict(lambda: {
        'count': 0,
        'total_duration_ms': 0,
        'max_duration_ms': 0
    })

    # The PyTorch profiler often nests the events list under a 'traceEvents' key.
    # We will stream the items from this list.
    try:
        with open(file_path, 'r') as f:
            events = ijson.items(f, 'traceEvents.item')
            
            # Iterate over each event in the trace file
            for event in events:
                # We are interested in "complete" events ('X'), which have a duration.
                if event.get('ph') == 'X' and 'dur' in event and 'name' in event:
                    name = event['name']
                    # Duration in trace is in microseconds (us). Convert to milliseconds (ms).
                    duration_ms = float(event['dur']) / 1000.0

                    # Aggregate the statistics
                    stats = op_stats[name]
                    stats['count'] += 1
                    stats['total_duration_ms'] += duration_ms
                    if duration_ms > stats['max_duration_ms']:
                        stats['max_duration_ms'] = duration_ms

    except ijson.JSONError as e:
        print(f"Error parsing JSON: {e}")
        print("Please ensure the file is a valid JSON trace.")
        return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    print("Analysis complete. Aggregating results...")

    # Convert the defaultdict to a list of dictionaries for sorting
    summary_list = []
    for name, stats in op_stats.items():
        summary_list.append({
            'operation_name': name,
            'count': stats['count'],
            'total_duration_ms': round(stats['total_duration_ms'], 2),
            'avg_duration_ms': round(stats['total_duration_ms'] / stats['count'], 4),
            'max_duration_ms': round(stats['max_duration_ms'], 2)
        })

    # Sort the list by total duration in descending order
    sorted_summary = sorted(summary_list, key=itemgetter('total_duration_ms'), reverse=True)

    # Return the top N results
    return sorted_summary[:limit]

def main():
    """Main function to run the trace analyzer from the command line."""
    parser = argparse.ArgumentParser(
        description="Analyze a PyTorch profiler Chrome trace file and summarize performance."
    )
    parser.add_argument(
        "trace_file",
        help="Path to the segmentation_trace.json file."
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=25,
        help="Number of top operations to display in the summary."
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional: Path to save the summary JSON file."
    )

    args = parser.parse_args()

    summary = analyze_trace(args.trace_file, args.limit)

    if summary:
        # Pretty-print the JSON summary to the console
        summary_json = json.dumps(summary, indent=2)
        print("\n--- PROFILER SUMMARY ---")
        print(summary_json)
        print("--- END OF SUMMARY ---\n")

        if args.output:
            with open(args.output, 'w') as f_out:
                f_out.write(summary_json)
            print(f"Summary saved to {args.output}")

if __name__ == "__main__":
    main()
