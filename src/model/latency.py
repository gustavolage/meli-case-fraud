#!/usr/bin/env python
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click
import json

def measure_latency(pipeline, payload, iterations=10):
    """
    Measure the average latency of each step in the pipeline using the given payload.
    Returns a dictionary with the average time (in milliseconds) for each step.
    """
    step_times = {}
    current_input = payload.copy()
    
    # Iterate over each step, except the last (estimator)
    for name, transformer in pipeline.steps[:-1]:
        times = []
        for _ in range(iterations):
            t0 = time.time()
            current_input = transformer.transform(current_input)
            t1 = time.time()
            times.append((t1 - t0) * 1000)  # convert to ms
        step_times[name] = np.mean(times)
    
    # Measure time for the final step (model prediction)
    model_name, model = pipeline.steps[-1]
    times = []
    for _ in range(iterations):
        t0 = time.time()
        model.predict(current_input, n_jobs=1)
        t1 = time.time()
        times.append((t1 - t0) * 1000)
    step_times[model_name] = np.mean(times)
    
    return step_times

def simulate_total_latency(num_requests, num_workers, per_request_time):
    """
    Simulate the expected total latency considering parallelization:
    expected total latency = t * num_requests / num_workers.
    per_request_time should be in milliseconds.
    """
    return per_request_time * num_requests / num_workers

@click.command()
@click.option('--num_requests', default=1000, type=int, help='Number of requests')
@click.option('--num_workers', default=4, type=int, help='Number of workers')
@click.option('--output_path', default='reports/figures', type=str, help='Path to save the generated plots')
def main(num_requests, num_workers, output_path):
    
    # Load payload 
    mass = json.load(open(os.path.join("data", "mass", "payload.json"), "r"))
    payload = mass['data']
    
    # Load model pipeline for production
    model_pipeline = pickle.load(open(f"models/wrapped/model_pipeline_prod.pkl", "rb"))  
    
    # Measure latency
    step_times = measure_latency(model_pipeline, payload, iterations=10)
    total_time_per_request = sum(step_times.values())
    expected_total_latency = simulate_total_latency(num_requests, num_workers, total_time_per_request)

    click.echo("Average latency per step (ms):")
    for step, t in step_times.items():
        click.echo(f"  {step}: {t:.4f} ms")
    click.echo(f"\nAverage total time per request: {total_time_per_request:.4f} ms")
    click.echo(f"Expected total latency for {num_requests} requests with {num_workers} workers: {expected_total_latency:.4f} ms")
    
    # Visualization 1: Average latency per pipeline step
    steps = list(step_times.keys())
    times = [step_times[s] for s in steps]
    plt.figure(figsize=(10, 6))
    plt.bar(steps, times, color='skyblue')
    plt.ylabel("Average time (ms)")
    plt.title("Average Latency per Pipeline Step")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "latency_steps.png"))
    plt.close()
    
    # Visualization 2: Real latency moving average over time
    # Measure the real per-request latency by calling pipeline.predict for each request
    real_latencies = []
      
    for _ in range(num_requests):
        payload_aux = payload.copy()
        start = time.time()
        model_pipeline.predict(payload_aux, n_jobs=1)
        real_latencies.append((time.time() - start) * 1000)
    
    # Create a timeline by computing the cumulative sum of latencies
    timeline = np.cumsum(real_latencies)
    # Determine window size for moving average
    window_size = 25 if num_requests >= 25 else 1
    moving_avg = pd.Series(real_latencies).rolling(window=window_size, min_periods=1).mean().to_numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(timeline, moving_avg, linestyle='-')
    plt.xlabel("Timeline (ms)")
    plt.ylabel("Latency Time Moving Average (ms)")
    plt.title("Latency Moving Average over Time")
    plt.tight_layout()
    plt.ylim(0,  2 * max(moving_avg))
    plt.savefig(os.path.join(output_path, "latency.png"))
    plt.close()

if __name__ == '__main__':
    main()