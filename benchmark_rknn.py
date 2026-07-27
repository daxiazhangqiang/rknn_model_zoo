#!/usr/bin/env python3
"""
Simple RKNN benchmark script.
Usage examples:
  python3 benchmark_rknn.py --model path/to/model.rknn --random-shape 1,3,640,640 --iters 100
  python3 benchmark_rknn.py --model path/to/model.rknn --input-npy sample_input.npy --iters 50

The script uses py_utils.rknn_executor.RKNN_model_container to load and run the model.
"""
import argparse
import numpy as np
import time
import sys
from pathlib import Path

# Try to import the convenience wrapper from the repo
try:
    from py_utils.rknn_executor import RKNN_model_container
except Exception as e:
    print('Failed to import py_utils.rknn_executor:', e)
    print('Make sure you run this from the repository root so py_utils is on PYTHONPATH.')
    sys.exit(1)


def parse_shape(s: str):
    parts = [int(x) for x in s.split(',') if x.strip()]
    if len(parts) == 2:
        h, w = parts
        return (1, 3, h, w)
    if len(parts) == 3:
        c, h, w = parts
        return (1, c, h, w)
    if len(parts) == 4:
        return tuple(parts)
    raise ValueError('Invalid random-shape format; expected H,W or C,H,W or N,C,H,W')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model', dest='model', help='Path to .rknn model')
    p.add_argument('model_positional', nargs='?', help='Path to .rknn model (positional)')
    p.add_argument('--iters', '-n', type=int, default=100, help='Number of timed iterations')
    p.add_argument('--warmup', type=int, default=10, help='Warm-up iterations to skip in timing')
    p.add_argument('--input-npy', help='Path to a .npy file containing input array (single input)')
    p.add_argument('--random-shape', default='640,640', help='Create random input with this shape. Provide H,W (default 640,640) or C,H,W or N,C,H,W. If H,W given, shape will be 1,3,H,W.')
    p.add_argument('--target', default='rk3588', help='RKNN target (default: rk3588). Use RKNN constants like RKNN.TARGET_NPU if needed')
    p.add_argument('--device-id', type=int, default=None, help='Device id when initializing runtime')
    p.add_argument('--perf-debug', action='store_true', help='Enable RKNN perf_debug when init_runtime')
    p.add_argument('--eval-mem', action='store_true', help='Enable RKNN memory eval when init_runtime')

    args = p.parse_args()

    model_arg = args.model if args.model else args.model_positional
    if not model_arg:
        p.error('the following arguments are required: --model/-m or model_positional')

    model_path = Path(model_arg)
    if not model_path.exists():
        print('Model not found:', model_path)
        sys.exit(2)

    # Prepare input
    if args.input_npy:
        arr = np.load(args.input_npy)
        inputs = [arr]
    elif args.random_shape:
        shape = parse_shape(args.random_shape)
        # default float32 random data
        arr = np.random.rand(*shape).astype(np.float32)
        inputs = [arr]
    else:
        print('Either --input-npy or --random-shape must be provided')
        sys.exit(2)

    # Load model
    print('Loading model:', model_path)
    rknn = RKNN_model_container(str(model_path), 
                                target=args.target, 
                                device_id=args.device_id, 
                                perf_debug=args.perf_debug,
                                eval_mem=args.eval_mem)

    # Warm-up
    print(f'Warm-up {args.warmup} runs...')
    for i in range(args.warmup):
        _ = rknn.run(inputs)

    # Eval Perf
    if args.perf_debug:
        print('--> Eval Perf')
        rknn.rknn.eval_perf()
    # Eval memory
    if args.eval_mem:
        print('--> Eval memory')
        rknn.rknn.eval_memory()
      
    # Timed runs
    print(f'Running {args.iters} timed iterations...')
    start = time.perf_counter()
    for i in range(args.iters):
        _ = rknn.run(inputs)
    end = time.perf_counter()

    # manual check time cost
    total = end - start
    avg = total / args.iters
    print(f'Total time: {total:.6f} s for {args.iters} runs')
    print(f'Average per run: {avg*1000:.3f} ms')
    if inputs and hasattr(inputs[0], 'shape'):
        batch = inputs[0].shape[0] if len(inputs[0].shape) > 0 else 1
        throughput = batch * args.iters / total if total>0 else 0.0
        print(f'Throughput: {throughput:.2f} samples/s (batch={batch})')


    rknn.release()


if __name__ == '__main__':
    main()
