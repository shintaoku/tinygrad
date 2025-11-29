import matplotlib.pyplot as plt
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
import time
import numpy as np
import shutil
import importlib
import sys
import os

# Llama-2 70B級のパラメータ
DIM = 5120
HIDDEN_DIM = 13824
SEQ_LEN = 128
BS = 1

class TransformerBlock:
    def __init__(self, dim, hidden_dim):
        self.w_q = Tensor.scaled_uniform(dim, dim)
        self.w_k = Tensor.scaled_uniform(dim, dim)
        self.w_v = Tensor.scaled_uniform(dim, dim)
        self.w_o = Tensor.scaled_uniform(dim, dim)
        self.w_ff1 = Tensor.scaled_uniform(dim, hidden_dim)
        self.w_ff2 = Tensor.scaled_uniform(hidden_dim, dim)
        self.norm1 = Tensor.ones(dim)
        self.norm2 = Tensor.ones(dim)

    def __call__(self, x):
        h = x.layernorm().linear(self.w_q) # Norm + Linear
        # Attention mechanism skipped for pure compute benchmark
        h = h.linear(self.w_o)
        x = x + h
        
        # FFN parts
        h = x.layernorm()
        h = h.linear(self.w_ff1).gelu().linear(self.w_ff2)
        return x + h

def run_benchmark_kernel():
    # 入力バッファ
    x = Tensor.randn(BS, SEQ_LEN, DIM).realize()
    model = TransformerBlock(DIM, HIDDEN_DIM)
    
    @TinyJit
    def run_step(x):
        return model(x).realize()

    # JIT Capture
    run_step(x)
    
    measurements = []
    # 20回実行
    for _ in range(20):
        # Dispatch時間のみ計測
        st = time.perf_counter()
        out = run_step(x)
        # Device.synchronize() を入れて、GPU完了まで待つ
        Device[Device.DEFAULT].synchronize()
        et = time.perf_counter()
        measurements.append((et-st)*1000)
    
    return measurements

if __name__ == "__main__":
    # 引数処理
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "slow", "plot"], default="plot")
    args = parser.parse_args()

    if args.mode == "plot":
        # サブプロセスで計測を実行
        import subprocess
        
        # Fast
        print(">>> Measuring FAST (New Metal Graph)...")
        # fast版は現在の metal.py なのでそのまま実行
        res_fast = subprocess.check_output([sys.executable, __file__, "--mode=fast"]).decode()
        data_fast = [float(x) for x in res_fast.strip().split()]
        
        # Slow
        print(">>> Measuring SLOW (Old Metal Graph)...")
        # ファイルを入れ替え
        shutil.copy("tinygrad/runtime/graph/metal.py", "tinygrad/runtime/graph/metal_fast_tmp.py")
        shutil.copy("tinygrad/runtime/graph/metal_old.py", "tinygrad/runtime/graph/metal.py")
        
        try:
            res_slow = subprocess.check_output([sys.executable, __file__, "--mode=slow"]).decode()
            data_slow = [float(x) for x in res_slow.strip().split()]
        finally:
            # 戻す
            shutil.copy("tinygrad/runtime/graph/metal_fast_tmp.py", "tinygrad/runtime/graph/metal.py")
            os.remove("tinygrad/runtime/graph/metal_fast_tmp.py")

        # グラフ作成
        avg_fast = np.mean(data_fast)
        avg_slow = np.mean(data_slow)
        speedup = avg_slow / avg_fast
        
        print(f"Fast Avg: {avg_fast:.2f} ms")
        print(f"Slow Avg: {avg_slow:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        plt.figure(figsize=(10, 6))
        plt.plot(data_slow, label=f'Slow (Original): Avg {avg_slow:.2f}ms')
        plt.plot(data_fast, label=f'Fast (Optimized): Avg {avg_fast:.2f}ms')
        plt.title(f"Metal Graph Dispatch Performance (Llama-2 13B Block)\nSpeedup: {speedup:.2f}x")
        plt.ylabel("Time per Step (ms)")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)
        plt.savefig("benchmark_result.png")
        print("Graph saved to benchmark_result.png")

    else:
        # 計測実行モード
        res = run_benchmark_kernel()
        # 結果を標準出力へ（親プロセスが拾う）
        print(" ".join(map(str, res)))
