import matplotlib.pyplot as plt
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
import time
import numpy as np
import shutil
import importlib
import sys
import os

# モデル設定
MODEL_CONFIGS = {
    "small": {"DIM": 512, "HIDDEN_DIM": 2048, "SEQ_LEN": 64, "BS": 1, "DESC": "軽量モデル (コマンド発行頻度高)"},
    "large": {"DIM": 5120, "HIDDEN_DIM": 13824, "SEQ_LEN": 128, "BS": 1, "DESC": "大規模モデル (Llama-2 13B級, 計算負荷高)"}
}

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
        h = x.layernorm().linear(self.w_q)
        h = h.linear(self.w_o)
        x = x + h
        h = x.layernorm()
        h = h.linear(self.w_ff1).gelu().linear(self.w_ff2)
        return x + h

def run_benchmark_kernel(config_name):
    cfg = MODEL_CONFIGS[config_name]
    
    # 入力バッファ
    x = Tensor.randn(cfg["BS"], cfg["SEQ_LEN"], cfg["DIM"]).realize()
    model = TransformerBlock(cfg["DIM"], cfg["HIDDEN_DIM"])
    
    @TinyJit
    def run_step(x):
        return model(x).realize()

    # JIT Capture
    run_step(x)
    
    measurements = []
    # 20回実行
    print(f"Running 20 loops ({config_name})...")
    
    for i in range(20):
        # プロファイリングイベントをクリア
        if hasattr(Device[Device.DEFAULT], "profile_events"):
            Device[Device.DEFAULT].profile_events = []
        
        st = time.perf_counter()
        out = run_step(x)
        Device[Device.DEFAULT].synchronize()
        et = time.perf_counter()
        cpu_time = (et-st)*1000
        measurements.append(cpu_time)
        
        # GPU時間の取得
        gpu_time = 0.0
        if hasattr(Device[Device.DEFAULT], "profile_events"):
            events = Device[Device.DEFAULT].profile_events
            if events:
                # ProfileGraphEventの場合、e.st, e.en がある (usec単位)
                # ただし、イベントが複数ある場合は合計する
                # (GraphRunnerの場合、通常は1つのProfileGraphEventになるはず)
                total_duration_us = 0
                for e in events:
                    if hasattr(e, 'en') and hasattr(e, 'st'):
                        total_duration_us += (e.en - e.st)
                gpu_time = total_duration_us / 1000.0 # us -> ms
            
        print(f"Step {i}: CPU={cpu_time:.4f} ms, GPU={gpu_time:.4f} ms")
    
    return measurements

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "slow", "plot"], default="plot")
    parser.add_argument("--size", choices=["small", "large"], default="small")
    args = parser.parse_args()

    if args.mode == "plot":
        import subprocess
        
        cfg = MODEL_CONFIGS[args.size]
        print(f"\n===== テスト実行: {cfg['DESC']} =====")
        print("このテストでは、Pythonランタイムのオーバーヘッドが全体に与える影響を測定します。")
        
        # Fast
        print("\n>>> 🚀 高速版 (Nano-Mirage / UOp Dynamic Compile) を測定中...")
        res_fast = subprocess.check_output([sys.executable, __file__, "--mode=fast", f"--size={args.size}"]).decode()
        data_fast = []
        for line in res_fast.splitlines():
            if line.startswith("Step"):
                # GPU時間などのログは無視して、最後の数値列だけ探す必要があるが、
                # ここでは簡易的に、run_benchmark_kernelがprintするログは無視し、
                # 最後に空白区切りの数値が出力されることを期待する。
                print(f"  {line}")
            else:
                try:
                    data_fast = [float(x) for x in line.strip().split()]
                except ValueError:
                    pass
        
        # Slow
        print("\n>>> 🐢 通常版 (Original Tinygrad / Python Loop) を測定中...")
        shutil.copy("tinygrad/runtime/graph/metal.py", "tinygrad/runtime/graph/metal_fast_tmp.py")
        shutil.copy("tinygrad/runtime/graph/metal_old.py", "tinygrad/runtime/graph/metal.py")
        
        try:
            res_slow = subprocess.check_output([sys.executable, __file__, "--mode=slow", f"--size={args.size}"]).decode()
            data_slow = []
            for line in res_slow.splitlines():
                if line.startswith("Step"):
                    print(f"  {line}")
                else:
                    try:
                        data_slow = [float(x) for x in line.strip().split()]
                    except ValueError:
                        pass
        finally:
            shutil.copy("tinygrad/runtime/graph/metal_fast_tmp.py", "tinygrad/runtime/graph/metal.py")
            os.remove("tinygrad/runtime/graph/metal_fast_tmp.py")

        # グラフ作成
        if not data_fast or not data_slow:
            print("Error: Could not parse benchmark data.")
            sys.exit(1)

        avg_fast = np.mean(data_fast)
        avg_slow = np.mean(data_slow)
        speedup = avg_slow / avg_fast
        
        print(f"\n📊 結果 ({args.size}):")
        print(f"  高速版 (Nano-Mirage): 平均 {avg_fast:.2f} ms")
        print(f"  通常版 (Original):    平均 {avg_slow:.2f} ms")
        print(f"  ⚡️ 速度向上率:       {speedup:.2f}倍")
        
        plt.figure(figsize=(10, 6))
        plt.plot(data_slow, label=f'Slow (Original): Avg {avg_slow:.2f}ms')
        plt.plot(data_fast, label=f'Fast (Nano-Mirage): Avg {avg_fast:.2f}ms')
        plt.title(f"Metal Graph Dispatch Performance ({args.size})\nSpeedup: {speedup:.2f}x")
        plt.ylabel("Time per Step (ms)")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)
        filename = f"benchmark_result_{args.size}.png"
        plt.savefig(filename)
        print(f"  📈 グラフを保存しました: tinygrad/{filename}")

    else:
        # 計測実行モード
        res = run_benchmark_kernel(args.size)
        print(" ".join(map(str, res)))
