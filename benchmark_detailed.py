from tinygrad import Tensor, Device, GlobalCounters, TinyJit
from tinygrad.helpers import getenv
import time
import numpy as np

# プロファイリング対象のダミーモデル（Transformer Block相当）
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
        # Attention parts (simplified)
        h = x.layernorm().linear(self.w_q) # Norm + Linear
        h = h.linear(self.w_o)
        x = x + h
        
        # FFN parts
        h = x.layernorm()
        h = h.linear(self.w_ff1).gelu().linear(self.w_ff2)
        return x + h

def benchmark():
    # Llama-2 7B size approximation
    DIM = 4096
    HIDDEN_DIM = 11008 
    SEQ_LEN = 128
    BS = 1
    
    print(f"Device: {Device.DEFAULT}")
    print(f"Config: Batch={BS}, Seq={SEQ_LEN}, Dim={DIM}, Hidden={HIDDEN_DIM}")

    # 1. モデル構築
    model = TransformerBlock(DIM, HIDDEN_DIM)
    
    # 入力バッファ（実体）を用意
    x = Tensor.randn(BS, SEQ_LEN, DIM).realize()
    
    # JIT化された関数
    @TinyJit
    def run_step(x):
        return model(x).realize()

    # 初回実行 (JIT Capture)
    print("Capturing JIT...")
    run_step(x)
    
    # ベンチマーク
    print("Running 10 loops with data update...")
    measurements = []
    
    # ランダムデータをあらかじめ用意
    input_data = [np.random.randn(BS, SEQ_LEN, DIM).astype(np.float32) for _ in range(10)]
    
    for i in range(10):
        # データ転送 (Host -> Device)
        if x.uop.realized is None:
            x.realize()
        x.uop.realized.copyin(input_data[i].data)
        
        # 計測開始
        st = time.perf_counter()
        
        out = run_step(x)
        
        # 同期 (Device -> Host転送は含まないように、コマンド完了だけ待つ)
        Device[Device.DEFAULT].synchronize()
        
        et = time.perf_counter()
        measurements.append((et-st)*1000)
    
    avg_time = np.mean(measurements)
    print(f"Execution Time (with Sync): {avg_time:.2f} ms")
    print(f"  - Min: {np.min(measurements):.2f} ms")
    print(f"  - Max: {np.max(measurements):.2f} ms")

if __name__ == "__main__":
    benchmark()
