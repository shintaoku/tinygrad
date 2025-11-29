from tinygrad import Tensor, Device
import time

print(f"Using Device: {Device.DEFAULT}")

# ウォームアップ
print("Warming up...")
x = Tensor.rand(1024, 1024).realize()
y = Tensor.rand(1024, 1024).realize()
z = x.matmul(y).realize()

# 計測
print("Running benchmark...")
start = time.perf_counter()
for _ in range(10):
    z = x.matmul(y).realize()
end = time.perf_counter()

print(f"Average time per matmul (1024x1024): {(end - start)/10 * 1000:.2f} ms")
print(f"Result shape: {z.shape}")


