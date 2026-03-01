import numpy as np

print("Inspecting combined pyramid npz...")
try:
    data = np.load('output/cost_pyramid_debug.npz')
    for arr_name in data.files:
        cost = data[arr_name]
        valid = cost[cost < 1e9]
        print(f"Array {arr_name}: shape={cost.shape}")
        print(f"  Max valid: {valid.max() if len(valid) > 0 else 0}")
        print(f"  Mean valid: {valid.mean() if len(valid) > 0 else 0}")
        print(f"  Impassable cells: {(cost >= 1e9).sum()}")
        print(f"  Pixels >1000 cost: {(cost > 1000).sum()} (including impassable)")
except Exception as e:
    print(f"Error loading npz: {e}")
