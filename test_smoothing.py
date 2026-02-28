import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from config import SLOPE_MAX_PCT

# Simulate a route. 
# 1st half is "flat" -> points have large noise, but mean slope is low.
# 2nd half is "steep" -> points have large noise, but mean slope is high.
np.random.seed(42)

# Create raw jagged path
N = 200
x = np.linspace(0, 100, N)
y = np.sin(x * 0.1) * 20 + np.random.randn(N) * 5

# Local slopes: first half is flat (0.02), second half is steep (0.15)
local_slopes = np.zeros(N)
local_slopes[:100] = 0.02
local_slopes[100:] = 0.15

# Map to smoothing multiplier
flat_thresh  = SLOPE_MAX_PCT * 0.4
steep_thresh = SLOPE_MAX_PCT * 0.8
t = np.clip((local_slopes - flat_thresh) / max(steep_thresh - flat_thresh, 0.01), 0, 1)
smooth_mult = 8.0 - t * 7.0

# Old global method
adaptive_s_old = N * 10.0 * float(np.median(smooth_mult))
tck_old, _ = splprep([x, y], s=adaptive_s_old, k=3)
x_old, y_old = splev(np.linspace(0, 1, 500), tck_old)

# New weighted method
weights = 1.0 / np.sqrt(smooth_mult)
base_s = N * 10.0
tck_new, _ = splprep([x, y], w=weights, s=base_s, k=3)
x_new, y_new = splev(np.linspace(0, 1, 500), tck_new)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x, y, '.', color='gray', label='Raw Noisy Path', alpha=0.5)

# Color the points by their weight to see where it grips
scatter = plt.scatter(x, y, c=weights, cmap='viridis', s=20, edgecolor='k', zorder=5)
plt.colorbar(scatter, label='Spline Point Weight (higher = tighter grip)')

plt.plot(x_old, y_old, 'r--', label=f'Old Global Median Smoothing (s={adaptive_s_old:.1f})', linewidth=2)
plt.plot(x_new, y_new, 'b-', label=f'New Weighted Smoothing (base_s={base_s:.1f})', linewidth=3)

plt.axvline(x[100], color='k', linestyle=':', label='Flat/Steep Boundary')

plt.title('Comparison of Global vs. Segment-Aware Weighted B-Spline Smoothing')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('test_smoothing_plot.png')
print("Saved plot to test_smoothing_plot.png")
