import numpy as np

# Load the indices
mw = np.load("mann_whitney_indices.npy")

print(f"Total features selected: {len(mw)}")
print(f"First 20 indices: {mw[:20]}")
print(f"Min index: {np.min(mw)}")
print(f"Max index: {np.max(mw)}")

# Check if they are sorted (they SHOULD be, if generated correctly alongside the model)
is_sorted = np.all(mw[:-1] <= mw[1:])
print(f"Are the indices sorted in order? {is_sorted}")