import numpy as np

# Create a sample array with shape (10000, 2)
array_10000_2 = np.random.rand(10000, 2)
print(f"Original shape: {array_10000_2.shape}")

# Method 1: Reshape to (100, 100)
# This flattens the array first, then reshapes to (100, 100)
array_100_100 = array_10000_2.reshape(100, 100)
print(f"Reshaped array shape: {array_100_100.shape}")

# Method 2: Using -1 for automatic dimension calculation
array_100_100_alt = array_10000_2.reshape(100, -1)
print(f"Alternative reshape: {array_100_100_alt.shape}")

# Verify the total number of elements is preserved
print(f"\nOriginal total elements: {array_10000_2.size}")
print(f"Reshaped total elements: {array_100_100.size}")
