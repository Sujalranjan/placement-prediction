import os
import numpy as np
import pandas as pd

# 1. Target the dataset path
data_path = None
for path in [
    'data/student_placement_data_v2.csv',
    'data/student_placement_data.csv',
    'student_placement_data_v2.csv',
    'student_placement_data.csv',
]:
    if os.path.exists(path):
        data_path = path
        break

if not data_path:
    raise FileNotFoundError('Could not locate dataset.')

df = pd.read_csv(data_path)

# Set seed for reproducibility while generating broad variance
np.random.seed(42)
n = len(df)

# 2. Generate a wide-spread random distribution covering 5.0 to 10.0
# Blend uniform randomness across the entire spectrum with a broad Gaussian component
raw_uniform = np.random.uniform(5.00, 10.00, size=n)
broad_normal = np.random.normal(loc=7.45, scale=1.45, size=n)

# Combine for organic spread across all bands (5s, 6s, 7s, 8s, 9s, 10)
randomized_cgpa = 0.60 * raw_uniform + 0.40 * broad_normal

# Add random perturbation noise
randomized_cgpa += np.random.uniform(-0.4, 0.4, size=n)

# 3. Clip strictly to the 5.00 to 10.00 boundaries and round to 2 decimals
df['cgpa'] = np.clip(np.round(randomized_cgpa, 2), 5.00, 10.00)

# 4. Synchronize 10th and 12th board marks so they span naturally with the new CGPA range
if 'tenth_pct' in df.columns or '10th_percentage' in df.columns:
    col_10 = 'tenth_pct' if 'tenth_pct' in df.columns else '10th_percentage'
    df[col_10] = np.clip(
        np.round(df['cgpa'] * 9.2 + np.random.normal(2, 5.0, size=n), 1),
        45.0,
        99.5,
    )

if 'twelfth_pct' in df.columns or '12th_percentage' in df.columns:
    col_12 = 'twelfth_pct' if 'twelfth_pct' in df.columns else '12th_percentage'
    df[col_12] = np.clip(
        np.round(df['cgpa'] * 8.9 + np.random.normal(1, 5.5, size=n), 1),
        42.0,
        98.5,
    )

# 5. Overwrite the CSV
df.to_csv(data_path, index=False)

print(f"✓ Fixed {data_path}!")
print(f"  Min CGPA:  {df['cgpa'].min():.2f}")
print(f"  Max CGPA:  {df['cgpa'].max():.2f}")
print(f"  Mean CGPA: {df['cgpa'].mean():.2f}")
print(f"  Std Dev:   {df['cgpa'].std():.2f}")