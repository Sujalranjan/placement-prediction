import os
import glob
import numpy as np
import pandas as pd

# 1. Search for student_placement_data_v2 in data/ or root directory
search_patterns = [
    'data/student_placement_data_v2.csv',
    'data/student_placement_data_v2.xlsx',
    'student_placement_data_v2.csv',
    'student_placement_data_v2.xlsx'
]

target_file = None
for path in search_patterns:
    if os.path.exists(path):
        target_file = path
        break

if not target_file:
    # Try finding any file starting with student_placement_data_v2
    matches = glob.glob('**/student_placement_data_v2.*', recursive=True)
    if matches:
        target_file = matches[0]

if not target_file:
    raise FileNotFoundError("Could not find student_placement_data_v2 (.csv or .xlsx) file.")

print(f"[*] Found target dataset: {target_file}")

# 2. Read dataset depending on format
is_excel = target_file.endswith('.xlsx') or target_file.endswith('.xls')
if is_excel:
    df = pd.read_excel(target_file)
else:
    df = pd.read_csv(target_file)

n = len(df)
np.random.seed(42)

# 3. Generate high-variance randomized CGPAs across 5.0 to 10.0
# Blend uniform noise with a broad normal distribution
broad_random = np.random.uniform(5.05, 9.95, size=n)
normal_curve = np.random.normal(loc=7.6, scale=1.3, size=n)

# Combine both distributions to get full spectrum coverage with realistic college density
combined = 0.55 * broad_random + 0.45 * normal_curve

# If rubric levels exist, add subtle correlation so top coders lean higher
tier_cols = [c for c in ['Lx_Level_Reached', 'Ax_Level_Reached', 'Cx_Level_Reached', 'Px_Level_Reached', 'Sx_Level_Reached'] if c in df.columns]
if tier_cols:
    tier_strength = df[tier_cols].mean(axis=1)
    combined = combined * 0.75 + (5.0 + (tier_strength / 4.0) * 4.6) * 0.25 + np.random.normal(0, 0.4, size=n)

# Strictly bound between 5.00 and 10.00 rounded to 2 decimal places
df['cgpa'] = np.clip(np.round(combined, 2), 5.00, 10.00)

# Synchronize 10th and 12th board marks so they don't contradict the new CGPAs
for col in ['tenth_pct', '10th_percentage']:
    if col in df.columns:
        df[col] = np.clip(np.round(df['cgpa'] * 9.2 + np.random.normal(3, 5, size=n), 1), 45.0, 99.0)

for col in ['twelfth_pct', '12th_percentage']:
    if col in df.columns:
        df[col] = np.clip(np.round(df['cgpa'] * 8.9 + np.random.normal(2, 5.5, size=n), 1), 42.0, 98.5)

# 4. Save updated dataset back to original format
if is_excel:
    df.to_excel(target_file, index=False)
else:
    df.to_csv(target_file, index=False)

print("\n" + "="*50)
print(f"✓ CGPA randomized successfully in: {target_file}")
print(f"  Minimum CGPA: {df['cgpa'].min():.2f}")
print(f"  Maximum CGPA: {df['cgpa'].max():.2f}")
print(f"  Mean CGPA:    {df['cgpa'].mean():.2f}")
print(f"  Standard Dev: {df['cgpa'].std():.2f}")
print("="*50)
print("\nSample Preview (First 8 Rows):")
print(df[['student_id', 'cgpa'] if 'student_id' in df.columns else ['cgpa']].head(8))