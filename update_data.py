import numpy as np
import pandas as pd

# Load existing dataset
df = pd.read_csv('data/student_placement_data.csv')

first_m = [
    'Aarav',
    'Rohan',
    'Vikram',
    'Siddharth',
    'Karan',
    'Rahul',
    'Arjun',
    'Varun',
    'Manish',
    'Nikhil',
    'Aditya',
    'Abhishek',
    'Gaurav',
    'Pranav',
    'Harsh',
]
first_f = [
    'Aditi',
    'Priya',
    'Ananya',
    'Neha',
    'Sneha',
    'Pooja',
    'Isha',
    'Divya',
    'Kavya',
    'Tanvi',
    'Riya',
    'Shreya',
    'Anushka',
    'Meera',
    'Kritika',
]
last = [
    'Sharma',
    'Verma',
    'Patel',
    'Rao',
    'Nair',
    'Mehta',
    'Gupta',
    'Iyer',
    'Reddy',
    'Singh',
    'Kumar',
    'Joshi',
    'Bhat',
    'Kulkarni',
    'Deshmukh',
]

np.random.seed(42)

names = []
for g in df['gender']:
  if g == 'Male':
    fn = np.random.choice(first_m)
  else:
    fn = np.random.choice(first_f)
  ln = np.random.choice(last)
  names.append(f'{fn} {ln}')

branch_map = {
    'CS': 'CS',
    'IT': 'IS',
    'AI': 'AI',
    'DS': 'CD',
    'Electrical': 'EE',
    'Mechanical': 'ME',
}

usns = []
for _, row in df.iterrows():
  b_code = branch_map.get(row['branch'], 'CS')
  batch_yr = 25 - (int(row['age']) - 18)
  usns.append(f"1CR{batch_yr:02d}{b_code}{int(row['student_id']):04d}")

# Insert or update student_name and usn columns
if 'student_name' not in df.columns:
  df.insert(1, 'student_name', names)
else:
  df['student_name'] = names

if 'usn' not in df.columns:
  df.insert(2, 'usn', usns)
else:
  df['usn'] = usns

df.to_csv('data/student_placement_data.csv', index=False)
print('Done! student_name and usn columns added successfully.')