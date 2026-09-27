import os
import numpy as np
import pandas as pd

np.random.seed(42)
n = 12000

os.makedirs('data', exist_ok=True)

# 1. Base Attributes
student_id = np.arange(1, n + 1)
gender = np.random.choice(['Male', 'Female'], size=n, p=[0.58, 0.42])
age = np.random.choice([20, 21, 22, 23, 24], size=n, p=[0.15, 0.35, 0.30, 0.15, 0.05])
degree = np.random.choice(['BE', 'BTech', 'BCA', 'BSc'], size=n, p=[0.40, 0.40, 0.12, 0.08])
branch = np.random.choice(['CS', 'IT', 'AI', 'DS', 'Electrical', 'Mechanical'], size=n)

cgpa = np.clip(np.random.normal(7.65, 1.2), 5.5, 9.8).round(2)
backlogs = np.random.choice([0, 1, 2, 3], size=n, p=[0.45, 0.30, 0.15, 0.10])
internships = np.random.choice([0, 1, 2, 3], size=n, p=[0.35, 0.35, 0.20, 0.10])
certifications = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
coding_skills = np.random.choice(range(1, 11), size=n)
communication_skills = np.random.choice(range(1, 11), size=n)
aptitude_score = np.clip(np.random.normal(69.3, 17.3), 40, 99).round().astype(int)
projects = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])

df = pd.DataFrame({
    'student_id': student_id,
    'gender': gender,
    'age': age,
    'degree': degree,
    'branch': branch,
    'cgpa': cgpa,
    'backlogs': backlogs,
    'internships': internships,
    'certifications': certifications,
    'coding_skills': coding_skills,
    'communication_skills': communication_skills,
    'aptitude_score': aptitude_score,
    'projects': projects
})

# Feature normalization
comm_norm = (df['communication_skills'] - 1) / 9.0
apt_norm = (df['aptitude_score'] - 40) / 59.0
code_norm = (df['coding_skills'] - 1) / 9.0
cgpa_norm = (df['cgpa'] - 5.5) / 4.3

# 2. Test Pillar Scores
# Language Lx (L1 >= 65, L2 >= 65, L3 >= 70, L4 >= 70)
df['lang_L1'] = np.clip(np.random.normal(comm_norm * 35 + 62, 7), 30, 100).round().astype(int)
df['lang_L2'] = np.clip(np.random.normal(comm_norm * 35 + 62, 7), 30, 100).round().astype(int)
df['lang_L3'] = np.clip(np.random.normal(comm_norm * 35 + 60, 8), 20, 100).round().astype(int)
df['lang_L4'] = np.clip(np.random.normal(comm_norm * 35 + 58, 9), 20, 100).round().astype(int)

# Aptitude Ax (A1 >= 50, A2 >= 50, A3 >= 50, A4 >= 50)
df['apt_A1'] = np.clip(np.random.normal(apt_norm * 50 + 45, 8), 10, 100).round().astype(int)
df['apt_A2'] = np.clip(np.random.normal(apt_norm * 50 + 45, 8), 10, 100).round().astype(int)
df['apt_A3'] = np.clip(np.random.normal(apt_norm * 50 + 42, 9), 10, 100).round().astype(int)
df['apt_A4'] = np.clip(np.random.normal(apt_norm * 50 + 40, 10), 10, 100).round().astype(int)

# Softskills Sx (S1 >= 50, S2 >= 50, S3 >= 50, S4 >= 50)
df['soft_S1'] = np.clip(np.random.normal(comm_norm * 45 + 48, 8), 10, 100).round().astype(int)
df['soft_S2'] = np.clip(np.random.normal(comm_norm * 45 + 48, 8), 10, 100).round().astype(int)
df['soft_S3'] = np.clip(np.random.normal(comm_norm * 45 + 44, 9), 10, 100).round().astype(int)
df['soft_S4'] = np.clip(np.random.normal(comm_norm * 45 + 42, 10), 10, 100).round().astype(int)

# Core Cx (C2 >= 10/25, C3 Odd >= 15/100, C3 Full >= 25/100, C4 >= 50/100, C5 >= 50/100)
df['core_C2_Odd'] = np.clip(np.random.normal(cgpa_norm * 12 + 10, 3), 0, 25).round().astype(int)
df['core_C2_Full'] = np.clip(np.random.normal(cgpa_norm * 12 + 10, 3), 0, 25).round().astype(int)
df['core_C3_Odd'] = np.clip(np.random.normal(cgpa_norm * 50 + 35, 12), 0, 100).round().astype(int)
df['core_C3_Full'] = np.clip(np.random.normal(cgpa_norm * 50 + 35, 12), 0, 100).round().astype(int)
df['core_C4_Odd'] = np.clip(np.random.normal(cgpa_norm * 50 + 30, 14), 0, 100).round().astype(int)
df['core_C4_Full'] = np.clip(np.random.normal(cgpa_norm * 50 + 30, 14), 0, 100).round().astype(int)
df['core_C5_Full'] = np.clip(np.random.normal(cgpa_norm * 50 + 25, 15), 0, 100).round().astype(int)

# Programming Px (P1 >= 50, P2 >= 50, P3 >= 60, P4 >= 70)
df['prog_P1_C'] = np.clip(np.random.normal(code_norm * 50 + 45, 10), 0, 100).round().astype(int)
df['prog_P2_Python'] = np.clip(np.random.normal(code_norm * 50 + 45, 10), 0, 100).round().astype(int)
df['prog_P3_Python'] = np.clip(np.random.normal(code_norm * 50 + 40, 12), 0, 100).round().astype(int)
df['prog_P3_Java'] = np.clip(np.random.normal(code_norm * 50 + 40, 12), 0, 100).round().astype(int)
df['prog_P4_Prog1'] = np.clip(np.random.normal(code_norm * 45 + 38, 14), 0, 100).round().astype(int)
df['prog_P4_Prog2'] = np.clip(np.random.normal(code_norm * 45 + 38, 14), 0, 100).round().astype(int)
df['prog_P4_MAD_FSD'] = np.clip(np.random.normal(code_norm * 45 + 35, 15), 0, 100).round().astype(int)
df['prog_P4_DS'] = np.clip(np.random.normal(code_norm * 45 + 35, 15), 0, 100).round().astype(int)

# 3. Compute Level Reached
def get_lx_level(r):
    lvl = 0
    if r['lang_L1'] >= 65:
        lvl = 1
        if r['lang_L2'] >= 65:
            lvl = 2
            if r['lang_L3'] >= 70:
                lvl = 3
                if r['lang_L4'] >= 70: lvl = 4
    return lvl

def get_ax_level(r):
    lvl = 0
    if r['apt_A1'] >= 50:
        lvl = 1
        if r['apt_A2'] >= 50:
            lvl = 2
            if r['apt_A3'] >= 50:
                lvl = 3
                if r['apt_A4'] >= 50: lvl = 4
    return lvl

def get_sx_level(r):
    lvl = 0
    if r['soft_S1'] >= 50:
        lvl = 1
        if r['soft_S2'] >= 50:
            lvl = 2
            if r['soft_S3'] >= 50:
                lvl = 3
                if r['soft_S4'] >= 50: lvl = 4
    return lvl

def get_cx_level(r):
    lvl = 0
    if (r['core_C2_Odd'] >= 10) and (r['core_C2_Full'] >= 10):
        lvl = 2
        if (r['core_C3_Odd'] >= 15) and (r['core_C3_Full'] >= 25):
            lvl = 3
            if (r['core_C4_Odd'] >= 50) and (r['core_C4_Full'] >= 50):
                lvl = 4
                if r['core_C5_Full'] >= 50: lvl = 5
    return lvl

def get_px_level(r):
    lvl = 0.0
    if r['prog_P1_C'] >= 50:
        lvl = 1.0
        if r['prog_P2_Python'] >= 50:
            lvl = 2.0
            p3_py = r['prog_P3_Python'] >= 60
            p3_java = r['prog_P3_Java'] >= 60
            if p3_py and p3_java: lvl = 3.5
            elif p3_py or p3_java: lvl = 3.0
            else: return lvl
            if (r['prog_P4_Prog1'] >= 70) and (r['prog_P4_Prog2'] >= 70):
                lvl = 4.0
                if (r['prog_P4_MAD_FSD'] >= 70) and (r['prog_P4_DS'] >= 70): lvl = 5.0
    return lvl

df['Lx_Level_Reached'] = df.apply(get_lx_level, axis=1)
df['Ax_Level_Reached'] = df.apply(get_ax_level, axis=1)
df['Cx_Level_Reached'] = df.apply(get_cx_level, axis=1)
df['Px_Level_Reached'] = df.apply(get_px_level, axis=1)
df['Sx_Level_Reached'] = df.apply(get_sx_level, axis=1)

# Overall Readiness Score
df['Overall_Level_Score'] = df[['Lx_Level_Reached', 'Ax_Level_Reached', 'Cx_Level_Reached', 'Px_Level_Reached', 'Sx_Level_Reached']].min(axis=1)

# Placement Rule: Level >= 3.0 gets placed
df['placed'] = np.where(df['Overall_Level_Score'] >= 3.0, 1, 0)

# Salary Range 4 LPA to 20 LPA for Placed Students
package = np.zeros(n)
comp_type = ['Not Placed'] * n

for i in range(n):
    score = df.loc[i, 'Overall_Level_Score']
    if score == 3.0:
        # Level 3: 4.0 LPA to 5.0 LPA
        package[i] = round(np.random.uniform(4.0, 5.0), 2)
        comp_type[i] = np.random.choice(['Mass Recruiter / IT Services', 'Service Based'])
    elif score == 3.5:
        # Level 3.5: 5.0 LPA to 8.0 LPA
        package[i] = round(np.random.uniform(5.0, 8.0), 2)
        comp_type[i] = np.random.choice(['IT Services (Digital/Specialist)', 'Mid-tier Product'])
    elif score == 4.0:
        # Level 4.0: 8.0 LPA to 14.0 LPA
        package[i] = round(np.random.uniform(8.0, 14.0), 2)
        comp_type[i] = np.random.choice(['Product Based', 'Fintech', 'Consulting'])
    elif score >= 5.0:
        # Level 5.0: 14.0 LPA to 20.0 LPA
        package[i] = round(np.random.uniform(14.0, 20.0), 2)
        comp_type[i] = np.random.choice(['Tier-1 Tech MNC', 'AI/Cloud Lab', 'Global Product'])

df['package_lpa'] = package
df['company_type'] = comp_type

# Save to CSV
df.to_csv('data/student_placement_data_v2.csv', index=False)
print("SUCCESS: data/student_placement_data.csv generated successfully.")
print(f"Placed Salary Range: {df[df['placed'] == 1]['package_lpa'].min()} LPA - {df[df['placed'] == 1]['package_lpa'].max()} LPA")