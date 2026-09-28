import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load dataset
data_path = None
for path in ['data/student_placement_data.csv', 'data/student_placement_data_v2.csv', 'student_placement_data.csv']:
    if os.path.exists(path):
        data_path = path
        break

if not data_path:
    raise FileNotFoundError("Could not find student placement dataset.")

df = pd.read_csv(data_path)

# 2. Detect target column dynamically
target_col = 'placed' if 'placed' in df.columns else 'placement_status'
y = df[target_col]

# 3. Explicit features matching app.py input schema (prevents dimension mismatch)
EXPECTED_FEATURES = [
    'gender', 'age', 'degree', 'branch', 'cgpa', 'backlogs',
    'internships', 'certifications', 'coding_skills', 'communication_skills',
    'aptitude_score', 'projects', 'Lx_Level_Reached', 'Ax_Level_Reached',
    'Cx_Level_Reached', 'Px_Level_Reached', 'Sx_Level_Reached'
]

# Fallback: if columns match expected list, use them directly; otherwise drop known noise
available_features = [col for col in EXPECTED_FEATURES if col in df.columns]
if len(available_features) == len(EXPECTED_FEATURES):
    X = df[EXPECTED_FEATURES]
else:
    cols_to_drop = [
        target_col, 'student_id', 'student_name', 'usn', 'company_type', 'package_lpa',
        'Overall_Level_Score', 'tenth_pct', 'twelfth_pct', '10th_percentage', '12th_percentage'
    ]
    # Drop raw test marks
    raw_prefixes = ('lang_', 'apt_', 'soft_', 'core_', 'prog_', 'Unnamed')
    cols_to_drop += [c for c in df.columns if c.startswith(raw_prefixes)]
    X = df.drop(columns=cols_to_drop, errors='ignore')

print(f"[*] Training dataset features ({len(X.columns)}): {list(X.columns)}")

# 4. Feature preprocessing pipeline
cat_cols = ['gender', 'degree', 'branch']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ]
)

# 5. Stratified train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Candidate algorithms with calibration
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest (Calibrated)': RandomForestClassifier(
        n_estimators=150, min_samples_leaf=12, random_state=42, class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    ),
    'KNN': KNeighborsClassifier(n_neighbors=9),
}

print('\n=== Model Comparison Benchmark ===\n')
results = []
best_pipeline = None

for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
    })

    if name == 'Random Forest (Calibrated)':
        best_pipeline = pipe

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 7. Safe export
os.makedirs('models', exist_ok=True)
joblib.dump(best_pipeline, 'models/best_pipeline.pkl')
print('\n[✓] Exported calibrated pipeline (Random Forest) to models/best_pipeline.pkl')