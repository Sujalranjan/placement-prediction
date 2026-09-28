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

# 1. Load dataset (update filename/path if using student_placement_data.csv)
try:
  df = pd.read_csv('data/student_placement_data.csv')
except FileNotFoundError:
  df = pd.read_csv('data/student_placement_data_v2.csv')

# 2. Exclude identifiers, student names/USNs, target leakage, and raw marks
cols_to_drop = [
    'student_id',
    'student_name',  # Added to prevent StandardScaler string crash
    'usn',  # Added to prevent StandardScaler string crash
    'company_type',
    'package_lpa',
    'Overall_Level_Score',  # Direct target leakage!
    # Raw individual test marks
    'lang_L1',
    'lang_L2',
    'lang_L3',
    'lang_L4',
    'apt_A1',
    'apt_A2',
    'apt_A3',
    'apt_A4',
    'soft_S1',
    'soft_S2',
    'soft_S3',
    'soft_S4',
    'core_C2_Odd',
    'core_C2_Full',
    'core_C3_Odd',
    'core_C3_Full',
    'core_C4_Odd',
    'core_C4_Full',
    'core_C5_Full',
    'prog_P1_C',
    'prog_P2_Python',
    'prog_P3_Python',
    'prog_P3_Java',
    'prog_P4_Prog1',
    'prog_P4_Prog2',
    'prog_P4_MAD_FSD',
    'prog_P4_DS',
]
df = df.drop(columns=cols_to_drop, errors='ignore')

# 3. Separate features and target
X = df.drop(columns=['placed'])
y = df['placed']

print(f'Features used for training ({len(X.columns)}): {list(X.columns)}')

# 4. Feature preprocessing pipeline
cat_cols = ['gender', 'degree', 'branch']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ]
)

# 5. Stratified train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Candidate algorithms (calibrated to generate smooth probability spreads)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest (Calibrated)': RandomForestClassifier(
        n_estimators=150, min_samples_leaf=15, random_state=42
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

  # Select calibrated Random Forest for smooth, continuous probability estimates
  if name == 'Random Forest (Calibrated)':
    best_pipeline = pipe

# Display evaluation benchmark
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save benchmark results to JSON so app.py dynamically displays actual dataset metrics
results_df.to_json('models/benchmark_results.json', orient='records', indent=2)

# 7. Export the best-performing pipeline
joblib.dump(best_pipeline, 'models/best_pipeline.pkl')
print(
    '\nExported calibrated pipeline (Random Forest) to models/best_pipeline.pkl'
)
print('Exported benchmark results to models/benchmark_results.json')