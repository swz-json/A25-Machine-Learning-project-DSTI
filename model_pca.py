from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

malware = pd.read_csv('smote_malware.csv')
ddos = pd.read_csv('smote_ddos.csv')
intrusion = pd.read_csv('smote_intrusion.csv')

malware = malware.drop('device_os', axis=1)
ddos = ddos.drop('device_os', axis=1)
intrusion = intrusion.drop('device_os', axis=1)

cat_cols = malware.select_dtypes('object').columns.to_list()

malware_x = malware.drop('target', axis=1)
malware_y = malware['target']

ddos_x = ddos.drop('target', axis=1)
ddos_y = ddos['target']

intrusion_x = intrusion.drop('target', axis=1)
intrusion_y = intrusion['target']

y_combined = np.concatenate([malware_y, ddos_y, intrusion_y])

le = LabelEncoder()
le.fit(y_combined)

malware_y_encoded = le.transform(malware_y)
ddos_y_encoded = le.transform(ddos_y)
intrusion_y_encoded = le.transform(intrusion_y)

X_train_malware, X_test_malware, y_train_malware, y_test_malware = train_test_split(
    malware_x, malware_y_encoded, test_size=0.2, random_state=42, stratify=malware_y_encoded
)

X_train_ddos, X_test_ddos, y_train_ddos, y_test_ddos = train_test_split(
    ddos_x, ddos_y_encoded, test_size=0.2, random_state=42, stratify=ddos_y_encoded
)

X_train_intrusion, X_test_intrusion, y_train_intrusion, y_test_intrusion = train_test_split(
    intrusion_x, intrusion_y_encoded, test_size=0.2, random_state=42, stratify=intrusion_y_encoded
)

X_train_combined = pd.concat([X_train_malware, X_train_ddos, X_train_intrusion], ignore_index=True)
y_train_combined = np.concatenate([y_train_malware, y_train_ddos, y_train_intrusion])

# print(f"Combined training data shape: {X_train_combined.shape}")
# print(f"Combined training labels shape: {y_train_combined.shape}")

cat_cols = X_train_combined.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train_combined.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
], remainder='passthrough')

X_train_transformed = preprocessor.fit_transform(X_train_combined)

pca = PCA().fit(X_train_transformed)
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

n_points = len(exp_var)
coords = np.vstack((range(n_points), exp_var)).T
first_pt = coords[0]
last_pt = coords[-1]
line_vec = last_pt - first_pt
line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
vec_from_first = coords - first_pt
scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
vec_parallel = np.outer(scalar_prod, line_vec_norm)
dist_to_line = np.sqrt(np.sum((vec_from_first - vec_parallel)**2, axis=1))
elbow_point = np.argmax(dist_to_line) + 1 # +1 for 1-based indexing

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(exp_var) + 1), exp_var, 'o-', markersize=4, color='royalblue')
plt.axvline(x=elbow_point, color='red', linestyle='--', label=f'Elbow Point: {elbow_point}')
plt.title('Scree Plot: Elbow Method')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Total original features: {X_train_transformed.shape[1]}")
print(f"Recommended components (Elbow): {elbow_point}")
print(f"Variance explained at elbow: {cum_var[elbow_point-1]*100:.2f}%")

X_test_combined = pd.concat([X_test_malware, X_test_ddos, X_test_intrusion], ignore_index=True)
y_test_combined = np.concatenate([y_test_malware, y_test_ddos, y_test_intrusion])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=elbow_point)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

print(f"Training Random Forest with {elbow_point} PCA components...")
model_pipeline.fit(X_train_combined, y_train_combined)

y_pred = model_pipeline.predict(X_test_combined)

print(f"Overall Accuracy: {accuracy_score(y_test_combined, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_combined, y_pred, target_names=le.classes_))

plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test_combined, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Malware, DDoS, & Intrusion')
plt.show()

with open('rf_pca_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)