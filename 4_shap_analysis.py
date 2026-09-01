import pickle
import numpy as np
import shap

out_dir = ''

# Load model
with open(out_dir+'RF_model_FINAL.pkl', 'rb') as file:
    RF_model = pickle.load(file)


# Load test data
X_test = np.load('X_test.npy')

# Randomly select a subset of the test points (if wanted, to decrease compute time)
# For final results we used the full test dataset.
np.random.seed(42)
n_rows = X_test.shape[0]
subset = n_rows // 2  # half
indices = np.random.choice(n_rows, size=subset, replace=False) # randomly choose row indices without replacement
X_test_subset = X_test[indices]

# TreeExplainer for fast, exact SHAP for RandomForest
explainer = shap.TreeExplainer(RF_model)

# Compute SHAP values for all test samples
shap_values = explainer.shap_values(X_test_subset)

# shap_values has shape (n_samples, n_features)
print(f'shap_values shape {shap_values.shape}')

# Save results
np.save('shap_values_FINAL_half.npy', shap_values)
print("SHAP values saved as 'shap_values_FINAL_half.npy'")