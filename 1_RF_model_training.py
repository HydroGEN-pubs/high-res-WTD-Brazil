import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.inspection import permutation_importance
from scipy import stats

import pickle

out_dir = ''

#######################################
###### Load and split the dataset #####
####################################### 

prepped_data = pd.read_csv('prepped_data_final_april2026.csv')
# # Note: In addition to the well data, we assume WTD=0 along 6,000 random locations along the river network
# This is used to enhance the model's ability to detect saturated conditions near water bodies 
# in addition to input data, 'river' is a column used to distinguish well data (0) from the points added along the river network (1)
prepped_data_noriver = prepped_data[prepped_data['river']==0] 

in_vars =['elevation','slope','x_dist_stream', 'hand',
           'pr','T2m', 'PME',
           'log_hydraulic_cond', 'sand','clay']
out_vars = ['mwtd']

# Select data that is complete and exclude dummy data for the train/test split
X = np.array(prepped_data_noriver[in_vars])
y = np.array(prepped_data_noriver.mwtd) 

print(f'Number of WTD observations: {len(X)}')

# Split into train and test (80% train; 20% test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

# Add points along the river to the training dataset but not in testing. 
river = prepped_data[['river', 'num_observations'] +out_vars + in_vars]
river = river[river['river']==1]
river['num_observations']= 1
X_river = np.array(river[in_vars])
y_river = np.array(river.mwtd)

print(f'With additional points along the river: {len(X_river)}')

X_train = np.concatenate((X_train, X_river), axis=0)
y_train = np.concatenate((y_train, y_river), axis=0)

np.save('X_train', X_train)
np.save('y_train', y_train)
np.save('X_test', X_test)
np.save('y_test', y_test)


################################
###### Random Forest model #####
################################

# Hyperparameter's 
max_samples = 0.8
max_features = 0.4 
n_estimators = 300 
max_depth = 20 
min_samples_leaf = 1

print(
    f"Hyperparameters:\n"
    f"  max_samples      = {max_samples}\n"
    f"  max_features     = {max_features}\n"
    f"  n_estimators     = {n_estimators}\n"
    f"  max_depth        = {max_depth}\n"
    f"  min_samples_leaf = {min_samples_leaf}"
)

# Train RF model 
RF_model = RandomForestRegressor(bootstrap=True,
                         max_samples=max_samples,
                         max_features=max_features,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         random_state=42, #to make sure model doesn't change
                         oob_score=False) #KEEP THIS: no out-of-bag because evaluation during testing


RF_model.fit(X_train, y_train)

#save model
with open(out_dir+'RF_model_FINAL.pkl', 'wb') as file:
    pickle.dump(RF_model, file)

#Calculate the median of the results of the decision trees in the random forest
estimators = RF_model.estimators_
train_results = []
test_results = []

#Obtain results from each tree
#estimators[n].predict gets prediction from each tree; instead of using RF_model.predict which would directly give the mean of trees
for n in range(n_estimators):
    train_results.append(estimators[n].predict(X_train)) 
    test_results.append(estimators[n].predict(X_test))

median_train_result = np.median(np.array(train_results), axis=0)
median_test_result = np.median(np.array(test_results), axis=0)

np.save('train_results_median.npy', median_train_result)
np.save('test_results_median.npy', median_test_result)

iqr_train = np.percentile(train_results, 75, axis=0) - np.percentile(train_results, 25, axis=0)
iqr_test = np.percentile(test_results, 75, axis=0) - np.percentile(test_results, 25, axis=0)
np.save('train_results_iqr.npy', iqr_train)
np.save('test_results_iqr.npy', iqr_test)   


###################################
###### Performance statistics #####
###################################

### Root mean square error (RMSE) ###
train_rmse_score = rmse(y_train,median_train_result)
test_rmse_score = rmse(y_test,median_test_result)
print('Train RMSE:',train_rmse_score)
print('Test RMSE:',test_rmse_score)

### Pearson correlation, r ####
pearson_train = stats.pearsonr(y_train,  median_train_result)
pearson_test = stats.pearsonr(y_test,  median_test_result)
print('Train Pearson-R:',pearson_train.statistic)
print('Test Pearson-R:',pearson_test.statistic)

### Mean Bias ###
train_bias_score = np.mean(np.asarray(median_train_result) - np.asarray(y_train))
test_bias_score = np.mean(np.asarray(median_test_result) - np.asarray(y_test))
print('Train Bias:',train_bias_score)
print('Test Bias:',test_bias_score)

### Mean absolute error (MAE) ###
train_mae_score = np.mean(np.abs(np.asarray(median_train_result) - np.asarray(y_train)))
test_mae_score = np.mean(np.abs(np.asarray(median_test_result) - np.asarray(y_test)))
print('Train MAE:',train_mae_score)
print('Test MAE:',test_mae_score)

############################
## For relative claculations let's use the overall dataset mean ####
overall_mean = prepped_data_noriver["mwtd"].mean()

###  relative RMSE ###
train_rrmse_score = train_rmse_score / overall_mean
test_rrmse_score = test_rmse_score/ overall_mean
print('Train r-RMSE:',train_rrmse_score)
print('Test r-RMSE:',test_rrmse_score)
###  relative MAE ###
train_rmae_score = train_mae_score / overall_mean
test_rmae_score = test_mae_score/ overall_mean
print('Train r-MAE:', train_rmae_score)
print('Test r-MAE:', test_rmae_score)


results = []

results.append({
    'train_rmse':     train_rmse_score,
    'test_rmse':      test_rmse_score,
    'train_mae':      train_mae_score,
    'test_mae':       test_mae_score,
    'train_bias':     train_bias_score,
    'test_bias':      test_bias_score,
    'train_pearson_r':  pearson_train.statistic,
    'test_pearson_r':   pearson_test.statistic,
    'train_rrmse':     train_rrmse_score,
    'test_rrmse':      test_rrmse_score,
    'train_rmae':      train_rmae_score,
    'test_rmae':       test_rmae_score,
})


np.save('statistics_RF.npy', results)



###########################################
###### Permutation Feature Importance #####
###########################################
in_vars =['elevation','slope','x_dist_stream', 'hand',
           'pr','T2m', 'PME',
           'log_hydraulic_cond', 'sand','clay']

## Permutation feature importance
r = permutation_importance(RF_model, X_test, y_test,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{in_vars[i]:<15}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")


importances_mean = r.importances_mean # Transpose to have features as rows
feature_names = in_vars


##### PLOTTING #####
data = {"feature": feature_names, "importance": importances_mean}

# Mapping to nicer labels
feature_labels = {
    "elevation": "elevation",
    "T2m": "avg. T",
    "hand": "HAND",
    "log_hydraulic_cond": "log(K)",
    "pr": "avg. P",
    "PME": "PME",
    "sand": "sand",
    "clay": "clay",
    "x_dist_stream": "horiz. dist. to stream",
    "slope": "slope"
}


# Create DataFrame
df = pd.DataFrame(data)

# Replace feature names with labels
df["feature"] = df["feature"].map(feature_labels)

# Sort (so largest ends up at top in horizontal bar plot)
df = df.sort_values(by="importance", ascending=True)
df.to_csv('feature_importance.csv', index=False)

# Plot
plt.figure(figsize=(8, 6))
plt.barh(df["feature"], df["importance"])

# Labels and title
plt.xlabel("Decrease in Accuracy Score", fontsize=12)
plt.title("Permutation Feature Importance", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(fontsize=12)

plt.tight_layout()
plt.savefig('feature_importance_bar.pdf', dpi=600)