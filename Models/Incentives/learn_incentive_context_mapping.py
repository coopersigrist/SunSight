'''
Learning the incentive context mapping (depricated)
'''
import numpy as np
import pandas as pd
from Models.Incentives.simulated_decision import *
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
import joblib

# Silence TF logs before importing TF.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics, Model

census_df = pd.read_csv('Data/Incentives/census_by_zip_complex.csv')
solar_by_zip_df = pd.read_csv('Data/Sunroof/solar_by_zip.csv')
all_state_factors = pd.read_csv('/Users/asitaram/Documents/GitHub/Untitled/SunSight/Data/Clean_Data/data_by_state_sum.csv')
calculated_multipliers = pd.read_csv('output_5.csv')
for col in calculated_multipliers.columns:
    if calculated_multipliers[col].dtype == object:
        try:
            calculated_multipliers[col] = calculated_multipliers[col].str.strip("[]").astype(float)
        except Exception:
            pass
    #all_state_factors['prop_adopted_status_quo'] = all_state_factors['prop_adopted_status_quo'].str.strip("[]").astype(float)
all_state_factors = pd.merge(all_state_factors, calculated_multipliers[['State code','payback_period_cutoff', 'prop_adopted_per_year_average','right_sized_multiplier','prop_adopted_status_quo', 'needed_phi_per_install', 'energy_consumption_kwh','median_npv','median_elec_consumption','median_energy_ratio','elec_cost' ]], on="State code", how="inner")

#all_state_factors['prop_adopted_per_year_average'] = 1/all_state_factors['right_sized_multiplier']

#search_engine = SearchEngine()


#training NN to predict multiplier
#factors that affect NN are...

#columns_state_df = ['Democrat_prop','Republican_prop', 'total_households','Median_income','households_below_poverty_line', 'black_prop','white_prop','asian_prop','yearly_sunlight_kwh_kw_threshold_avg','existing_installs_count_per_capita', "Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)","Adjusted Payback Period (Years, under energy generation assumptions)",'Numeric state-level upfront incentive']
#columns_state_df = ['Democrat_prop','Republican_prop', 'total_households','Median_income','households_below_poverty_line', 'black_prop','white_prop','asian_prop','yearly_sunlight_kwh_kw_threshold_avg','existing_installs_count_per_capita', "Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)","Adjusted Payback Period (Years, under energy generation assumptions)",'Numeric state-level upfront incentive']
all_possible_columns = ['Democrat_prop','Republican_prop', 'total_households','Median_income','households_below_poverty_line', 'black_prop','white_prop','asian_prop','yearly_sunlight_kwh_kw_threshold_avg','existing_installs_count_per_capita', "Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)","Adjusted Payback Period (Years, under energy generation assumptions)",'Numeric state-level upfront incentive']+ ['payback_period_cutoff', 'prop_adopted_status_quo','needed_phi_per_install', 'energy_consumption_kwh','median_npv','median_elec_consumption','median_energy_ratio','elec_cost']
#columns_state_df = ['Democrat_prop','Median_income','households_below_poverty_line', 'black_prop','existing_installs_count_per_capita','elec_cost' ]
columns_state_df = [
    "elec_cost",
    "median_npv",
    "median_elec_consumption",
    "median_energy_ratio",
    "Democrat_prop"
]
#columns_state_df=['per_capita_income','prop_adopted_status_quo', 'elec_cost', 'existing_installs_count_per_capita', 'households_below_poverty_line']
#all = ['Democrat_prop', 'total_households','Median_income', 'black_prop','white_prop','asian_prop','yearly_sunlight_kwh_kw_threshold_avg','existing_installs_count_per_capita', "Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)","Adjusted Payback Period (Years, under energy generation assumptions)",'Numeric state-level upfront incentive','right_sized_multiplier', 'prop_adopted_status_quo' ]
#columns_state_df = ['Democrat_prop', 'total_households','Median_income', 'black_prop','white_prop','asian_prop','yearly_sunlight_kwh_kw_threshold_avg','existing_installs_count_per_capita', 'prop_adopted_status_quo']
y_column = ['prop_adopted_per_year_average']
all = columns_state_df + y_column

all_state_truly_all = all_state_factors[all_possible_columns]
all_state_factors = all_state_factors[all]
def plot_actual_vs_pred(y_true, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="k")
    
    # Plot 45-degree line (perfect predictions)
    lims = [
        np.min([y_true.min(), y_pred.min()]),
        np.max([y_true.max(), y_pred.max()])
    ]
    plt.plot(lims, lims, "r--", label="Perfect Prediction")
    
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.show()
# 2. Prepare data
# X = features (numpy array or DataFrame)
# y = target (numpy array or Series)
X = all_state_factors[columns_state_df].to_numpy()
cols = all_state_factors[columns_state_df].columns
print(f'cols {cols}')
y = all_state_factors[y_column].to_numpy()

scaler = StandardScaler()
scaler_y = StandardScaler()
# Fit preprocessing on train, transform both
X = scaler.fit_transform(X)
#y = scaler_y.fit_transform(y)

corr_df = all_state_truly_all
corr_df['target'] = y
correlations = corr_df.corr(method='pearson')["target"].sort_values(ascending=False)
print(correlations)
print("Target mean:", np.mean(y))
print("Target std:", np.std(y))
print("Target min:", np.min(y))
print("Target max:", np.max(y))
# 3. Define model
rf = RandomForestRegressor(
    n_estimators=500,      # number of trees
    max_depth=None,        # can tune
    random_state=42,
    n_jobs=-1              # parallelism
)

gb= GradientBoostingRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=3, 
    subsample=0.8, 
    max_features="sqrt", 
    random_state=42
)




joblib.dump(gb, "gradient_boosting_model.pkl")

kf = KFold(n_splits=10, shuffle=True, random_state=42)

maes, rmses, r2s = [], [], []
maes_gb, rmses_gb, r2s_gb = [], [], []
all_y_true, all_y_pred = [],[]
all_y_true_gb, all_y_pred_gb = [],[]
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]


    # fit model
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # predict
    y_pred = rf.predict(X_test)
    y_pred_gb = gb.predict(X_test)

    # evaluate
    maes.append(mean_absolute_error(y_test, y_pred))
    rmses.append(mean_squared_error(y_test, y_pred, squared=False))
    r2s.append(r2_score(y_test, y_pred))

    maes_gb.append(mean_absolute_error(y_test, y_pred_gb))
    rmses_gb.append(mean_squared_error(y_test, y_pred_gb, squared=False))
    r2s_gb.append(r2_score(y_test, y_pred_gb))
    all_y_true.append(y_test)
    all_y_pred.append(y_pred)
    all_y_true_gb.append(y_test)
    all_y_pred_gb.append(y_pred_gb)
all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)
all_y_pred_gb = np.concatenate(all_y_pred_gb)
plot_actual_vs_pred(all_y_true, all_y_pred)
plot_actual_vs_pred(all_y_true, all_y_pred_gb)
# 6. Report average CV metrics
print('RANDOM FOREST')
print("MAE:", np.mean(maes), "+-", np.std(maes))
print("RMSE:", np.mean(rmses), "+-", np.std(rmses))
print("R²:", np.mean(r2s), "+-", np.std(r2s))
print('GB')
print("MAE:", np.mean(maes_gb), "+-", np.std(maes_gb))
print("RMSE:", np.mean(rmses_gb), "+-", np.std(rmses_gb))
print("R²:", np.mean(r2s_gb), "+-", np.std(r2s_gb))

rf.fit(X, y)

joblib.dump(rf, "random_forest_model.pkl")
# 8. Feature importances
importances = rf.feature_importances_
feat_imp = pd.DataFrame({
    "feature": cols,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feat_imp.head(20))




def make_mlp(input_dim: int, hidden: List[int]) -> Model:
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

    out = layers.Dense(1, activation="linear")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanAbsoluteError(name="mae"), metrics.RootMeanSquaredError(name="rmse")],
    )
    return model



def ensure_binary_y(y: pd.Series) -> Tuple[np.ndarray, Optional[LabelBinarizer]]:
    if pd.api.types.is_bool_dtype(y) or set(pd.unique(y.dropna())) <= {0, 1}:
        return y.astype(int).to_numpy(), None
    lb = LabelBinarizer()
    ybin = lb.fit_transform(y.to_numpy())
    if ybin.ndim == 2 and ybin.shape[1] == 1:
        ybin = ybin.ravel()
    return ybin.astype(int), lb

def permutation_importance_nn(model, X_val, y_val, metric_func):
    base_pred = model.predict(X_val).ravel()
    #print(metric_func)
    base_score = metric_func(y_val, base_pred)
    importances = {}

    for i, col in enumerate(columns_state_df):
        X_perm = X_val.copy()
        np.random.shuffle(X_perm[:, i])
        perm_pred = model.predict(X_perm).ravel()
        perm_score = metric_func(y_val, perm_pred)
        importances[col] = base_score - perm_score  # positive = feature helps
    return importances

def permutation_importance_cv(model, X, y, feature_names, metric='r2', n_splits=1):
    """
    Computes permutation importance across K-Fold CV.
    
    model_fn: function that returns a compiled model
    X: numpy array or pandas DataFrame
    y: numpy array or pandas Series/DataFrame
    feature_names: list of feature names
    """
    # Force to NumPy arrays for safe indexing
    X_np = np.asarray(X)  
    y_np = np.asarray(y).ravel()  # flatten to 1D if needed

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    importances_list = []
    input_dim = X_np.shape[1]
    hidden = [8]
    model = make_mlp(input_dim, hidden)
    all_y_true, all_y_pred = [], []
    for train_idx, val_idx in kf.split(X_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        
        model.fit(
        X_train,
        y_train,
        validation_split=0,         # 20% of training set for validation
        epochs=100,                   # max 100 passes (with early stopping)
        batch_size=32,                # small batch size works well for tabular data
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ],
        verbose=1                     # progress bar style logs
        )

        df_imp = permutation_importance_nn(model, X_val, y_val, metric)
        importances_list.append(df_imp)
        y_pred = model.predict(X_val).ravel()
        all_y_true.append(y_val)
        all_y_pred.append(y_pred)
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    plot_actual_vs_pred(all_y_true, all_y_pred)
    # Average importances across folds
    print(importances_list)
    df = pd.DataFrame(importances_list)

    # compute mean across folds
    avg_importances = df.mean(axis=0).sort_values(ascending=False)
    #avg_importances = np.mean(importances_list, axis=0)
    df_avg = pd.DataFrame({
        #'feature': feature_names,
        'avg_importance': avg_importances
    }).sort_values(by='avg_importance', ascending=False)
    # Flag harmful vs helpful
    df_avg['flag'] = df_avg['avg_importance'].apply(lambda x: 'harmful/neutral' if x <= 0 else 'helpful')

    return df_avg


def main():
    
    # Split
    train_df, test_df = train_test_split(all_state_factors, test_size=10, random_state=42, stratify=None)
    y_train = train_df[y_column]
    y_test = test_df[y_column]
    y_all = all_state_factors[y_column]

    X_train_raw = train_df[columns_state_df]
    X_test_raw = test_df[columns_state_df]

    scaler = StandardScaler()
    scaler_y = StandardScaler()
    y_all = scaler_y.fit_transform(y_all)
    y_test = scaler_y.transform(y_test)
    y_train = scaler_y.transform(y_train)

    # Fit preprocessing on train, transform both
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_all = scaler.transform(all_state_factors[columns_state_df])

    input_dim = X_train.shape[1]
    hidden = [len(columns_state_df)]
    model = make_mlp(input_dim, hidden)

    '''cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(3, args.patience // 4), factor=0.5, min_lr=1e-6),
    ]'''

    history = None
    metrics_out = {"task": 'regression'}
    # Regression
    #y_train_vals = y_train.to_numpy(dtype=float)
    y_test_vals = y_test
    history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,         # 20% of training set for validation
    epochs=100,                   # max 100 passes (with early stopping)
    batch_size=32,                # small batch size works well for tabular data
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ],
    verbose=1                     # progress bar style logs
)
    y_pred = model.predict(X_all, verbose=0).ravel()
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
    #y_train = scaler_y.transform(y_train)
    #print(permutation_importance_nn(model, X_train, y_train, r2_score))
    print(permutation_importance_cv(model, X_all, y_all, columns_state_df, metric=r2_score, n_splits=5))
    mae = mean_absolute_error(y_all, y_pred)
    rmse = mean_squared_error(y_all, y_pred, squared=False)
    r2 = r2_score(y_all, y_pred)
    metrics_out.update({"mae": mae, "rmse": rmse, "r2": r2, "n_test": int(len(y_all))})
    print(metrics_out)

    # Persist artifacts
    model_path = "solar_nn.keras"
    pre_path = "preprocess.joblib"
    metrics_path =  "metrics.json"

    model.save(model_path, include_optimizer=True)
    joblib.dump(scaler, pre_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(json.dumps(metrics_out, indent=2))


if __name__ == "__main__":
    main()


def learn_prop_cap_added_yearly():
     # Loop over states and fit regression
    for state, group in total_cap_per_year_per_state.groupby(["State"]):
        print(state[0])
        print(group)
        #print(total_cap_per_year_per_state.columns)
        years = total_cap_per_year_per_state[total_cap_per_year_per_state['State'] == state[0]]['Year'].to_list()
        group = group.replace(np.nan, 0)
        #X = group["Year"].values.reshape(-1, 1) 
        X = np.array(years).reshape(-1,1)
        print(X)# independent variable: Year
        y = group["Prop_Added_Capacity"].values          # dependent variable: yearly diff
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)  # goodness of fit
        
        results.append({
            "State": state,
            "Slope": slope,
            "Intercept": intercept,
            "R2": r2
        })

    # Convert to DataFrame and sort by R² or slope
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

    print(results_df)

def get_context_mapping_per_zipcode(save_dir='Data/Incentives/context_mapping_per_zipcode.csv'):
    '''
    each zip code has a context mapping of census information
    buckets are:
    black_prop
    hispanic_prop
    households_below_poverty_line_prop
    white_prop
    asian_prop
    income_prop
    income_bucket
    region
    existing_install_counts
    '''
    income_buckets = [-np.inf, 25000, 50000, 75000, 100000, np.inf]
    install_buckets = [-np.inf, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, np.inf]
    poverty_buckets = [-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf]
    race_buckets = [-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf]
    region_buckets = len(census_df['region_name'].unique())

    merged_df = pd.merge(census_df, solar_by_zip_df, on='zip', how='left')
    merged_df['black_prop'] = merged_df['black_population'] / merged_df['total_households']
    merged_df['hispanic_prop'] = merged_df['hispanic_population'] / merged_df['total_households']
    merged_df['white_prop'] = merged_df['white_population'] / merged_df['total_households']
    merged_df['asian_prop'] = merged_df['asian_population'] / merged_df['total_households']
    merged_df['income_bucket'] = pd.cut(merged_df['median_income'], bins=[-np.inf, 25000, 50000, 75000, 100000, np.inf], labels=['<25k', '25-50k', '50-75k', '75-100k', '>100k'])
    merged_df['existing_install_counts_bucket'] = pd.cut(merged_df['existing_installs_count'], bins=[-np.inf, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, np.inf], labels=['0', '1-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000', '>1000'])
    merged_df['households_below_poverty_line_prop'] = merged_df['households_below_poverty_line'] / merged_df['total_households']
    merged_df['households_below_poverty_line_prop_bucket'] = pd.cut(merged_df['households_below_poverty_line_prop'], bins=[-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf], labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'])

    incentive_context_total = merged_df['region_name'].unique().shape[0] * merged_df['income_bucket'].unique().shape[0] * merged_df['existing_install_counts_bucket'].unique().shape[0] * merged_df['households_below_poverty_line_prop_bucket'].unique().shape[0] * merged_df['black_prop'].unique().shape[0] * merged_df['hispanic_prop'].unique().shape[0] * merged_df['white_prop'].unique().shape[0] * merged_df['asian_prop'].unique().shape[0]

    merged_df.to_csv(save_dir, index=False)

    return merged_df, incentive_context_total

def get_prop_adopted_after_incentive(zipcode, incentive_offered):
    model = SolarAdoptionModelZipCode(zipcode)
    model.generate_agents()
    model.set_incentive_offered_function(incentive_offered)
    model.apply_incentive_to_agents()
    return model.percent_agents_adopting()


def get_prop_adopted_per_incentive_context(context_mapping_df, incentive_offered):
    for _, row in context_mapping_df.iterrows():
        prop_adopted = get_prop_adopted_after_incentive(row['zip'], incentive_offered)
        context_mapping_df.loc[_, 'prop_adopted'] = prop_adopted
    return context_mapping_df
