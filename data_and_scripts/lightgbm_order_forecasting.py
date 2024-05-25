""" Forecasting demand is important in many practial applications including food, retail, energy and finance. The goal of this project is to predict how many food items (num_orders) will be ordered from different restaurant centers (center_id) locations serving different types of meals (meal_id). The objective is to predict the number of orders (num_orders) for the next 10 time-steps (week) minimizing the total root-mean-squared-error (RMSE). Thanks to Analytics Vidhya for providing this dataset. More information can be found here: https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/ """

import sys
import numpy as np  
import pandas as pd
import ml_vis_eda 
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import pickle
import lightgbm as lgb
pd.set_option('display.max_columns', None)  # displays all columns (wrap-around) in pandas statistics .info(), .corr(), .describe() etc

# files needed for this dataset
df_name = 'foodDemand_train/train.csv'  # training data filename
df_test_name = 'foodDemand_train/food_Demand_test.csv'  # features for the data to predict  
df_sample_name = 'foodDemand_train/sample_submission.csv'  # example (feature predictions)
Additional_merge_dfs = {'foodDemand_train/fulfilment_center_info.csv': 'center_id', 'foodDemand_train/meal_info.csv': 'meal_id'}

""" run mode is used to validate the entire code right before any data manipulation.
Normally to submit predictions to kaggle we use run_mode=0 which uses the entire train data file and predicts using the test file.
Alternatively, we can first use run_mode=1 to use the last known data points in the train file and separate it into two virtual train and test files. Then using run_mode=2 we can test how well the algorithm performs on the last known data. """
run_mode = 0  # How to run this program 
# run_mode = 0: Use base file-names above and write final predictions for submission to Kaggle and normal predictions
# run_mode = 1: use base file-names above and write new files for validation with 'file_name_ext' + '.csv' appended to them
# run_mode = 2: use new files (from run_mode=1) and run end-to-end, then calculate final predictions with validation
test_time_len = None  # for run_mode=1 only: number of time-steps to use for predictions (None: same number of time-steps as run-mode=0)
file_name_ext = '_virtual'  # string to append to validateion base filenames (run_mode = 1 and 2 only)

# some high-level hyperparameters
algorithms = ['mean value', 'LightGBM']
default_algorithm = 'LightGBM' # 'mean value'
plot_data = True # plot data for visualization
plot_ts = 5 # number of individual time-series with targets, predictions and confidence intervals to plot
lgb_model_str = 'lightGBM_opt_pickles/lgb_model'

""" variable names (depends on the dataset). Categorical variables have more than two categories."""
target_feature = 'num_orders'  # this is what we will fit and predict 
categorical_features = ['center_id', 'meal_id', 'category', 'cuisine', 'city_code', 'region_code', 'center_type']
t_var = 'week'  # unit of 'time' (the column name in df)
id_var = 'id'  # common column to identify submission data and training data
unique_cols = ['center_id', 'meal_id']  # required columns to identify time-series uniquely 
quantile_alphas = [0.05, 0.5, 0.95]  # predict quantiles for the predictions (used for plotting confidence intervals)

# feature search hyperparameters -- since orders is a non-negative integer (count-like) we use 'poisson regression' for the objective. 
# Note that during validation, number of boosters (stopping rounds) is determined by L2 (RMSE) loss which is what we want to minimize
nfold = 5  # number of cross-validation folds
use_important_features = 4  # start with this many current features
param_vals = {'num_leaves':None, 'learning_rate':0.05, 'max_depth':None, 'min_child_samples':40, 'objective': 'poisson',
        'metric':['l2', 'poisson'], 'early_stopping_round':1000, 'num_iterations':10000, 'verbose': -1,
        'min_split_gain': 0., 'min_child_weight': 1e-3, 'reg_alpha': 0., 'reg_lambda': 0.,
        'subsample': 1.0, 'subsample_freq': 10, 'boosting_type': 'gbdt', 'first_metric_only': True} 

# Sequence of steps in finding feature importances/relevance, feature engineering, hyperparameter optimization, and saving results
# Each step depends on the one before it. It is a good idea to check the results after each step before proceeding to the next one.
# Edit each step as necessary before running and check results after running it.
# Set the step you are working on to 'True' to run and test the results of that step.
find_relevant_raw_features = False  # find relevant raw features
find_relevant_eng_features = False  # feature engineering / find relevant features --> EDA used to decide on features 
do_recurrent_opt = False  # do recurrent feature selection if files do not exist
do_recurrent_opt_force = False  # always overwrite existing files
write_new_model = False  # this creates the final model with the recurrent features
write_new_data = False  # this runs the final model to generate the data predictions 

# optional parameters or hyperparameter optimization in some of the above steps
find_recurrent_features = True  # feature engineering: temporally lagged features 
use_average_target_properties = False  # use temporal average statistics (keep False because it did not help CV scores improve)
do_lr_opt = False  # optimize learning rate for gradient boosting
do_pars_opt = False  # optimize hyperparameters for gradient boosting 
test_recurrent = True  # check if recurrent features improve test and CV results for each time step

# functions to create new file-names for end-to-end testing predictions (using run-mode=1, 2)
def new_file_name(fname: str, ext_in: str, ext_out: str) -> str:
    return fname.split(ext_in)[0] + ext_out + ext_in
def new_file_names(fnames: 'list(str)', ext_in: str, ext_out: str) -> 'list(str)':
    return [new_file_name(fname, ext_in, ext_out) for fname in fnames]
if run_mode == 2:  # get virtual file names to read
    df_name, df_test_name, df_sample_name = new_file_names(
        [df_name, df_test_name, df_sample_name], '.csv', file_name_ext)

# read in the data 
df = ml_vis_eda.pd_read_csv_stats_describe(df_name)

# read in the related files
df_test = pd.read_csv(df_test_name)
df_sample = pd.read_csv(df_sample_name)
df_predictions = df_sample.copy()
df_predictions[target_feature] = np.nan  # to make sure all get filled in
assert all(df_test[id_var].sort_values() == df_sample[id_var].sort_values())
assert df_test[t_var].min() > df[t_var].max() # check that the data to predict happens after all the valid data
if run_mode == 2:
    lgb_model_str = lgb_model_str + file_name_ext

# write virtual files if run_mode=1, and then exit
if run_mode == 1: # overwrite df and df_test data to test the whole code end-to-end
    end_time = df[t_var].max()  # last valid sample for testing
    if test_time_len is None:
        test_time_len = df_test[t_var].max() - end_time
    test_time = end_time - test_time_len
    # modify the dataframes 
    df_test = df[df[t_var] > test_time]
    df_sample = df_test[[id_var, target_feature]]
    df = df[df[t_var] <= test_time]
    df_name, df_test_name, df_sample_name = new_file_names(
        [df_name, df_test_name, df_sample_name], '.csv', file_name_ext)
    # write the modified csvs
    df.to_csv(df_name, index=False)
    df_test[target_feature] = np.nan
    df_test.to_csv(df_test_name, index=False)
    df_sample.to_csv(df_sample_name, index=False)
    sys.exit(f'Files written: {df_name}, {df_test_name}, {df_sample_name}. Test this script using these files by running it again with run_mode=2')

########## set up the time intervals that define the training+validation, test and kaggle data ##########
start_time = df[t_var].min()  # first sample
end_time = df_test[t_var].max()  # last sample to predict
test_time = df[t_var].max()  # last valid sample
tstep = end_time - test_time
df = pd.concat((df, df_test), axis=0)  # avoids errors later if manipulating df_test and df_tree differently
train_time = test_time - tstep

# plot histograms of all features
plt.close('all')  # close all open figures
if plot_data:  
    ml_vis_eda.plot_multiple(df.values, df.columns, targets=df[target_feature], target_label=target_feature, 
        nbins=40, no_data=np.nan, range_targets=[0., 1500.], cmap=None)

# merge the columns from all of the datasets to see if there is additional information that can help the model more accurately 
for merge_df, merge_col in Additional_merge_dfs.items():
    df_to_merge = ml_vis_eda.pd_read_csv_stats_describe(merge_df)
    df = df.merge(df_to_merge, on=merge_col)

# make the last column the target we want to predict
df_targets = df.pop(target_feature)
df = df.join(df_targets)
df.head()

# sort the data so that adjacent rows represent the same time-series location with increasing time
df = df.sort_values(by=[*tuple(unique_cols), t_var])

def apply_unique_ts_map(dvalues, tvalues, train_time, test_time):
    """ 
    identify individual time-series
    return row index indicators (start, end train set, end test set, end predition set) for each time-series

    :param dvalues: (int numpy array) shape = (nrows, *) each row uniquely identifies time-series 
    :param tvalues: (int numpy array) shape = (nrows,) each row represents time
    :param train_time: (int) latest time for training data 
    :param test_time: (int) latest time for test data

    return
        ts_inds_inv: [list] of size==(num_rows,) time-series index each row in df belongs to (0, 1, 2, ...
        ts_inds: [int numpy array] shape=(num_ts, 4), for each time-series (row index), columns represent row indices of input
            start time, end train time, end test time, end time
    """
    nrows = dvalues.shape[0]
    ts_inds_inv = np.zeros((nrows,), dtype=int)
    last_cur = tuple(dvalues[0])
    ts_inds = [[0] * 4]
    cur_ts = 0
    def get_t_status(tvalue):
        if tvalue > test_time:
            return 2
        elif tvalue > train_time:
            return 1
        return 0
    for ind in range(nrows):
        cur = tuple(dvalues[ind])
        tvalue = tvalues[ind]
        ts_status = get_t_status(tvalue)
        if cur != last_cur:
            ts_inds[-1][3] = ind
            ts_inds.append([ind] * 4)
            cur_ts += 1
            last_cur = cur
        ts_inds[-1][ts_status+1:] = [ind+1] * (3-ts_status)
        ts_inds_inv[ind] = cur_ts
    return np.array(ts_inds), ts_inds_inv, 
ts_inds, ts_inds_inv = apply_unique_ts_map(df[unique_cols].values, df[t_var].values, train_time, test_time)
num_ts = ts_inds.shape[0]  # the number of timeseries in the dataframe

# We convert all category objects to natural number values for simpler computational logic, e.g. [0, 1, 2, ..., C]
def apply_unique_cats(df, categorical_features):
    """
    Convert a Pandas dataframe's categorical features to non-negative integer labeling, provide the feature mapping
    
    :param df: [pandas.core.frame.DataFrame] shape = (num_rows, num_columns)
    :param categorical_features: [list of str] features to modify 
    return
        df_copy: [pandas.core.frame.DataFrame] shape = (num_rows, num_columns) -- 
            copy of input df with non-negative integer labeling for all categorical_features
        cat_codes: [dict] keys gives each feature as str, values are original ordered df values
    """
    cat_codes = {}
    df_copy = df.copy()
    for cat_feature in categorical_features:
        df_copy[cat_feature], cat_values = df_copy[cat_feature].factorize(sort=True)
        cat_codes[cat_feature] = cat_values
    return df_copy, cat_codes
df_copy, cat_codes = apply_unique_cats(df, categorical_features)
# this is the one-to-one mapping for categorical features between df and df_copy 
# print(cat_codes)

if plot_data:    
    """
    To observe how the feature distributions are related to the target variable (num_orders) we plot their frequency histograms.
    The heights of the bars 'num_counts' represents the frequency of occurence for a given feature for the entire dataset.
    The color of each bar represents the average value of (target_label) at a given feature value
    """
    ml_vis_eda.plot_multiple(df_copy[df_copy.columns].values, df_copy.columns, targets=df_copy[target_feature], target_label=target_feature, nbins=30, no_data=-1, range_targets=[0., 1500.], cmap=mpl.cm.cool)
    # The color variation indicates that target_feature depends strongly on 'emailer_for_promotion', 'homepage_featured', 'base_price', 'checkout_price' and 'op_area'
    
    # The correlation between all features is plotted below using 'spearman ranking' correlation coefficient which is reasonable for all variables that can ordered including real-valued, ranking data, and binary categorical features
    ordered_features = ['week', 'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'op_area', 'num_orders']
    ml_vis_eda.plot_corr(df_copy[ordered_features], method='spearman')  # 'pearson; method is best used for linearly related numerical features
    # The correlation heatmap indicates that 'checkout_price' and 'base_price' are strongly correlated (R^2=0.96) meaning that together they do not add much new information. 
    
    # Based on the heatmap correlations, 'checkout_price' is plotted below against all other features in scatter plots.
    # These show that the targets (num_orders) depends strongly on 'checkout price' for some meals but not all based on 'meal_id'.
    ml_vis_eda.plot_multiple(df_copy[df_copy.columns].values, df_copy.columns, ylabel='checkout_price', targets=df_copy[target_feature], 
            target_label=target_feature, nbins=30, no_data=-1, range_targets=[0., 1500.], cmap=mpl.cm.cool)


def get_val_time(do_test, train_time, test_time):
    """ 'do_test=True' is for validation instead of predicting unknown future data """
    return train_time if do_test else test_time

def create_gbm_data_and_params(do_test, train_time, test_time, df_tree, t_var, tree_features, target_feature, num_step=0):
    """ find train and test indices of pandas time-series data so that 'test_inds' occurs 'num_step' after 'train_inds' """
    val_time = get_val_time(do_test, train_time, test_time)
    train_inds = df_tree[t_var] <= val_time - num_step
    test_inds = (df_tree[t_var] > val_time) & (df_tree[t_var] <= test_time)
    df_X = df_tree[tree_features]
    df_y = df_tree[target_feature]
    return df_X, df_y, train_inds, test_inds

def get_gbm_params(do_test, **param_vals):
    params = param_vals.copy()
    if not do_test:
        params['early_stopping_round'] = None
    return params

def train_val_lightGBM(cur_features, train_time, test_time, df_tree, t_var, 
        target_feature, categorical_features, param_vals, num_step=0, do_test=True, n_estimators=None, alpha=None):
    """ train LightGBM model with validation data to find best number of boosting rounds (trees) with gbm returned object """
    params = get_gbm_params(do_test, **param_vals)
    df_X, df_y, train_inds2, test_inds2 = create_gbm_data_and_params(do_test, train_time, test_time, df_tree, t_var, 
        cur_features, target_feature, num_step) 
    categorical_features_cur = [c for c in categorical_features if c in cur_features]
    X_train2, X_test2, y_train2, y_test2 = df_X[train_inds2], df_X[test_inds2], df_y[train_inds2], df_y[test_inds2]
    if alpha is not None:
        params['objective'] = 'quantile'
        params['metric'] = ['l2', 'quantile']
    if do_test:
        eval_set=(X_test2, y_test2)
    else:
        eval_set=None
        params['num_iterations'] = n_estimators
        params['early_stopping_round'] = None
        cur_score_test = None
    gbm = lgb.LGBMRegressor(alpha=alpha, **params)
    gbm.fit(X_train2, y_train2, eval_set=eval_set, feature_name=cur_features, 
        categorical_feature=categorical_features_cur)
    cur_score_train = gbm.score(X_train2, y_train2)
    if do_test:
        cur_score_test = gbm.score(X_test2, y_test2)
        print(f"lightgbm test score (R2): {cur_score_test}, number of boosting rounds: {gbm.n_estimators_}")
        print(f"lightgbm train score (R2): {cur_score_train}")
    else:
        print(f"lightgbm final train score (R2): {cur_score_train}, number of boosting rounds: {gbm.n_estimators_}")
    return {'model':gbm, 
            'score_test':cur_score_test, 
            'score_train':cur_score_train}

def get_tree_features(param_recurrent, df_tree, best_features, unique_cols, t_var):
    tstep = test_time - train_time
    if param_recurrent is not None:
        istep, t_min, t_max, window_pow, observed_cols, known_cols, observed_cols_stats = param_recurrent
        df_tree_t, df_tree_t_features = modify_tree_recurrent(df_tree, observed_cols, known_cols, observed_cols_stats, unique_cols,
            t_var, t_min, t_max, istep, tstep, window_pow, best_features)
    else:
        df_tree_t, df_tree_t_features, istep = df_tree, best_features, None

    return df_tree_t, df_tree_t_features, tstep, istep


def val_and_finalize_lightGBM(train_time, test_time, df_tree, t_var, lgb_model_str,
        best_features, param_recurrent, target_feature, categorical_features, param_vals, alphas):
    """ Validate GBM model to optimize boosting rounds and then train final model """
    # create the features on the fly for this istep 
    df_tree_t, df_tree_t_features, tstep, istep = get_tree_features(param_recurrent, df_tree, best_features, unique_cols, t_var)

    # features needed for the current iteration
    print(f'features for step {istep} to predict {target_feature}: {df_tree_t_features}')
        
    # train with validation rounds using train/test split
    for alpha in alphas:
        num_step = -1-istep if istep is not None else 0
        val_dict = train_val_lightGBM(df_tree_t_features, train_time, test_time, df_tree_t, t_var, 
            target_feature, categorical_features, param_vals, num_step, True, alpha=alpha)
        gbm_val = val_dict['model']
    
        # evaluate final test model for future prediction based on validation rounds with all test data
        if num_step != 0:
            test_dict = train_val_lightGBM(df_tree_t_features, train_time, train_time, df_tree_t, t_var,
                target_feature, categorical_features, param_vals, 0, False, 
                int(gbm_val.n_estimators_ * train_time / (train_time - num_step)), alpha=alpha)
        else:
            test_dict = val_dict

        # evaluate final model for future prediction based on validation rounds with all data
        final_dict = train_val_lightGBM(df_tree_t_features, train_time, test_time, df_tree_t, t_var,
            target_feature, categorical_features, param_vals, 0, False, 
            int(gbm_val.n_estimators_ * test_time / (train_time - num_step)), alpha=alpha)

        # save models
        num_step = -1-istep if istep is not None else None
        lgb_model_str_test = get_pickle_file_name(lgb_model_str, True, num_step, alpha)
        lgb_model_str_final = get_pickle_file_name(lgb_model_str, False, num_step, alpha)
        save_model_pickle(lgb_model_str_test, test_dict)
        save_model_pickle(lgb_model_str_final, final_dict)
        print(f'files written: {lgb_model_str_test} and {lgb_model_str_final}')


def get_cv_score_lightGBM(tree_features, do_test, train_time, test_time, df_tree, t_var, target_feature,
        categorical_features, param_vals, nfold, tstep, num_step=0):
    """ LightGBM CV fitting using training data with nfolds """
    # Create LightGBM train and test set
    if do_test:
        val_time = train_time - num_step
    else:
        val_time = test_time
    train_inds = df_tree[t_var] <= val_time
    df_X = df_tree[tree_features]
    df_y = df_tree[target_feature]
    X_train, y_train = df_X[train_inds], df_y[train_inds]
    ntrain = len(y_train)
    categorical_features_tree = [c for c in categorical_features if c in tree_features]
    t_vals = df_tree.loc[train_inds, t_var].values
    ts_cv_fold_causal = ts_CV_fold_causal(nfold, t_vals, tstep, val_time, num_step)
    params = param_vals.copy()

    # creating LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False,
        feature_name=tree_features, categorical_feature=categorical_features_tree)
    
    # do kfold cross-validation
    cv_results = lgb.cv(params, train_data, folds=ts_cv_fold_causal, stratified=False, return_cvbooster=True)

    # get the R2 total score (not average fold score as returned by lgb.cv)
    ts_cv_fold_causal = ts_CV_fold_causal(nfold, t_vals, tstep, val_time, num_step)  # reinitialize iterator object
    cv_preds = []
    cv_targets = []
    for fold in range(nfold):
        inds_train, inds_val = ts_cv_fold_causal.__next__()
        cv_preds.extend(cv_results['cvbooster'].boosters[fold].predict(X_train.iloc[inds_val, :]))
        cv_targets.extend(y_train.iloc[inds_val])
    cv_preds = np.array(cv_preds)
    cv_targets = np.array(cv_targets)
    y_var = np.var(cv_targets)
    cv_score = 1. - np.var(cv_preds - cv_targets) / y_var  # this is the score we want
    n_estimators = int(len(cv_results['valid l2-mean']))
    cv_score_mean = 1. - cv_results['valid l2-mean'][-1] / y_var  # this is R**2 value
    cv_score_std = cv_results['valid l2-stdv'][-1] / y_var / np.sqrt(nfold - 1.)  # error in the mean
    print(f'final LightGBM CV score: {cv_score} +/- {cv_score_std} with {n_estimators} boosting rounds')
    print(f'mean CV score (not used): {cv_score_mean}')
    
    # setting up LightGBM training parameters with optimal n_estimators (from CV stopping rounds)
    if not do_test:
        n_estimators = int(n_estimators * ntrain / (ntrain - 0.5*len(cv_targets)))
    params['num_iterations'] = n_estimators
    params['early_stopping_round'] = None  # for predictions we have no validation sets!

    # training LightGBM and obtain final train/test scores
    bst = lgb.train(params, train_data)
    feature_importances = bst.feature_importance()
    preds_train = bst.predict(X_train)
    score_train = 1. - np.var(preds_train - y_train) / np.var(y_train)
    print(f'final LightGBM train score: {score_train} with {n_estimators} boosting rounds')
    if do_test:
        test_inds = (df_tree[t_var] > train_time) & (df_tree[t_var] <= test_time)
        X_test, y_test = df_X[test_inds], df_y[test_inds]
        preds_test = bst.predict(X_test)
        score_test = 1. - np.var(preds_test - y_test) / np.var(y_test)
        print(f'final LightGBM test score: {score_test}')
    else:  # we can't get the test score since we don't have the targets
        score_test = None
    return {'feature_importances': feature_importances,
            'score': cv_score,
            'score_std': cv_score_std,
            'score_train': score_train,
            'score_test': score_test,
            'n_estimators': n_estimators}


def get_remove_set(feature_ind, feature_inds, selectable_features):
    """ find the set that is not in """
    set_A = set(selectable_features[feature_ind])  # 
    sets_U = [set(selectable_features[feature_i]) for feature_i in feature_inds if feature_i != feature_ind]
    set_Y = set_A.copy()
    for set_U in sets_U:
        if set_A < set_U:
            set_Y = set_Y | set_U
    for set_U in sets_U:
        if (set_U & set_A) != set_A:
            set_Y = set_Y - set_U
    return set_Y

def add_remove_best_feature(cur_score_cv, cur_features, selectable_features, obj_func,
        obj_params, obj_params_dict, feature_importances=None, num_stds_cv=0., remove=False):
    """ do kfold cross validation and test the feature that beats and gives the best new CV score"""
    cur_feature_inds = np.array([any(x not in cur_features for x in xs) for xs in selectable_features])
    feature_inds = np.where(cur_feature_inds ^ remove)[0]
    if feature_importances is not None:
        sort_inds = np.argsort((2. * remove - 1.) * feature_importances[feature_inds])
        feature_inds = feature_inds[sort_inds]
    else:
        feature_inds = np.random.permutation(feature_inds)
    best_score_cv = cur_score_cv
    best_feature = None
    best_features = cur_features
    best_score_test = None
    for feature_ind in feature_inds:
        if remove:
            cur_feature = get_remove_set(feature_ind, feature_inds, selectable_features)
            test_features = list(set(cur_features) - cur_feature)  # set difference 
            teststr = 'removing'
        else:
            cur_feature = [x for x in selectable_features[feature_ind]]
            test_features = list(set(cur_features) | set(cur_feature))  # set union
            teststr = 'adding'
        print(f'\n{teststr} feature {cur_feature}')
        print(f'Current CV score: {cur_score_cv}, with features: {cur_features}')
        cv_dict = obj_func(test_features, *obj_params, **obj_params_dict)
        score_cv, score_cv_std, score_test = cv_dict['score'], cv_dict['score_std'], cv_dict['score_test']
        if (score_cv > best_score_cv) and score_cv > cur_score_cv + num_stds_cv * score_cv_std:
            best_feature = cur_feature
            best_score_cv = score_cv
            best_features = test_features
            best_score_test = score_test
            print(f'New best CV score: {best_score_cv} using {best_features}')
            if feature_importances is not None:
                break
    return best_score_cv, best_feature, best_score_test, best_features

def print_plot_importance(feature_importances, tree_features):
    """ Plot feature importance for lightGBM fitted data """
    (pd.Series(feature_importances, index=tree_features).plot(kind='barh', figsize=(12, 12)))
    plt.title("Feature Importance")
    plt.show()
    print(f'\n Feature importances:')
    for x in zip(tree_features, feature_importances):
        print(f'{x[0]}: {x[1]}')

class ts_CV_fold_causal:
    """ 
    iterator class to create cv folds (train_inds, test_inds) for time-series data with causal features 
    (For pandas time-series data, e.g. not independent identically distributed (IID) data for each row)
    This means that features may contain information about past values of the targets (e.g. recurrent features)
    """
    def __init__(self, nfold, tvals, tstep, max_t, num_step=0):
        self.nfold = nfold  # number of folds (for kfold validations)
        self.tvals = tvals  # time values
        self.index = 0  
        self.max_t = max_t  # largest time value to use
        self.tstep = tstep  # number of time-steps (time length) to use for each test set 
        self.num_step = num_step  # number of time-steps (time length) to predict into the future (use with lagged columns)
    def __iter__(self):
        return self
    def __next__(self):
        if self.index < self.nfold:
            start_t = self.max_t - (self.nfold - self.index) * self.tstep
            end_t = start_t + self.tstep
            train_vals = self.tvals <= start_t - self.num_step  # past data for train set
            test_vals = (self.tvals > start_t) & (self.tvals <= end_t)  # data to predict (test set)
            train = list(np.where(train_vals)[0])
            test = list(np.where(test_vals)[0])
            self.index += 1
            return (train, test)
        else:
            raise StopIteration

# outlier analysis
df_tree = df_copy.copy()
plot_cols = ['checkout_price', 'base_price', 'num_orders']
for plot_col in plot_cols:
    df_tree[plot_col] = np.log(df_tree[plot_col])
tree_features = list(df_tree.columns)
if plot_data:  # more exploratory data analysis
    ml_vis_eda.pd_read_csv_stats_describe(None, df_tree[tree_features])
    # The color of each bar represents the average value of (target_label) at a given feature value
    ml_vis_eda.plot_multiple(df_tree[tree_features].values, tree_features, targets=df_tree[target_feature], target_label=target_feature, nbins=30, no_data=-1, range_targets=[df_tree[target_feature].min(), df_tree[target_feature].max()], cmap=mpl.cm.cool)
    # The correlation between all features is plotted below using 'spearman ranking' correlation coefficient which is reasonable for all variables that can ordered including real-valued, ranking data, and binary categorical features
    ml_vis_eda.plot_corr(df_tree[ordered_features], method='spearman')  # 'pearson; method is best used for linearly related numerical features
    # These show that the targets (num_orders) depends strongly on 'checkout price' for some meals but not all based on 'meal_id'.
    ml_vis_eda.plot_multiple(df_tree[tree_features].values, tree_features, ylabel='checkout_price', targets=df_tree[target_feature], 
            target_label=target_feature, nbins=30, no_data=-1, range_targets=[df_tree[target_feature].min(), df_tree[target_feature].max()], cmap=mpl.cm.cool)


######## functions for saving/loading results from each step 
def save_model_pickle(filename, model):
    """ use Python pickle to save file to disk for later usage """
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model_pickle(filename):
    """ use Python pickle to load file from disk """ 
    with open(filename, 'rb') as file:
        return pickle.load(file)

def get_pickle_file_name(lgb_model_str, do_test, num_step=None, alpha=None):
    if alpha is not None:
        lgb_model_str = lgb_model_str + '_alpha' + str(alpha)
    if do_test:
        lgb_model_str = lgb_model_str + '_' + 'train_stage'
    else:
        lgb_model_str = lgb_model_str + '_' + 'final_stage'
    if num_step is not None:
        lgb_model_str = lgb_model_str + '_' + str(num_step+1)
    return lgb_model_str + '.pkl'


######## Tree-based algorithm #################
# We will do some feature engineering 
def add_merged_feature(df_tree, columns, merge_col_name, mean_col_name=None, count_col_name=None, 
        median_col_name=None, sum_col_name=None, any_col_name=None):
    if mean_col_name is not None:  # gets mean by group
        grpby_df = df_tree.groupby(columns)[mean_col_name].mean().reset_index()
    elif count_col_name is not None:  # gets valid count by group
        grpby_df = df_tree.groupby(columns)[count_col_name].count().reset_index()
    elif median_col_name is not None:
        grpby_df = df_tree.groupby(columns)[median_col_name].median().reset_index()
    elif sum_col_name is not None:
        grpby_df = df_tree.groupby(columns)[sum_col_name].sum().reset_index()
    elif any_col_name is not None:
        grpby_df = df_tree.groupby(columns)[any_col_name].any().reset_index()
    columns2 = list.copy(columns)
    columns2.append(merge_col_name)
    grpby_df.columns= columns2
    df_tree = df_tree.merge(grpby_df, on=columns, how='left')
    return df_tree

def unique_list(xss):
    return list(set([x for xs in xss for x in xs]))

class Sfs_FW:
    """ 
    Perform sequential feature selection (SFS) in successively alternating forward and backward steps
    using greedy approach until convergence to the highest score

    obj_func should return a score to be maximized (first output tuple parameter)
    
    ----- Expected signature of obj_func -----
    def obj_func(features, *obj_params, **obj_params_dict):
        ...
        return dictionary with required keys: 'score', 'feature_importances'

    ---- After each sfs step test_func is evaluated with expected signature ----
    def test_func(features, *test_params, **test_params_dict):
        ...
        return dictionary with required keys: 'score'
        
    """
    def __init__(self, obj_func, obj_params, obj_params_dict, cur_features=None, selectable_features=[],
            use_important_features=None, test_func=None, test_params=[], test_params_dict={}, remove=False):
        self.obj_func = obj_func
        self.obj_params = obj_params
        self.obj_params_dict = obj_params_dict
        self.cur_features = cur_features
        self.selectable_features = selectable_features
        self.tree_features = unique_list(selectable_features)
        self.use_important_features = use_important_features
        self.feature_importances = None
        self.test_func = test_func
        self.test_params = test_params
        self.test_params_dict = test_params_dict
        self.remove = remove  
        # outputs from each sfs step
        self.best_features=None  # set of features found
        self.best_scores = []  # score
        self.best_add_removes = []  # selectable_feature added/removed
        self.removes = []  # True/False indicates sfs step is a feature removal (or addition) step
        self.best_scores_test = []  # outputs from running optional sfs_func (e.g. test score)
        self.cur_score = -np.inf
        self.test_score = -np.inf

    def get_feature_importances_(self, feature_importances_tree):
        print_plot_importance(feature_importances_tree, self.tree_features)
        if self.cur_features is None:
            self.cur_features = [self.tree_features[ind] for ind in np.argsort(
                -feature_importances_tree)[:self.use_important_features]]
    
        # get feature importances in order based on order of selectable_features
        feature_importances = np.zeros(len(self.selectable_features))
        for ind, features in enumerate(self.selectable_features):
            feature_importances[ind] = feature_importances_tree[self.tree_features.index(features[0])]
        self.feature_importances = feature_importances
        print(f'\n Feature groups importances:')
        for x in zip(self.feature_importances, self.selectable_features):
            print(f'{x[0]}: {x[1]}')
            
    def get_feature_importances(self):
        """ 
        find most important features and populate cur_features with use_important_features best features if not populated
        using one booster run with 'feature_importances' and all available selectable_features 
        """
        all_dict = self.obj_func(self.tree_features, *self.obj_params, **self.obj_params_dict)
        feature_importances_tree = all_dict['feature_importances']
        all_score = all_dict['score']
        if self.test_func is not None:
            test_dict = self.test_func(self.tree_features, *self.test_params, **self.test_params_dict)
        self.get_feature_importances_(feature_importances_tree)

    def get_best(self, keep_n):
        """ Perform one additive feature selection step and rank all the selectable_features (feature_importances) using the CV score """
        self.get_cur_score()
        all_scores = np.ones(len(self.selectable_features)) * self.cur_score
        if self.test_func is not None:
            all_test_scores = np.ones(len(self.selectable_features)) * self.test_score
        for ind, features in enumerate(self.selectable_features):
            cur_features = self.cur_features.copy() if self.cur_features is not None else []
            add_features = []
            for feature in features:
                if feature not in cur_features:
                    add_features.append(feature)
            if len(add_features) > 0:
                print(f'Adding features: {add_features}')
                cur_features.extend(add_features)
                all_scores[ind] = self.obj_func(cur_features, *self.obj_params, **self.obj_params_dict)['score']
                if self.test_func is not None: 
                    all_test_scores[ind] = self.test_func(cur_features, *self.test_params, **self.test_params_dict)['score_test']
            print(f'{cur_features}: {all_scores[ind]}, ({add_features})')
        self.inds_sorted = np.argsort(-all_scores)
        self.all_scores = all_scores[self.inds_sorted]
        self.all_test_scores = all_test_scores[self.inds_sorted]
        selectable_features_sorted = [self.selectable_features[inds_sort] for inds_sort in self.inds_sorted]
        print('\n sorted cv scores and test scores for features added:')
        selectable_features_n = []
        ind = 0
        for (selectable_features, cv_score, test_score) in zip(selectable_features_sorted, 
            self.all_scores, self.all_test_scores):
            print(f'{selectable_features}, cv score: {cv_score:2.4f}, test score: {test_score:2.4f}')
            if not all([selectable_feature in self.cur_features for selectable_feature in selectable_features]):
                if ind == 0:  # just add the best found features now!
                    self.cur_features = list(set(self.cur_features) | set(selectable_features))  # set union
                ind += 1
            selectable_features_n.extend(selectable_features)
            if ind >= keep_n:
                break
        selectable_features_n = list(set(self.cur_features) | set(selectable_features_n))
        self.tree_features = list(set(selectable_features_n))
        keep_selectable = []
        for selectable_features in self.selectable_features:
            if all([selectable_feature in self.tree_features for selectable_feature in selectable_features]):
                keep_selectable.append(selectable_features)
        self.selectable_features = keep_selectable

    
    def get_cur_score(self):
        
        # CV using current features
        if self.cur_features is not None:
            print(f'Starting features that have a significant effect on {target_feature} based on initial run using feature importances: {self.cur_features}')
            start_dict = self.obj_func(self.cur_features, *self.obj_params, **self.obj_params_dict)
            self.cur_score = start_dict['score']

            # # train with validation rounds using current features
            if self.test_func is not None:
                test_dict = self.test_func(self.cur_features, *self.test_params, **self.test_params_dict)
                self.test_score = test_dict['score_test']
        

    def optimize(self, start_optimizer=True):
        if start_optimizer:
            if self.use_important_features and self.cur_features is None:
                self.get_feature_importances()

            # Start with all raw features. Now we will decide on the first few features to select based on feature_importances
            self.get_cur_score()
            self.best_scores = [self.cur_score]
            if self.test_func is not None:
                self.best_scores_test.append(self.test_score)

            self.best_features = self.cur_features.copy()
        not_found = 0
        while not_found < 2:
            # add or remove the best feature to the current features (if any) that improves the CV score the most
            best_score_cv, best_add_remove, best_score_test, self.best_features = add_remove_best_feature(self.best_scores[-1], 
                    self.best_features, self.selectable_features, self.obj_func, self.obj_params, self.obj_params_dict, 
                    feature_importances=self.feature_importances, num_stds_cv=0., remove=self.remove)
            if best_add_remove is not None:
                self.removes.append(self.remove)
                self.best_scores.append(best_score_cv)
                self.best_add_removes.append(best_add_remove)
                if self.test_func is not None:
                    test_dict = self.test_func(self.best_features, *self.test_params, **self.test_params_dict)
                    self.best_scores_test.append(test_dict['score_test'])
                not_found = 0
            else:
                not_found += 1
            self.remove = not self.remove

        # print results
        print(f'best_scores_test: {self.best_scores_test}')
        print(f'best_scores_cv: {self.best_scores}')
        print(f'best_test_features: {self.best_add_removes}')
        print(f'removed/added: {self.removes}')
        print(f'best_features: {self.best_features}')
        print(f'cur_features: {self.cur_features}')


#### steps: find_relevant_raw_features, find_relevant_eng_features 
df_tree = df_copy.copy()
relevant_raw_features_fname = lgb_model_str + '_relevant_raw_features.pkl'
relevant_eng_features_fname = lgb_model_str + '_relevant_eng_features.pkl'
best_features = None  # initialize best features
for ind_relevant, find_relevant in enumerate([find_relevant_raw_features, find_relevant_eng_features]):
    # the features common in each sublist below must appear together during feature selection
    selectable_features = [['center_id'], ['meal_id'], ['checkout_price'], ['base_price'], ['emailer_for_promotion'], 
            ['homepage_featured'], ['city_code'], ['region_code'], ['center_type'], ['op_area'], ['category'], ['cuisine']]
    
    if ind_relevant == 1:
        selectable_features.append([t_var])
        # get temporal feature mean -- these inputs are known for future predictions so we can put it here
        eng_cat_features = ['base_price', 'checkout_price', 'emailer_for_promotion','homepage_featured']
        for eng_cat_feature in eng_cat_features:
            grp = df_tree.sort_values(t_var).groupby(['center_id', 'meal_id'])[eng_cat_feature]
            feature_name = eng_cat_feature + '_mean_ts'
            df_tree[feature_name] = grp.transform(lambda x: x.expanding().mean())
            selectable_features.append([feature_name, eng_cat_feature])
        
        # add some engineered features 
        eng_cat_features = ['center_id', 'meal_id', 'city_code',
                'region_code', 'center_type', 'category', 'cuisine']
        for eng_cat_feature in eng_cat_features:
            feature_name = eng_cat_feature + '_' + t_var + '_count'
            df_tree = add_merged_feature(df_tree, [eng_cat_feature, t_var], feature_name, count_col_name='id')
            selectable_features.append([feature_name, eng_cat_feature])
#             grp = df_tree.sort_values(t_var).groupby(['center_id', 'meal_id'])[feature_name]
#             df_tree[feature_name + '_d'] = grp.transform(lambda x: x - x.rolling(tstep).mean())
#             selectable_features.append([feature_name + '_d', eng_cat_feature])
        
        # global count for each time-step
        df_tree = add_merged_feature(df_tree, [t_var], t_var + '_count', count_col_name='id')
        feature_name = t_var + '_count'
        selectable_features.append([feature_name])

        feature_name = 'ts_mean'
        grp = df_tree.sort_values(t_var).groupby(['center_id', 'meal_id'])[id_var]
        df_tree[feature_name] = grp.transform(lambda x: x.expanding().count())
        df_tree[feature_name] = df_tree[feature_name] / df_tree[t_var]
        selectable_features.append([feature_name])
#         grp = df_tree.sort_values(t_var).groupby(['center_id', 'meal_id'])[feature_name]
#         df_tree[feature_name + '_d'] = grp.transform(lambda x: x - x.rolling(tstep).mean())
#         selectable_features.append([feature_name + '_d'])
        
        if plot_data:
            plot_cols = ['num_orders', 'checkout_price', 'base_price', 'week_count']
            ml_vis_eda.df_plot_ts(df_tree, t_var, f'time ({t_var})', ycols=plot_cols, ylabels=plot_cols, col_keyvals={}, cat_cols=[],
                    no_data_value=np.nan, show=True, start=0, end=5, max_plots=5)

        # price ratios to mean price grouped by (timestep and categorical features)
        eng_cat_features = ['center_id', 'meal_id', 'city_code', 'region_code', 'center_type', 'category', 'cuisine']
        for eng_cat_feature in eng_cat_features:
            eng_cat_features2 = ['base_price', 'checkout_price']
            for eng_cat_feature2 in eng_cat_features2:
                feature_name = eng_cat_feature + '_' + t_var + '_' + eng_cat_feature2 + '_ratio'
                df_tree = add_merged_feature(df_tree, [eng_cat_feature, t_var], feature_name, mean_col_name=eng_cat_feature2)
                df_tree[feature_name] = df_tree[eng_cat_feature2] / df_tree[feature_name]
                selectable_features.append([feature_name, eng_cat_feature, eng_cat_feature2])
            for eng_cat_feature2 in ['emailer_for_promotion','homepage_featured']:
                feature_name = eng_cat_feature + '_' + eng_cat_feature2 + '_mean'
                df_tree = add_merged_feature(df_tree, [eng_cat_feature, t_var], feature_name, mean_col_name=eng_cat_feature2)
                selectable_features.append([feature_name, eng_cat_feature, eng_cat_feature2])
        
    # Find best relevant features
    if find_relevant:
        get_cv_score_lightGBM_params = False, train_time, test_time, df_tree, t_var, target_feature, categorical_features, param_vals, nfold, tstep
        train_val_lightGBM_params = [train_time, test_time, df_tree, t_var, target_feature, categorical_features, param_vals]
        sfs_fw = Sfs_FW(get_cv_score_lightGBM, get_cv_score_lightGBM_params, {}, cur_features=best_features, 
                selectable_features=selectable_features, use_important_features=use_important_features,
                test_func=train_val_lightGBM, test_params=train_val_lightGBM_params,)
        if ind_relevant == 1:
            keep_n = len(best_features)  # double the possible features
            sfs_fw.get_best(keep_n) 
        sfs_fw.optimize()
        best_features = sfs_fw.best_features
        if ind_relevant == 0:
            save_model_pickle(relevant_raw_features_fname, best_features)
        elif ind_relevant == 1:
            save_model_pickle(relevant_eng_features_fname, best_features)

    # load the last features found (if any)
    if ind_relevant == 0:
        best_features = load_model_pickle(relevant_raw_features_fname)
    elif ind_relevant == 1:
        best_features = load_model_pickle(relevant_eng_features_fname)

# remove columns no longer needed to release memory
cols_keep = [id_var, t_var, target_feature]
cols_keep.extend(best_features.copy())
for col in df_tree.columns:
    if col not in cols_keep:
        df_tree.pop(col)


# try some hyperparameter optimization with optuna
import optuna

def objective(trial):
    param_vals_ = param_vals.copy()

    if do_lr_opt:  # first find the lr 
        param_vals_['learning_rate'] = trial.suggest_float('learning_rate', 2e-2, 5e-1, log=True)
    else:
        param_vals_['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 300, log=True) 
    cv_dict = get_cv_score_lightGBM(best_features, False,
        train_time, test_time, df_tree, t_var, target_feature, categorical_features, param_vals_, nfold, tstep)
    return cv_dict['score']

def plot_opt(study):
    """ 
    plot optuna study -- https://github.com/optuna/optuna-examples/blob/main/visualization/plot_study.py
    """
    from optuna.visualization import plot_optimization_history, plot_contour, plot_parallel_coordinate, plot_param_importances, plot_slice
    plot_optimization_history(study).show()  # show history score vs iteration with best result
    plot_contour(study).show()  # plot score contour estimate
    plot_parallel_coordinate(study).show()
    plot_param_importances(study).show()  # hyperparameter importances
    plot_slice(study).show()

if do_lr_opt:
    study = optuna.create_study(direction='maximize')  # trying to maximize the R**2
    study.optimize(objective, n_trials=50)  # find optimal learning rate
    print(study.best_params)
    plot_opt(study)

    # save the study and load it later
    fname_study = f"{lgb_model_str}_optuna_study_lr.pkl"
    save_model_pickle(fname_study, study)
    study2 = load_model_pickle(fname_study)

# optimize other hyperparameters
if do_pars_opt:
    study_pars = optuna.create_study(direction='maximize')
    study_pars.optimize(objective, n_trials=10)
    plot_opt(study_pars)
    # save the study and load it later
    fname_study_pars = f"{lgb_model_str}_optuna_study_pars.pkl"
    save_model_pickle(fname_study_pars, study_pars)
    study_pars = load_model_pickle(fname_study_pars)
    # set the parameters to be the best ones
    for k,v in study_pars.best_params.items():
        param_vals[k] = v


##### functions to create lags when 'find_recurrent_features = True'
def make_windows(window_pow, max_lag):
    """ return rolling window average length for each time lag """
    windows = []
    for lag in range(1, max_lag+1):
        window = int((lag) ** window_pow)
        windows.append(window)
    return windows

def make_lagged_series(base_colname, t_cur, unique_cols, window, t_var, df_tree):
    """ 
    return a pandas series of values for base_colname in df_tree 
    that is at time variable 't_var', 't_cur' steps into the future and averaged by 'window' into the past
    time-series must be identified uniquely by unique_cols 
    """
    if base_colname == t_var:  # find the actual time-step
        current_time = df_tree.loc[:, t_var]
    grp = df_tree.groupby(unique_cols)[base_colname]
    t_series = grp.transform(lambda x: x.shift(-t_cur).rolling(window, min_periods=1).mean())
    if base_colname == t_var:
        t_series = current_time - t_series
    return t_series

def get_lagged_colname(base_colname, t_cur, window):
    """ return str column name for a lagged feature """
    past_str = 'n' if t_cur < 0 else 'p'
    return base_colname + '_' + str(abs(t_cur)) + past_str + str(window)

def get_lagged_features(base_colname, t_var, t_min, t_max, istep, tstep, window_pow, unique_cols=None, df_tree=None, names_only=False):
    steps, windows = get_lags_windows(t_min, t_max, istep, tstep, window_pow)
    t_colnames = []
    for t_cur, window in zip(steps, windows):
        t_colnames.append(get_lagged_colname(base_colname, t_cur, window))
    if names_only:
        return t_colnames
    else:
        t_df = pd.DataFrame(index=df_tree.index) 
        for (t_colname, t_cur, window) in zip(t_colnames, steps, windows):
            t_df[t_colname] = make_lagged_series(base_colname, t_cur, unique_cols, window, t_var, df_tree)
        return t_df, t_colnames

def get_lags_windows(t_min, t_max, istep, tstep, window_pow):
    """ 
    The point to forecast is at time t = 0
    istep is the time position of the the last 'observed' data: istep <= 0
    the last 'known' data is at t = tstep + istep >= 0
    we start at istep in order to ensure that averaging windows overlap without staggering wrt istep
    """
    def get_window(cur_t, window_pow):
        return int((abs(cur_t) + 1) ** window_pow)
    steps = []
    windows = []
    if t_max > tstep + istep: 
        raise ValueError(f't_max = {t_max} cannot be greater than {tstep + istep}')
    if t_min > t_max:
        raise ValueError(f't_min = {t_min} cannot be greater than t_max={t_max}')
    # get all data before and at istep
    cur_t = istep
    cur_win = get_window(cur_t, window_pow)
    last_t = cur_t
    while cur_t > t_min:
        if cur_t == last_t:
            cur_win = get_window(cur_t, window_pow)
            if cur_t >= t_min and cur_t <= t_max and cur_t != 0: 
                steps.insert(0, cur_t)
                windows.insert(0, cur_win)
            last_t = cur_t - cur_win
        cur_t -= 1
    # get all data after istep (operation is only valid for 'known' inputs)
    cur_t = istep
    cur_win = get_window(cur_t + 1, window_pow)
    last_t = cur_t + cur_win
    while cur_t <= t_max:
        if cur_t == last_t:
            if cur_t >= t_min and cur_t <= t_max and cur_t != 0:
                steps.append(cur_t)
                windows.append(cur_win)
            cur_win = get_window(cur_t + 1, window_pow)
            last_t = cur_t + cur_win
        cur_t += 1
    return steps, windows

def modify_tree_recurrent(df_tree, observed_cols, known_cols, observed_cols_stats, unique_cols,
        t_var, t_min, t_max, istep, tstep, window_pow, best_features):
    df_tree_t = df_tree.copy()
    df_tree_t_columns = best_features.copy()
    for observed_col in observed_cols:
        t_df, t_colnames = get_lagged_features(observed_col, t_var, t_min, istep, istep, tstep, window_pow, 
                unique_cols, df_tree)
        df_tree_t = pd.concat((df_tree_t, t_df), axis=1)
        df_tree_t_columns.extend(t_colnames)
    for known_col in known_cols:
        t_df, t_colnames = get_lagged_features(known_col, t_var, t_min, t_max, istep, tstep, window_pow,
                unique_cols, df_tree)
        df_tree_t = pd.concat((df_tree_t, t_df), axis=1)
        df_tree_t_columns.extend(t_colnames)
    for observed_col_stat in observed_cols_stats:
        t_df, t_colnames = get_lagged_features(observed_col_stat, t_var, istep-1, istep, istep, tstep, 0.,
                unique_cols, df_tree)
        df_tree_t = pd.concat((df_tree_t, t_df), axis=1)
        df_tree_t_columns.extend(t_colnames)
    return df_tree_t, df_tree_t_columns


# find best recurrent features using optuna
def objective_recurrent(trial):
    t_min = trial.suggest_int('t_min', -20, istep-1)
    t_max = trial.suggest_int('t_max', istep, tstep + istep)
    window_pow = trial.suggest_float('window_pow', 0., 0.8)
    df_tree_t, df_tree_t_features = modify_tree_recurrent(df_tree, observed_cols, known_cols, observed_cols_stats, unique_cols,
             t_var, t_min, t_max, istep, tstep, window_pow, best_features)
            
    # CV using current features
    print(f'Starting features that have a significant effect on {target_feature} based on initial run using feature importances: {df_tree_t_features}')
    cv_dict = get_cv_score_lightGBM(df_tree_t_features, False,
            train_time, test_time, df_tree_t, t_var, target_feature, categorical_features, param_vals, 
            nfold, tstep, -1-istep)
    return cv_dict['score']

def get_objective_recurrent_params(study_re, observed_cols_all, known_cols_all): 
    known_cols = []
    observed_cols = []
    best_params = study_re.best_params
    for observed_col in observed_cols_all:
        if best_params[observed_col] == 1:
            observed_cols.append(observed_col)
    for known_col in known_cols_all:
        if best_params[known_col] == 1:
            known_cols.append(known_col)
    return observed_cols, known_cols, best_params['t_min'], best_params['t_max'], best_params['window_pow']


if find_recurrent_features:  # add recurrent (temporally lagged) features
    fname_recurrent_params = f"{lgb_model_str}_steps_features_params.pkl"
    if do_recurrent_opt or not os.path.exists(fname_recurrent_params):
        # features known at all times including future
        known_cols_all = ['checkout_price', 'emailer_for_promotion', 'homepage_featured', 'center_id_week_count', 
                'meal_id_week_count', t_var]
        # features only known at present and past times 
        observed_cols_all = [target_feature]  # only the target is not known at future times
       
        # # anything using target variable must be a recurrent variable to avoid information leakage!
        observed_cols_stats = []
        if use_average_target_properties:
            grp = df_tree.sort_values(t_var).groupby(['center_id', 'meal_id'])[target_feature]
            df_tree['num_orders_ts'] = grp.transform(lambda x: x.expanding().mean())
            grp = df_tree.sort_values(t_var).groupby(['center_id', 'meal_id', 'emailer_for_promotion'])[target_feature]
            df_tree['num_orders_ts_email'] = grp.transform(lambda x: x.expanding().mean())
            grp = df_tree.sort_values(t_var).groupby(['center_id', 'meal_id', 'homepage_featured'])[target_feature]
            df_tree['num_orders_ts_homepage'] = grp.transform(lambda x: x.expanding().mean())
            grp = df_tree.sort_values(t_var).groupby(['center_id', 'meal_id', 'emailer_for_promotion', 'homepage_featured'])[target_feature]
            df_tree['num_orders_ts_email_homepage'] = grp.transform(lambda x: x.expanding().mean())
            observed_cols_stats = ['num_orders_ts', 'num_orders_ts_email', 'num_orders_ts_homepage', 'num_orders_ts_email_homepage']
    
        # optimize recurrent hyperparameters
        isteps = -np.arange(tstep) - 1  # -1, -2, ..., -tstep
        best_features_steps = []
        t_min, t_max, window_pow = -16, 0, 0.4596953866951567
        params_recurrent = []
        for ind, istep in enumerate(isteps):  # delays to use for finding best recurrent features
            t_max = min(t_max, tstep + istep)  # t_max cannot be greater than number of possible future points
            fname_recurrent_features1 = f"{lgb_model_str}_re_features_Sfs_FW_{-istep}.pkl"
            fname_recurrent_features2 = f"{lgb_model_str}_re_features_range_power_{-istep}.pkl"
            print(f'{istep}: {fname_recurrent_features1}')
    
            # first training step using recursive algorithm to find relevant recursive features 
            if do_recurrent_opt_force or not os.path.exists(fname_recurrent_features1):
                # perform recursive addition/subtraction feature selection until convergence
                df_tree_t, df_tree_t_features = modify_tree_recurrent(df_tree, observed_cols_all, known_cols_all, observed_cols_stats, 
                        unique_cols, t_var, t_min, t_max, istep, tstep, window_pow, best_features)
                # selectable_features = [[best_feature] for best_feature in best_features]
                selectable_features = []
                for known_col in known_cols_all:
                    selectable_features.append(get_lagged_features(known_col, t_var, t_min, t_max, istep, tstep, 
                        window_pow, names_only=True))
                cur_features = best_features.copy()
                for observed_col in observed_cols_all:
                    cur_features.extend(get_lagged_features(observed_col, t_var, t_min, istep, istep, tstep,
                        window_pow, names_only=True))
                get_cv_score_lightGBM_params = False, train_time, test_time, df_tree_t, t_var, target_feature, categorical_features, param_vals, nfold, tstep
                train_val_lightGBM_params = [train_time, test_time, df_tree_t, t_var, target_feature, categorical_features, param_vals]
                get_cv_score_lightGBM_params_dict = {'num_step': -1-istep} 
                train_val_lightGBM_params_dict = {'num_step': -1-istep}
                sfs_fw = Sfs_FW(get_cv_score_lightGBM, get_cv_score_lightGBM_params, get_cv_score_lightGBM_params_dict, 
                                 cur_features=cur_features, selectable_features=selectable_features, 
                                 use_important_features=use_important_features, test_func=train_val_lightGBM, 
                                 test_params=train_val_lightGBM_params, test_params_dict=train_val_lightGBM_params_dict)
                sfs_fw.optimize()
                best_features1 = sfs_fw.best_features
                known_cols = [known_col for known_col in known_cols_all if any(
                    col[:len(known_col) + 1] == known_col + '_' for col in best_features1)]
                observed_cols = observed_cols_all.copy()  # just use the target_features 
                save_model_pickle(fname_recurrent_features1, (sfs_fw, known_cols, observed_cols, observed_cols_stats))
            sfs_fw, known_cols, observed_cols, observed_cols_stats = load_model_pickle(fname_recurrent_features1)
                        
            # second step using optuna to find optimal range and power (t_min, t_max, window_pow)
            if do_recurrent_opt_force or not os.path.exists(fname_recurrent_features2):
                sampler = optuna.samplers.TPESampler()
                study_re = optuna.create_study(direction='maximize', sampler=sampler)
                study_re.enqueue_trial(
                    {
                        't_min': t_min,
                        "t_max": t_max,
                        "window_pow": window_pow,
                    }
                )
                study_re.optimize(objective_recurrent, n_trials=30)
                save_model_pickle(fname_recurrent_features2, study_re)
            study_re = load_model_pickle(fname_recurrent_features2)
            best_params = study_re.best_params
            t_min, t_max, window_pow = best_params['t_min'], best_params['t_max'], best_params['window_pow']
            df_tree_t, df_tree_t_features = modify_tree_recurrent(df_tree, observed_cols, known_cols, observed_cols_stats, unique_cols,
                    t_var, t_min, t_max, istep, tstep, window_pow, best_features)
            if test_recurrent:  # use CV 'score' to decide whether time-dependence helps (otherwise use vanilla model)
                print('testing recurrent best params:')
                val_dict = train_val_lightGBM(df_tree_t_features, train_time, test_time, df_tree_t, t_var,
                        target_feature, categorical_features, param_vals, num_step=-1-istep)
                cv_dict = get_cv_score_lightGBM(df_tree_t_features, False, train_time, test_time, df_tree_t, t_var, 
                        target_feature, categorical_features, param_vals, nfold, tstep, num_step=-1-istep)
                print('testing default "best_features":')
                val_dict_n = train_val_lightGBM(best_features, train_time, test_time, df_tree_t, t_var,
                        target_feature, categorical_features, param_vals, num_step=-1-istep)
                cv_dict_n = get_cv_score_lightGBM(best_features, False, train_time, test_time, df_tree, t_var,
                        target_feature, categorical_features, param_vals, nfold, tstep, num_step=-1-istep)
    
            best_features_steps.append(df_tree_t_features)
            params_recurrent.append([istep, t_min, t_max, window_pow, observed_cols, known_cols, observed_cols_stats])
        save_model_pickle(fname_recurrent_params, (best_features_steps, params_recurrent, best_features))
    best_features_steps, params_recurrent, best_features = load_model_pickle(fname_recurrent_params)

def recurrent_models_save(train_time, test_time, df_tree, t_var, fname_recurrent_params,
        target_feature, categorical_features, param_vals, lgb_model_str, alphas):
    """ train and validate models for all time-steps with recurrent features and save to disk with pickle """
    _, params_recurrent, best_features = load_model_pickle(fname_recurrent_params)
    val_and_finalize_lightGBM(train_time, test_time, df_tree, t_var, lgb_model_str,
            best_features, None, target_feature, categorical_features, param_vals, alphas)
    for num_step in range(len(params_recurrent)):
        # validate and finalize GBM model using train/test data for time-series
        val_and_finalize_lightGBM(train_time, test_time, df_tree, t_var, lgb_model_str, 
                best_features, params_recurrent[num_step], target_feature, categorical_features, param_vals, alphas)

alphas = [None]
if quantile_alphas is not None:
    alphas.extend(quantile_alphas)
if write_new_model:  # write all the partial models
    recurrent_models_save(train_time, test_time, df_tree, t_var, fname_recurrent_params, target_feature, categorical_features, 
            param_vals, lgb_model_str, alphas)

def model_predictions_gbm(df_tree, fname_recurrent_params, t_var,
        id_var, target_feature, do_test, train_time, test_time, lgb_model_str, alphas):
    """ Obtain time-series predictions for the test set (do_test=True) or future predictions (do_test=False) """
    _, params_recurrent, best_features = load_model_pickle(fname_recurrent_params)

    # get model evaluation for each time-step on the test set and obtain predictions 
    preds_gbm = []
    for alpha in alphas:
        preds_gbm.append(pd.DataFrame(columns=[id_var, target_feature]))
    for num_step in range(len(params_recurrent)):
        # create the features on the fly for this istep 
        param_recurrent = params_recurrent[num_step]
#         if num_step > 10:
#             param_recurrent = None
        df_tree_t, df_tree_t_features, tstep, istep = get_tree_features(param_recurrent, df_tree, best_features, unique_cols, t_var)
        print(istep)
        for ind, alpha in enumerate(alphas):
            # load the model with pickle if we did not save them
#             if num_step > 10:
#                 lgb_model_str_ = get_pickle_file_name(lgb_model_str, do_test, None, alpha)
#             else:
            lgb_model_str_ = get_pickle_file_name(lgb_model_str, do_test, num_step, alpha)
            gbm = load_model_pickle(lgb_model_str_)['model']
        
            # make predictions with LightGBM
            val_time = get_val_time(do_test, train_time, test_time)
            # Create LightGBM train and test set
            df_X = df_tree_t[df_tree_t_features]
            pred_inds = df_tree_t[t_var] == val_time + num_step + 1
            ids = df_tree_t[pred_inds][id_var]
            X_test_ = df_X[pred_inds]
            gbm_frame = pd.DataFrame({id_var: ids, target_feature: gbm.predict(X_test_)})
            preds_gbm[ind] = pd.concat((preds_gbm[ind], gbm_frame), axis=0)
    return preds_gbm


def write_csv_sort(preds_gbm, ids_submit, target_feature, df_name):
    ids1 = preds_gbm.id.values
    sorter = np.argsort(ids1)
    inds1_2 = sorter[np.searchsorted(ids1, ids_submit, sorter=sorter)]  # permutation of ids1 <--> ids2
    assert np.all(preds_gbm.iloc[inds1_2].id.values == ids_submit)
    target_values = preds_gbm.iloc[inds1_2][target_feature]
    df_predictions = pd.DataFrame({id_var: ids_submit, target_feature: np.around(target_values).astype('int')})
    df_predictions.to_csv(df_name, index=False)
    print(f'{df_name} written')


# get model predictions for testing and future predictions
if write_new_data:
    test_preds_gbm = model_predictions_gbm(df_tree, fname_recurrent_params, t_var, 
        id_var, target_feature, True, train_time, test_time, lgb_model_str, alphas)
    final_preds_gbm = model_predictions_gbm(df_tree, fname_recurrent_params, t_var, 
        id_var, target_feature, False, train_time, test_time, lgb_model_str, alphas)
else:
    test_preds_gbm = []
    final_preds_gbm = []
for ind, alpha in enumerate(alphas):
    fname_pred = lgb_model_str + '_test_alpha'+ str(alpha if alpha is not None else '_l2') +'.csv'
    fname_final = lgb_model_str + '_alpha'+str(alpha if alpha is not None else '_l2')+'.csv'
    if write_new_data:
        nan_sum = 0
        nan_sum += np.count_nonzero(np.isnan(test_preds_gbm[ind][target_feature].values))
        nan_sum += np.count_nonzero(np.isnan(final_preds_gbm[ind][target_feature].values))
        if nan_sum == 0:        
            # write results for test set
            ids2 = df[(df[t_var] > train_time) & (df[t_var] <= test_time)][id_var].values
            write_csv_sort(test_preds_gbm[ind], ids2, target_feature, fname_pred)

            # write results for kaggle submission test
            ids2 = df_predictions[id_var].values
            write_csv_sort(final_preds_gbm[ind], ids2, target_feature, fname_final)

        else:
            raise ValueError(f'{nan_sum} missing values. Not writing {fname_pred} or {fname_final}!')
    else:
        test_preds_gbm.append(pd.read_csv(fname_pred))
        final_preds_gbm.append(pd.read_csv(fname_final))

# R2 LightGBM test score for all the predictions
def get_R2_score(predictions, target_values, weights=None):
    """ 
    Calculate final score (R**2 value) 
    Note that sklearn.metrics.r2_score gives weird results when input arrays are not flattened

    :param predictions: [np.array] predicted values
    :param target_values: [np.array] corresponding target values
    :param weights: [np.array] inverse variance estimates

    :Return float
    """
    predictions_f = predictions.astype('f8')
    target_values_f = target_values.astype('f8')
    if weights is None:
        weights_f = np.ones_like(predictions_f)
    else:
        weights_f = weights.astype('f8')
    unexplained_var = np.mean(weights_f * (target_values_f - predictions_f)**2)  # unexplained weighted variance
    mean_target_value = np.sum(target_values_f * weights_f) / np.sum(weights_f)  # weighted average
    explained_var = np.mean(weights_f * (target_values_f - mean_target_value)**2)  # explained weighted variance
    return 1. - unexplained_var / explained_var

def get_lgb_scores(df, preds_gbm, id_var, target_feature, alphas):
    df_tree_scores = []
    for ind, alpha in enumerate(alphas):
        preds_gbm_ = preds_gbm[ind]
        mask_gbm = df[id_var].isin(preds_gbm_[id_var])
        lgb_targets = df[mask_gbm].sort_values(id_var)[target_feature].values
        lgb_preds = preds_gbm_.sort_values(id_var)[target_feature].values
        df_tree_scores.append(get_R2_score(lgb_preds, lgb_targets))
        print(f'alpha: {alpha}, score: {df_tree_scores[-1]}')
    return df_tree_scores

print('test scores: ')
df_tree_test_scores =  get_lgb_scores(df_tree, test_preds_gbm, id_var, target_feature, alphas)
if run_mode == 2:
    print('final scores:' )
    df_tree_final_scores = get_lgb_scores(df_sample, final_preds_gbm, id_var, target_feature, alphas)

# look at (outliers, anomalies, statistics) for each time-step: see if this can help with feature engineering
if False:
    num_step = 0
    do_test = False
    alpha = alphas[0]
    param_recurrent = params_recurrent[0]   # use the next prediction
    df_tree_t, df_tree_t_features, tstep, istep = get_tree_features(param_recurrent, df_tree, best_features, unique_cols, t_var)
    lgb_model_str_ = get_pickle_file_name(lgb_model_str, do_test, num_step, alpha)
    gbm = load_model_pickle(lgb_model_str_)['model']
    num_outliers = 3
    print(f'num_step: {num_step}, do_test: {do_test}, tstep: {tstep}, istep: {istep}, num_outliers: {num_outliers}')
    for t_val in range(start_time, end_time):
        df_X, df_y, train_inds, test_inds = create_gbm_data_and_params(True, t_val, t_val+1, df_tree_t, t_var,
            df_tree_t_features, target_feature)
        X_train, X_test, y_train, y_test = df_X[train_inds], df_X[test_inds], df_y[train_inds], df_y[test_inds]
        predictions_test = gbm.predict(X_test)
        num_l1 = len(predictions_test)
        if t_val < test_time:
            residuals_test_l1 = np.abs(predictions_test - y_test.values)
            median_l1 = np.median(residuals_test_l1)
            mean_l1 = np.mean(residuals_test_l1)
            std_l1 = np.std(residuals_test_l1)
            residuals_test_sortinds = np.argsort(residuals_test_l1)
            largest_l1 = residuals_test_l1[residuals_test_sortinds][-num_outliers:]
            largest_l1_expected = y_test.iloc[residuals_test_sortinds[-num_outliers:]].values
            predictions_l1_expected = predictions_test[residuals_test_sortinds[-num_outliers:]]
            print(f'\n{t_var}: {t_val+1}, score: {gbm.score(X_test, y_test):.2f}, count: {num_l1}, residuals avg: {mean_l1:.2f}, median: {median_l1:.2f}, std: {std_l1:.2f}, largest: {largest_l1}, target: {largest_l1_expected}, predictions: {predictions_l1_expected}, mean predictions: {np.mean(predictions_test):.2f}')
        else:
            print(f'\n{t_var} {t_val+1}, count: {num_l1}, mean predictions: {np.mean(predictions_test):.2f}')



########## The rest of this code is related to time-series plotting/visualization only ########
plot_cols = ['checkout_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders']  # time-varying parameters to plot 
num_pars = len(plot_cols) # number of features 
default_alg_index = algorithms.index(default_algorithm) 

# populate known data for plotting/visualization of time series
if run_mode == 2:
    for row in df_sample.iterrows():
        df.loc[df[id_var] == row[1][id_var], target_feature] = row[1][target_feature]

# By plotting averaged quantities vs time we see that target_feature is correlated with 'emailer_for_promotion' and 'homepage_featured'
if plot_data:
    ml_vis_eda.df_plot_ts(df_copy, t_var, f'time ({t_var})', ycols=plot_cols, ylabels=plot_cols)

# run through each time-series comparing losses for these algorithms
np.random.seed(1)  # for reproducible plots
if plot_ts > 0:
    for ind_ts in np.random.choice(num_ts, plot_ts, replace=False):
        for do_test in [True, False]:
            if do_test:
                preds_gbm = test_preds_gbm
                ts_inds_pred = [1, 2]
            else:
                preds_gbm = final_preds_gbm
                ts_inds_pred = [2, 3]

            # row indices of dataframe for the time series
            df_row_inds_train = np.arange(ts_inds[ind_ts, 0], ts_inds[ind_ts, ts_inds_pred[0]]) 
            df_row_inds_pred = np.arange(ts_inds[ind_ts, ts_inds_pred[0]], ts_inds[ind_ts, ts_inds_pred[1]])
            time_values_pred = list(df.iloc[df_row_inds_pred][t_var])  # time values for the predictions

            # mean value of the target for comparison
            target_values_train = df.iloc[df_row_inds_train][target_feature]
            test_pred1 = np.full([len(df_row_inds_pred)], target_values_train.mean())
        
            # predictions and corresponding confidence (quantile) estimates
            test_pred2 = []
            for ind, alpha in enumerate(alphas):
                target_values_pred_ids = df.iloc[df_row_inds_pred][id_var]
                mask_gbm = preds_gbm[ind][id_var].isin(target_values_pred_ids)
                test_pred2.append(preds_gbm[ind].loc[mask_gbm, target_feature].values)
                if alpha is None:
                    ind_use = ind
        
            # plot the data using the plotter object
            category_title_vals = list(df.iloc[df_row_inds_pred[0]][categorical_features].values)
            col_keyvals = {k:v for k,v in zip(categorical_features, category_title_vals)}
            plotter = ml_vis_eda.TimeSeriesPlotter(df)
            plotter.filter_data(col_keyvals)
            test_preds = np.array([test_pred1, test_pred2[ind_use]])
            plotter.plot_single_time_series(t_var, plot_cols, plot_cols, f'time ({t_var})', target_feature, 
            test_preds, algorithms, \
            [time_values_pred]*len(algorithms), test_pred2[1:], alphas[1:], default_algorithm)
plt.show()

# Strange discretization pattern in the num_prders (only takes on unique values in a repeating sequence!)
df_another = pd.read_csv('foodDemand_train/train.csv')
x = np.unique(df_another[target_feature]).astype('int')
i = np.arange(100)  # sequence index i starting from 0
n = 13 + (i % 3) + (i // 6) * 27 + ((i // 3) % 2) * 13  # returns ith value in the sequence
print(n)  # first 100 terms in the repeating sequence (period of 6)
print(np.all(x[:100] == n))  # all values are in this sequence
i_x = x * 6 // 27 - 2 + (x % 27 == 15) + (x % 27 == 1)  # gives the index i for each x value
x_ = 13 + (i_x % 3) + (i_x // 6) * 27 + ((i_x // 3) % 2) * 13
print(np.all(x_ == x))

b = lambda x_: ((x_ - 13) % 27 in [0, 1, 2, 13, 14, 15]) and x_ >= 13  # returns True if x_ is in the sequence n[i]
print([b(x_) for x_ in range(100)])
print(all([b(x_) for x_ in x]))  

