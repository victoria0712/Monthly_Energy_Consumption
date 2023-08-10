import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import re

from ispots.forecaster.utils.lime_sp import lime_tabular_sp
from ispots.forecaster.utils.feature_builder import feature_builder

class lime_explanation():
    """ Lime explanation to identify top n (deafult = 10) most important features for a model's forecast
        Currently supports XGB, LGB, and Linear models 

    Parameters
    ----------
    model: AutoForecasting class
        Forecast model

    instance: string
        Date and time of the forecast instance that needs explanation

    df_pp: pd.DataFrame of shape (n_samples, 1)
        Use data retrieved from the system that has been preprocessed. includes the most recent data available
        The data is used to generate features of which the distribution will be used for Lime training and Boruta feature selection.
        The dataframe should contain the most recent data available and should have sufficient look back data from the instance to be 
        explained. The look back data should include the model look back period plus the maximum of the 2 periods used during model 
        explainability, i.e. look back period for lime and look back period for feature selection. When using the default parameters  
        for look back periods, the dataframe should have:
        * model look back period + 7 days of data for sub-daily forecasting model 
        * model look back period + 12 weeks of data for daily forecasting model

    params: dict 
        Contains parameters for lime_look_back_dur, feat_selctn_look_back_dur
        lime_look_back_dur: number of look back days in features built for Lime to learn 
        feat_selctn_look_back_dur: number of look back days in features built for feature selection
        If not specified, for model_freq < 1day: the values are '7d', '3d', respectively
                        for model_freq >= 1day: the values are '12w' and '12w' respectively

    """

    def __init__(self, 
                model, 
                instance, 
                df_pp, 
                params=None):

        self.model = model
        self.model_name = model.best_model.name

        # check whether model is supported
        supported_models = ['XGB', 'LGB', 'Linear', 'LSTM']
        if self.model_name not in supported_models:
            raise ValueError(f'{self.model_name} is not supported, only {supported_models} is supported for explainability.')

        self.model_freq = model.freq
        self.model_cyclic_feature_encoding = model.best_model.cyclic_feature_encoding
        self.model_look_back_periods = model.best_model.look_back_periods 
        self.model_horizon = model.horizon
        self.instance = pd.to_datetime(instance) 
        self.df = df_pp

        if self.model_name == 'LSTM':
            self.instance_time = pd.to_datetime(instance).time()
            self.model_scaler_input = model.best_model.scaler_input
            self.model_scaler_output = model.best_model.scaler_output
            self.model_fcst_periods = model.best_model.fcst_periods
            self.model_cycle_periods = model.best_model.cycle_periods

        if params is None:
            if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                self.model_params = {
                    'lime_look_back_dur':'7d',   
                    'feat_selctn_look_back_dur': '3d'
                }
            else:
                self.model_params = {
                    'lime_look_back_dur':'12w',   
                    'feat_selctn_look_back_dur': '12w',
                }
        else:
            self.model_params = params

        # calculate the number of periods for lime_look_back_periods and feat_selctn_look_back_periods
        self.lime_look_back_periods = int(self.model_params['lime_look_back_dur'] / pd.Timedelta(self.model_freq))
        self.feat_selctn_look_back_periods = int(self.model_params['feat_selctn_look_back_dur'] / pd.Timedelta(self.model_freq))

        # calculate the number of periods to retrieve from past data df
        # this includes the number of days required to build 1 set of features in feature builder and maximum periods 
        # from the number of sets required in Lime look back or feature selection look back
        if self.model_name in ['Linear', 'LGB', 'XGB']:
            self.past_periods_req = self.model_look_back_periods + max(self.lime_look_back_periods, self.feat_selctn_look_back_periods)
        elif self.model_name == "LSTM":
            self.past_periods_req = self.model_look_back_periods + self.lime_look_back_periods
 
        # retrieve the required period of data from past data df
        end_ts = max(self.df.index)
        start_ts = end_ts - pd.Timedelta(self.model_freq)*(self.past_periods_req -1 )
        self.df = self.df.loc[start_ts:end_ts]

        # check if instance to be explained is in the forecast horizon
        if end_ts >= self.instance or (end_ts + pd.Timedelta(self.model_horizon)) < self.instance:                
            raise ValueError('Instance to be explained is not in the forecast horizon')

        # if instance is in forecast horizon, calculate for LSTM, the position of the instance to be explained in the forecast horizon    
        if self.model_name == 'LSTM':
            time_gap = self.instance - end_ts
            pos = time_gap / pd.Timedelta(self.model_freq) -1
            self.pos = int(pos) 

        # check if look back data is sufficient
        if len(self.df) < self.past_periods_req:                
            raise ValueError('Not enough data for look back period')

        # calculate the number of periods to forecast (this is between look back data and instance to be explained)
        self.periods_to_fcst = int((self.instance - end_ts) / pd.Timedelta(self.model_freq))

    def explain(self,
                discretize_continuous=False,
                final_num_features = 10,
                num_generated_samples = 5000, 
                num_samples_surrogate_model = 5000,
                distance_metric_type = 'euclidean'):

        """Generates explanations for a prediction - identifies the top n features and
        their percentage contribution.

        First, forecasting features will be built based on historical data passed when 
        initializing lime_explanation. Next, feature selection using Boruta will be 
        performed to identify potential most important features. Lime will identify the 
        top n most important features from this list. The perentage contribution of 
        each feature to the prediction explanation is calculated. 

        Args:
            discretize_continuous: if True, non-categorical features will be discretized 
                into quartiles. Default is False where no discretizing is performed.
            final_num_features: maximum number of features present in explanation.
            num_generated_samples: size of the synthetic neighborhood to generate in Lime.
            num_samples_surrogate_model: size of the neighborhood to learn the linear model
                in Lime. This value should not be more than the num_generated_samples.
            distance_metric_type: type of metric used to calculate the distance between the 
                instance to be explained and each of the synthetic samples generated.

        Returns:
            A dataframe containing the top n features, the values of these features, the 
            Lime coefficient of these features and the percentage contribution of each feature. 
        """

        self.discretizer = discretize_continuous

        # check that the number of synthetic samples generated should be no less than 5000 
        if num_generated_samples < 5000:                
            raise ValueError('Number of synthetic samples generated should be no less than 5000')

        # check that the number of samples used to build surrogate model is not more than the number of samples generated
        if num_samples_surrogate_model > num_generated_samples:                
            raise ValueError('The number of samples used for building surrogate model should not be more than the number of samples generated')        

        # check that for non-discretizing approach, num_samples_surrogate_model should be the same as num_generated_samples
        if (not self.discretizer) and (num_samples_surrogate_model != num_generated_samples):  
            num_samples_surrogate_model = num_generated_samples
            print("When non-driscretizing approach is used, the number of samples used to build the surrogate model is the same as that generated.")

        # define the predict function for a single instance
        def predict_fn(X_feature):                      
            if self.model_name == 'XGB':
                test_matrix = xgb.DMatrix(X_feature)
                y_fcst = self.model.best_model.model.predict(test_matrix, ntree_limit=self.model.best_model.model.best_ntree_limit)
            
            elif self.model_name == 'LGB':
                y_fcst = self.model.best_model.model.predict(X_feature, num_iteration=self.model.best_model.model.best_iteration)

            elif self.model_name == 'Linear':
                X_feature = self.model.best_model.scaler.transform(X_feature)
                y_fcst = self.model.best_model.model.predict(X_feature)

            elif self.model_name == 'LSTM':
                test = X_feature.reshape((num_samples_surrogate_model, self.model_look_back_periods, -1))
                pred = self.model.best_model.model.predict(test)

                if self.model.best_model.type == 'vanilla':
                    pass
                elif self.model.best_model.type == 'encoder-decoder':
                    pred = pred.reshape(num_samples_surrogate_model, -1)
                
                fcst = self.model_scaler_output.inverse_transform(pred)
                y_fcst = fcst[:, self.pos]

            return y_fcst

        if self.model_name in ['XGB', 'LGB', 'Linear']:
            # build features for past df 
            X_feature_full = []
            y_full = []
            X_feature, y_values, feature_names = feature_builder(self.df, self.model_freq, self.model_name, self.model_cyclic_feature_encoding)
            X_feature_full.extend(X_feature.tolist())
            y_full.extend(y_values.reshape(-1).tolist())

            # forecast and build features for the period between past data and instance to be explained
            X_temp = self.df[-self.model_look_back_periods:].copy()
            for i in range(self.periods_to_fcst):
                start_time = X_temp.index[i]
                end_time = X_temp.index[-1] + pd.Timedelta(self.model_freq)
                idx_curr = pd.date_range(start=start_time, end=end_time, freq=self.model_freq)
                X_curr = X_temp.reindex(index=idx_curr)
                X_feature, _, _ = feature_builder(X_curr, self.model_freq, self.model_name, self.model_cyclic_feature_encoding)
                X_feature = X_feature[[-1]]
                X_feature_full.extend(X_feature.tolist())

                y_fcst = predict_fn(X_feature)

                idx_new = pd.date_range(start=X_temp.index[0], end=X_temp.index[-1]+pd.Timedelta(self.model_freq), freq=self.model_freq)
                X_temp = X_temp.reindex(idx_new)
                X_temp.iloc[-1] = y_fcst
                y_full.extend(y_fcst)

            # slice for data row
            data_row_X = np.array(X_feature_full[-1]) 

            # slice data to get periods for Lime
            lime_train_X = np.array(X_feature_full[(-self.lime_look_back_periods-1):-1])

            if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                # slice data to get periods for feature selection
                feature_selctn_X = np.array(X_feature_full[-self.feat_selctn_look_back_periods-1:-1])
                feature_selctn_y = pd.Series(y_full[-self.feat_selctn_look_back_periods-1:-1])

            # identify names of categorical features to be provided into Lime Explainer
            hour_features = []
            week_features = []
            lime_categorical_features = {}
            for i in feature_names:
                if i.find('hour') != -1:
                    hour_features.append(i)
                elif i.find('week') != -1:
                    week_features.append(i)

            # hour features are present only when the frequency is less than 1 day         
            if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                lime_categorical_features = {'hour_features': hour_features, 
                                            'week_features': week_features}
                subdaily_freq = True
            else:
                lime_categorical_features = {'week_features': week_features}     
                subdaily_freq = False  

            # following variable is not required for non-LSTM models
            lstm_full_feature_names = None
            lstm_merge_features = None
            lstm_cat_features_df = None

        elif self.model_name == "LSTM":            
            temp_data_X = self.df
            temp_data_X_time = feature_builder(temp_data_X, self.model_freq, self.model_name, self.model_cyclic_feature_encoding).values
            temp_data_X_scaled = self.model_scaler_input.transform(temp_data_X)
            temp_data_X_total = np.concatenate([temp_data_X_scaled, temp_data_X_time], axis=1)

            # prepare data in row format (each row will have all the lag features in lookback period)
            X_idx_range = range(len(temp_data_X_total)-self.model_look_back_periods+1)
            X_in_row_data = np.stack([temp_data_X_total[idx:idx+self.model_look_back_periods].reshape(-1) for idx in X_idx_range])

            # generate  list of feature names for (1) full data with all lag periods & (2) Lime training data &
            # (3) a full list of categorical feature names & (4) a list of only lag 1 and lag 2 categorical features
            if self.model_cyclic_feature_encoding == 'onehot':
                if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                    num_lags_req = int(pd.Timedelta('1h') / pd.Timedelta(self.model_freq))
                    past_lag_req = list(range(1,num_lags_req+1))
                else:
                    past_lag_req = [1]
                # (1) generate names for the features required by trained forecasting model 
                lstm_full_feature_names = []
                for i in range(self.model_look_back_periods,0,-1):
                    lstm_full_feature_names.append(f'lag_{i}')
                    if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                        for hour in range(24):
                            lstm_full_feature_names.append(f'lag_{i}_hour_{hour}')
                    for day in range(1,8):
                        lstm_full_feature_names.append(f'lag_{i}_dayofweek_{day}')
                    lstm_full_feature_names.append(f'lag_{i}_weekend')

                # (2) generate names for the features in Lime training data (and data_row_X)
                feature_names = []
                for i in range(self.model_look_back_periods,0,-1):
                    feature_names.append(f'lag_{i}')
                for i in past_lag_req:
                    if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                        for hour in range(24):
                            feature_names.append(f'lag_{i}_hour_{hour}')
                    for day in range(1,8):
                        feature_names.append(f'lag_{i}_dayofweek_{day}')
                    feature_names.append(f'lag_{i}_weekend')

                # (3) generate an ordered list of categorical feature names. This will be used on Lime synthetic samples 
                # to build the remaining categorical features other than those in lag_1 and lag_2.
                ordered_cat_feature_names = []
                for i in range(1,self.model_look_back_periods+1):
                    if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                        for hour in range(24):
                            ordered_cat_feature_names.append(f'lag_{i}_hour_{hour}')
                    for day in range(1,8):
                        ordered_cat_feature_names.append(f'lag_{i}_dayofweek_{day}')
                    ordered_cat_feature_names.append(f'lag_{i}_weekend')

                # (4) also, generate a list of only lag 1  and lag 2 categorical features names. This will be used to as join fields to
                # build the subsequent lag3, lag4, ... features. This list of features will be used for merging Lime synthetic 
                # samples and lstm_cat_features_df.
                lstm_merge_features = []
                for i in past_lag_req:
                    if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                        for hour in range(24):
                            lstm_merge_features.append(f'lag_{i}_hour_{hour}')
                    for day in range(1,8):
                        lstm_merge_features.append(f'lag_{i}_dayofweek_{day}')
                    lstm_merge_features.append(f'lag_{i}_weekend')

            elif self.model_cyclic_feature_encoding == 'sincos':
                if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                    past_lag_req = [1,2]
                else:
                    past_lag_req = [1]

                # (1) generate names for the features required by trained forecasting model 
                lstm_full_feature_names = []
                for i in range(self.model_look_back_periods,0,-1):
                    lstm_full_feature_names.append(f'lag_{i}')
                    if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                        lstm_full_feature_names.append(f'lag_{i}_hour_sin')
                        lstm_full_feature_names.append(f'lag_{i}_hour_cos')
                    lstm_full_feature_names.append(f'lag_{i}_dayofweek_sin')
                    lstm_full_feature_names.append(f'lag_{i}_dayofweek_cos')
                    lstm_full_feature_names.append(f'lag_{i}_weekend')

                # (2) generate names for the features in Lime training data (and data_row_X)
                feature_names = []
                for i in range(self.model_look_back_periods,0,-1):
                    feature_names.append(f'lag_{i}')                
                for i in past_lag_req:
                    if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                        feature_names.append(f'lag_{i}_hour_sin')
                        feature_names.append(f'lag_{i}_hour_cos')
                    feature_names.append(f'lag_{i}_dayofweek_sin')
                    feature_names.append(f'lag_{i}_dayofweek_cos')
                    feature_names.append(f'lag_{i}_weekend')  

                # (3) generate an ordered list of categorical feature names. This will be used on Lime synthetic samples 
                # to build the remaining categorical features other than those in lag_1 and lag_2.
                ordered_cat_feature_names = []
                for i in range(1,self.model_look_back_periods+1):
                    if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                        ordered_cat_feature_names.append(f'lag_{i}_hour_sin')
                        ordered_cat_feature_names.append(f'lag_{i}_hour_cos')
                    ordered_cat_feature_names.append(f'lag_{i}_dayofweek_sin')
                    ordered_cat_feature_names.append(f'lag_{i}_dayofweek_cos')
                    ordered_cat_feature_names.append(f'lag_{i}_weekend')

                # (4) also, generate a list of only lag 1  and lag 2 categorical features names. This will be used to as join fields to
                # build the subsequent lag3, lag4, ... features. This list of features will be used for merging Lime synthetic 
                # samples and lstm_cat_features_df.
                lstm_merge_features = []
                for i in past_lag_req:
                    if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                        lstm_merge_features.append(f'lag_{i}_hour_sin')
                        lstm_merge_features.append(f'lag_{i}_hour_cos')
                    lstm_merge_features.append(f'lag_{i}_dayofweek_sin')
                    lstm_merge_features.append(f'lag_{i}_dayofweek_cos')
                    lstm_merge_features.append(f'lag_{i}_weekend')    

            # identify categorical features in Lime training data (and data_row_X)
            hour_features = []
            week_features = []
            lime_categorical_features = {}
            for i in feature_names:
                if i.find('hour') != -1:
                    hour_features.append(i)
                elif i.find('week') != -1:
                    week_features.append(i)

            # hour features are present only when the frequency is less than 1 day         
            if pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
                lime_categorical_features = {'hour_features': hour_features, 
                                            'week_features': week_features}
                subdaily_freq = True
            else:
                lime_categorical_features = {'week_features': week_features}     
                subdaily_freq = False 

            # slice data to get periods for Lime and data row. They should contain only features in 'feature names'
            temp_df_full = pd.DataFrame(X_in_row_data, columns = lstm_full_feature_names)
            temp_df_lime = temp_df_full[feature_names].copy()
            temp_df_lime = temp_df_lime.values
            # slice to get data_row
            data_row_X = temp_df_lime[-1]
            # slice data to get periods for Lime
            lime_train_X = temp_df_lime[(-self.lime_look_back_periods-1):-1]
            
            # get df containing all possible combinations of categorical features.This will be used on synthetic samples  
            # to build the remaining categorical features other than those in lag_1
            temp_df_full = temp_df_full.iloc[-self.lime_look_back_periods:]
            lstm_cat_features_df = temp_df_full[ordered_cat_feature_names]
            if lstm_cat_features_df.duplicated().any():
                lstm_cat_features_df.drop_duplicates(inplace = True)
            self.lstm_cat_features_df = lstm_cat_features_df

        ## perform feature selection for machine learning models when data frequency is less than 1 day
        if self.model_name in ['XGB', 'LGB', 'Linear'] and pd.Timedelta(self.model_freq) < pd.Timedelta('1d'):
            rf = RandomForestRegressor(n_jobs = -1, max_depth=10, random_state=2022)
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=2022)
            feat_selector.fit(feature_selctn_X, feature_selctn_y)

            # store selected features into a list
            feature_selection_list = [feature for feature, outcome in zip(feature_names, list(feat_selector.support_)) if outcome == True]

        else:
            feature_selection_list = feature_names     

        explainer = lime_tabular_sp.LimeTabularExplainer(training_data = lime_train_X,
                                                    categorical_names = lime_categorical_features,
                                                    feature_names = feature_names,
                                                    feature_names_selected =feature_selection_list,
                                                    verbose=False, 
                                                    mode='regression',
                                                    discretize_continuous=self.discretizer,
                                                    random_state = 2022,
                                                    kernel_width = None,
                                                    subdaily_freq = subdaily_freq,
                                                    lstm_full_feature_names = lstm_full_feature_names,
                                                    lstm_merge_features = lstm_merge_features,
                                                    lstm_cat_features_df = lstm_cat_features_df)

        exp = explainer.explain_instance(data_row = data_row_X,
                                        predict_fn = predict_fn,
                                        num_features = final_num_features,
                                        num_samples = num_generated_samples,
                                        num_samples_final = num_samples_surrogate_model,
                                        distance_metric = distance_metric_type)

        self.exp = exp
        # format feature importance results into percentages
        # retrieve names and coefficients of the top n selected features by Lime
        actual_feature_name_list = []
        lime_feature_name_list = []
        coeff_list = []
        top_features = exp.as_list()
        p = re.compile(r'[a-z]\w+')
        for i in top_features:
            feature = p.findall(i[0])[0] 
            actual_feature_name_list.append(feature)
            lime_feature_name_list.append(i[0])
            coeff_list.append(i[1])

        # retrieve actual values of these features
        instance_df = pd.DataFrame(data_row_X.reshape(1,-1), columns = feature_names)
        actual_values = instance_df[actual_feature_name_list].values.tolist()[0]

        # retrieve the names of all categorical features
        all_cat =[]
        if self.model_name in ['XGB', 'LGB', 'Linear']:
            for features in lime_categorical_features.values():
                all_cat.extend(features)   
        else: 
            all_cat = ordered_cat_feature_names

        # retrieve list of binary categorical features that are identified to be of top importance and 
        # convert remaining feature values to 2 dp
        selected_binary_cat_idx = []
        for idx, name in enumerate(actual_feature_name_list):
            if name in all_cat:
                if 'sin' not in name and 'cos' not in name:
                    selected_binary_cat_idx.append(idx)
                else:
                    actual_values[idx] = '{:.2f}'.format(actual_values[idx])
            else:
                actual_values[idx] = '{:.2f}'.format(actual_values[idx])

        feature_names_plot = lime_feature_name_list.copy()
        # format binary categorical feature names in plot to True / False
        if selected_binary_cat_idx is not None:        
            for idx in selected_binary_cat_idx:
                if actual_values[idx] == 1.0:
                    actual_values[idx] = 'True'
                elif actual_values[idx] == 0.0:
                    actual_values[idx] = 'False'
                feature_names_plot[idx] = actual_feature_name_list[idx] + "=" + actual_values[idx]


        if self.discretizer == False:
            # retrieve scaled values of these selected features 
            feature_idx = []
            for feature in actual_feature_name_list:
                if feature in feature_selection_list:
                    feature_idx.append(feature_selection_list.index(feature))
            scaled_values = list(explainer.scaled_data[0][feature_idx])

            # format results into dataframe and calculate % coefficient
            results_df = pd.DataFrame({'Feature': actual_feature_name_list, 'Feature_name_plot': feature_names_plot, 'Values': actual_values, 'Coefficient': coeff_list, 'Scaled_value': scaled_values})
            total = (results_df['Coefficient'] * results_df['Scaled_value']).abs().sum()
            results_df['% Coefficient'] = (results_df['Coefficient'] * results_df['Scaled_value']).abs()/total * 100
            results_df['Feature_weight'] = results_df.apply(lambda row: row['Coefficient'] * row['Scaled_value'], axis = 1)
            results_df[['Coefficient', 'Scaled_value', '% Coefficient', 'Feature_weight']] = results_df[['Coefficient', 'Scaled_value', '% Coefficient', 'Feature_weight']].round(2)
            
            # sort results by % coefficient
            results_df.sort_values('% Coefficient', ascending = False, inplace = True)
            results_df.index = np.arange(1, len(results_df) + 1)
            self.results_table = results_df

        else:
            # format categorical feature names to exclude values
            for idx, name in enumerate(actual_feature_name_list):
                if name in all_cat:
                    lime_feature_name_list[idx] = actual_feature_name_list[idx]
            # format results into dataframe and calculate % coefficient
            results_df = pd.DataFrame({'Feature': lime_feature_name_list, 'Feature_name_plot': feature_names_plot, 'Values': actual_values, 'Coefficient': coeff_list})
            total = sum(abs(results_df['Coefficient']))
            results_df['% Coefficient'] = results_df['Coefficient'].abs()/total * 100
            results_df['Feature_weight'] = results_df.apply(lambda row: row['Coefficient'] * row['Values'] if type(row['Values']) == float else row['Coefficient'], axis = 1)
            results_df[['Coefficient', '% Coefficient', 'Feature_weight']] = results_df[['Coefficient', '% Coefficient', 'Feature_weight']].round(2)
            results_df.index = np.arange(1, len(results_df) + 1)
            self.results_table = results_df

        # check whether the number of explainers is at least equal to the final number of variables specified 
        if len(results_df) < final_num_features:
            count = len(results_df)
            print(f'Fewer than {final_num_features} features found to be most important. Explaination is provided using {count} features.')

        final_df = results_df.drop(['Feature_name_plot', 'Feature_weight'], axis = 1)

        return final_df

    def results_plot(self):
        """Generates visualisations for explanation results. 
        One of them is a bar plot for the top n feature importance in absolute percentage.
        Another plot is a bar plot for the coefficients of these top n features.        
        """

        self.results_table.sort_values('% Coefficient', inplace = True, ascending = True)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (12,6))

        ax1.barh(self.results_table['Feature_name_plot'], self.results_table['% Coefficient'])
        for i, v in enumerate(self.results_table['% Coefficient']):
            ax1.text(v + 0.2, i, str(v), color='black', fontsize=8, ha='left', va='center')
        ax1.set_title('Top 10 feature importance in absolute %', fontsize = 11, fontweight="bold")
        ax1.set_yticklabels(labels = self.results_table['Feature_name_plot'], fontsize = 10)
        ax1.yaxis.set_ticks_position('none')
        ax1.get_xaxis().set_visible(False)
        for s in ['top', 'right', 'bottom']:
            ax1.spines[s].set_visible(False)

        colors = ['green' if x > 0 else 'red' for x in self.results_table['Feature_weight']]
        ax2.barh(self.results_table['Feature_name_plot'], self.results_table['Feature_weight'], align = 'center', color = colors)
        for i, v in enumerate(self.results_table['Feature_weight']):
            if v < 0:
               ax2.text(0.2 , i, str(v), color='black', fontsize=8, ha='left', va='center')
            else: 
                ax2.text(v + 0.5, i, str(v), color='black', fontsize=8, ha='left', va='center')
        ax2.set_title('Feature weight for top 10 features', fontsize = 11, fontweight="bold")
        ax2.set_yticklabels(labels = self.results_table['Feature_name_plot'], fontsize = 10)
        ax2.yaxis.set_ticks_position('none')
        ax2.get_xaxis().set_visible(False)
        for s in ['top', 'right', 'bottom', 'left']:
            ax2.spines[s].set_visible(False)
        ax2.axvline(x=0, color='black')

        colors = ['green' if x > 0 else 'red' for x in self.results_table['Coefficient']]
        ax3.barh(self.results_table['Feature_name_plot'], self.results_table['Coefficient'], align = 'center', color = colors)
        for i, v in enumerate(self.results_table['Coefficient']):
            if v < 0:
               ax3.text(0.2 , i, str(v), color='black', fontsize=8, ha='left', va='center')
            else: 
                ax3.text(v + 0.5, i, str(v), color='black', fontsize=8, ha='left', va='center')
        ax3.set_title('Coefficients for top 10 features', fontsize = 11, fontweight="bold")
        ax3.set_yticklabels(labels = self.results_table['Feature_name_plot'], fontsize = 10)
        ax3.yaxis.set_ticks_position('none')
        ax3.get_xaxis().set_visible(False)
        for s in ['top', 'right', 'bottom', 'left']:
            ax3.spines[s].set_visible(False)
        ax3.axvline(x=0, color='black')
        
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        ax4.yaxis.set_ticks_position('none')
        for s in ['top', 'right', 'bottom', 'left']:
            ax4.spines[s].set_visible(False)

        fig.tight_layout(pad = 5)

        plt.show()