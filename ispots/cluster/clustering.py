import itertools
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.base import ClusterMixin, BaseEstimator
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

from ispots.utils.logger import get_logger

logger = get_logger('clustering-engine')

figsize = (15,3)

def scaling_on_timestamp(ts_df, scaler, scale_factors = None):
    """ Conducts scaling (either Standard Scaling or MinMax Scaling) on timestamp level and returns standardised daily time series and scale factors used.
    
    Parameters
    ----------
    ts_df :  pandas.DataFrame 
        daily time series dataframe (rows: timestamp, columns: days)
    scaler: str
        the type of scaling to do. It accepts either "standard" or "minmax"
    scale_factors: dict
        dictionary of scalers to use for each time stamp
    
    Returns
    -------
    pandas.DataFrame

    """
    tranposed = ts_df.T.copy(deep=True)
    
    if scaler == 'standard':
        if scale_factors is None: # Calculate scale_factors
            scale_factors = {timestamp:(tranposed[timestamp].mean(), np.std(tranposed[timestamp]) ) for timestamp in tranposed.columns} 
        
        for timestamp, (tmean, tsd) in scale_factors.items(): # Scale on timestamp level
            tranposed.loc[:,timestamp] = tranposed.loc[:,timestamp].apply(lambda x: (x - tmean)/tsd)
        
    elif scaler == 'minmax':
        if scale_factors is None:  # Calculate scale_factors
            scale_factors = {timestamp:(tranposed[timestamp].min(), tranposed[timestamp].max() ) for timestamp in tranposed.columns}
        
        for timestamp, (tmin, tmax) in scale_factors.items(): # Scale on timestamp level
            tranposed.loc[:,timestamp] = tranposed.loc[:,timestamp].apply(lambda x: (x - tmin)/(tmax - tmin))
    
    else:
        raise Exception('Invalid scaler used, accept only "standard" or "minmax"')
    
    return tranposed.T, scale_factors

def minkowski_dist(ts_df, p):
    """ Calculates Minkowski distance between column pairs and returns a square distance matrix.
    
    Parameters
    ----------
    ts_df :  pandas.DataFrame 
        daily time series dataframe (rows: timestamp, columns: days)
    p: float
        order of the norm of the difference ||u-v||p
    
    
    Returns
    -------
    pandas.DataFrame

    """
    dist_list = []
    cross = itertools.combinations(ts_df.columns, r=2) # Pairwise Combinations
    
    for (col1, col2) in cross:
        diff = distance.minkowski(ts_df[col1], ts_df[col2], p)
        dist_list.append(diff)
        
    dist_mat = pd.DataFrame(distance.squareform(dist_list), index=ts_df.columns, columns=ts_df.columns)
    
    return dist_mat

def dtw_dist(ts_df):
    """ Calculates DTW distance between column pairs and returns a square distance matrix.
    
    Parameters
    ----------
    ts_df :  pandas.DataFrame 
        daily time series dataframe (rows: timestamp, columns: days)
    
    Returns
    -------
    pandas.DataFrame

    """
    dist_list = []
    cross = itertools.combinations(ts_df.columns, r=2) # Pairwise Combinations
    
    for (col1, col2) in cross:
        diff = dtw(ts_df[col1], ts_df[col2])
        dist_list.append(diff)
        
    dist_mat = pd.DataFrame(distance.squareform(dist_list), index=ts_df.columns, columns=ts_df.columns)
    
    return dist_mat

def hc_detection(dist_method, dist_mat, linkage_method, n_clusters):
    """ Conducts AgglomerativeClustering based on specified linkage method and number of clusters. Returns model and clustering labels.
    
    Parameters
    ----------
    dist_method :  str
        distance used to calculate dist_mat
    
    dist_mat: array-like, shape (n_samples, n_features) or (n_samples, n_samples)
        training instances to cluster
        If dist_method=='euclidean', dist_mat is a time series dataframe with shape(n_samples, n_features)
        else, expects dist_mat to be a precomputed distance matrix with shape(n_samples, n_samples)
    
    linkage_method: str
        linkage criterion to use, accepts the following: ‘ward’, ‘complete’, ‘average’, ‘single’
    
    n_clusters: int
        number of clusters to find
    
    Returns
    -------
    sklearn.cluster.AgglomerativeClustering
        model used to cluster 
    
    array, shape (n_samples)
        clustering labels for each point
    """
    if (linkage_method == 'ward') and (dist_method=='euclidean'): # Can only use 'euclidean' distance
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage = 'ward').fit(dist_mat)
    
    elif (linkage_method in ['single', 'average', 'complete']) and (dist_method in ['dtw','minkowski']):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage = linkage_method).fit(dist_mat)
    
    else:
        raise Exception(f'{linkage_method} linkage method and {dist_method} not compatible for hc_detection')
    
    return clustering, clustering.labels_

def kmeans_detection(dist_method, dist_mat, n_clusters):
    """ Conducts TimeSeriesKMeans based on specifed number of clusters. Returns model and clustering labels.
    
    Parameters
    ----------
    dist_method :  str
        distance used to calculate dist_mat, accepts either 'euclidean' or 'dtw'
        
    dist_mat: array-like, shape(n_samples, n_features) 
        only accepts time series data with shape(n_samples, n_features) 
    
    n_clusters: int
        number of clusters to find
    
    Returns
    -------
    tslearn.clustering.TimeSeriesKMeans
        model used to cluster 
    
    array, shape (n_samples)
        clustering labels for each point
    """
    if dist_method not in ['dtw','euclidean']:
        raise Exception("Invalid distance method for time series kmeans")
        
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric = dist_method)
    labels = km.fit_predict(dist_mat)
    
    return km, labels

def dbscan_find_optimal_epsilon_and_min_samples(dist_mat, k_list):
    """ Finds optimal epsilon with correspoinding min_samples range and returns a list of tuple pairs (eps, list of min_samples). 
    
    Parameters
    ----------
    dist_mat: array-like, shape(n_samples, n_features) or shape(n_samples, n_samples)
        accepts both distance matrix and time series data
    
    k_list: array
        list of k values for nearest neighbours
    
    Returns
    -------
    array
        list of tuple pairs (eps, list of min_samples)
    """
    
    parameter_list = [] ## tuple pair: (eps, list of min_samples)
    upperLim_minSamples = int(len(dist_mat)/3)
    
    for k in k_list:
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        nbrs = neigh.fit(dist_mat)
        distances, indices = nbrs.kneighbors(dist_mat)
        distances = np.sort(distances, axis=0) # Sort according to increasing distance
        distances = distances[:,-1] # Get maximum distance for each row
        
        ## Approach 1 - Max Epsilon
        epsilon_max = distances.max()
        if epsilon_max > 0: # eps need to be positive
            parameter_list.append((epsilon_max, list(range(k+1,upperLim_minSamples)))) 

        ## Approach 2 - 75th Percentile
        epsilon_percentile = np.percentile(distances, 75)
        if epsilon_percentile > 0: # eps need to be positive
            parameter_list.append((epsilon_percentile, list(range(k,upperLim_minSamples)))) 

        ## Approach 3 - Max Jump 
        dist_diff = (distances[1:] - distances[:-1])
        index_max = np.argmax(dist_diff)
        epsilon_jump = distances[index_max]
        if epsilon_jump > 0: # eps need to be positive
            parameter_list.append((epsilon_jump, list(range(k,upperLim_minSamples))))
                
    return parameter_list

def dbscan_detection(dist_mat, eps, min_samples):
    """ Conducts DBScan detection based on specified epsilon and min samples. Returns model and clustering labels.
    
    Parameters
    ----------
    dist_mat: array-like, shape(n_samples, n_samples) 
        only accepts distance matrix and not time series data
    
    n_clusters: int
        number of clusters to find
    
    Returns
    -------
    sklearn.cluster.DBSCAN
        model used to cluster 
    
    array, shape (n_samples)
        clustering labels for each point
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric = 'precomputed').fit(dist_mat)    
    return db, db.labels_

def evaluate(algorithm, dist_method, dist_mat, labels):
    '''Returns Silhouette Score based on labels and distance matrix

    Parameters
    ----------
    algorithm: str 
        expects one of the three: ['kmeans', 'hc', 'dbscan']
    
    dist_method: str
        expects one of the three: ['dtw', 'euclidean', 'minkowski']

    dist_mat: array-like, shape(n_samples, n_samples)
        distance matrix or time series data used to get clustering labels
    
    labels: list
        clustering labels from clustering algorithm

    Returns
    -------
    float
        evaluation score
        
    '''
    if len(set(labels)) == 1: # If all samples are clustered in one group, return Silhouette score of 0
        sil_score = 0
    
    elif (algorithm == 'kmeans') and (dist_method in ['dtw', 'euclidean']):
        sil_score = silhouette_score(dist_mat, labels, metric = dist_method)
    
    elif (algorithm == 'hc') and (dist_method == 'euclidean'):
        sil_score = silhouette_score(dist_mat, labels, metric = 'euclidean')
    
    else:
        sil_score = silhouette_score(dist_mat, labels, metric='precomputed')
    
    return sil_score

def get_minkowski_p_from_meta(string):
    return int(string.split('=')[1][0])

def plot_overall_daily_profiles(ts_df, label_df, colour_dict):
    logger.debug("Overview of Daily Profiles:")

    for cluster in sorted(label_df.cluster.unique()):
        days = label_df.loc[label_df.cluster == cluster].day.values
        cluster_percentage = round(len(days)/len(label_df)*100,2)
        title = f"Cluster {cluster} (Cluster Size Percentage: {cluster_percentage}%)" 
        color = colour_dict[cluster]
        ylim = None

        if len(days)>15: # If the num_of_samples in cluster > 15, we find the upperlim and lowerlim for plotting
            upperLim = ts_df.loc[:,days].max().quantile(0.75) + ts_df.loc[:,days].quantile(0.75).std()*3
            lowerLim = ts_df.loc[:,days].min().quantile(0.25) - ts_df.loc[:,days].quantile(0.25).std()*3
            ylim = (lowerLim, upperLim)
            
            ax = ts_df.loc[:,days].plot(figsize = figsize, title = title, alpha=.50, lw=.30, color=color, legend = False, ylim = ylim)
            ax.plot(np.mean(ts_df.loc[:,days],axis=1), color=color, lw=2, ls='--')
        else: 
            ax = ts_df.loc[:,days].plot(figsize = figsize, title = title, color=color, legend = True, ylim = ylim)
        
        plt.show()
        
def plot_detected_days(df, label_df, colour_dict):
    logger.debug("Overview of Time Series:")
    
    for cluster in sorted(label_df.cluster.unique()):
        days = label_df.loc[label_df.cluster == cluster].day
        cluster_percentage = round(len(days)/len(label_df)*100,2)
        title = f"Cluster {cluster} (Cluster Size Percentage: {cluster_percentage}%)"

        df.plot(figsize=figsize, lw=.5, title = title, legend=False)
        color = colour_dict[cluster]
        for date in days:
            plt.axvspan(date, date + pd.Timedelta(days = 1), alpha = 0.1, color = color)

        plt.show()

def convert_daily_time_series_into_univariate(dataframe):
    date_df = dataframe.melt()
    time_df = pd.DataFrame({'Time':list(dataframe.index)*len(dataframe.columns)})
    univar_df = pd.concat([date_df, time_df], ignore_index = True, axis = 1)
    univar_df.columns = ['Date','Val','Time']
    univar_df['datetime'] = univar_df.apply(lambda x: datetime.datetime.combine(x.Date, x.Time), axis = 1)
    final_df = univar_df.set_index(univar_df.datetime)
    final_df = final_df[['Val']]
    return final_df

class AutoClustering(ClusterMixin, BaseEstimator):
    """ Conducts Automated Clustering which finds the best combination of clustering algorithm, 
    distance measure and hyperparameters based on the evaluation metric. 
    It outputs the clustering labels from the best model. 

    Parameters
    ----------
    algorithm_list : list, default=['hc','dbscan','kmeans']
        The list of clustering algorithms to use.

    dist_list : : list, default=['minkowski','dtw','euclidean'] 
        The list of distance measures to use.

    minkowski_p_list: iterable, default=[2, *range(1,40,5)]
        Search space for minkowski p value for distance computation

    num_of_clusters_list: iterable, default=range(2,8) 
        Search space for number of clusters for AgglomerativeClustering and TimeSeriesKMeans algorithm

    linkage_list: iterable, default=['single', 'average', 'complete', 'ward'] ## say whats valid
        Search space for linkage method for AgglomerativeClustering algorithm

    k_list: iterable, default=None
        Search space for k nearest neighbours for DBScan algorithm
        If None, search space will be set to (2, len(n_samples)/3)

    verbose: int, default = 1
        Controls the verbosity: the higher, the more messages.
        If verbose=0: No print logs
        If verbose>= 1: Only prints the main components (Distance Matrix, Algorithm and Silhouette Score)
        If verbose>= 2: Individual distance and algorithm steps are also printed

    Attributes
    ----------
    report_: pd.DataFrame
        Dataframe which contains the Silhoutte Score for each combination of algorithm, distance measure and hyperparameters
    
    clustering_labels_ : list
        Clustering labels based on best model
        This is the raw clustering labels from the clustering algorithm
    
    """
    def __init__(
        self,
        algorithm_list= ['hc','dbscan','kmeans'],
        dist_list = ['minkowski','dtw','euclidean'],
        minkowski_p_list = [2, *range(1,40,5)],
        num_of_clusters_list = range(2,8),
        linkage_list = ['single', 'average', 'complete', 'ward'],
        k_list = None,
        verbose = 1
    ):

        if not all(item in ['hc','dbscan','kmeans'] for item in algorithm_list):
            raise ValueError("Invalid algorithms specified. Current valid algorithms are: 'hc','dbscan','kmeans'") 
        
        if not all(item in ['minkowski','dtw','euclidean'] for item in dist_list):
            if 'manhattan'in dist_list:
                raise ValueError("For manhattan distance, use 'minkowski' distance in dist_list and include 1 in minkowski_p_list")
            raise ValueError("Invalid distance measures specified. Current valid distance measures are: 'minkowski','dtw','euclidean'") 

        if min(num_of_clusters_list) < 0:
            raise ValueError("Negative number in num_of_clusters_list. Ensure that only postive values are specified for num_of_clusters_list")

        if not all(item in ['single', 'average', 'complete', 'ward'] for item in linkage_list):
            raise ValueError("Invalid linkage methods specified. Valid linkage methods are: 'single', 'average', 'complete', 'ward'")

        if k_list is not None:
            if min(k_list) < 0:
                raise ValueError("Negative number in k_list. Ensure that only postive values are specified for k_list")

        self.algorithm_list = algorithm_list
        self.dist_list = dist_list
        self.minkowski_p_list = minkowski_p_list
        self.num_of_clusters_list = num_of_clusters_list
        self.linkage_list = linkage_list
        self.k_list = k_list
        self.verbose = verbose

    def fit(self, X, y=None):
        pass
    
    def fit_predict(self, X, y=None):
        """Finds the best combination of clustering algorithm, distance measure and hyperparameters for a daily time series dataframe.

        1. Conducts Feature Scaling (both minmax and standard)
        2. Calculates distance matrices based on dist_list provided
        3. For each algorithm, use the different distance matrices and hyperparameters to output the clustering labels
        4. Each clustering label will be evaluated using the evaluation function
        5. The combination of algorithm, distance measure, hyperparameter and corresponding evaluation score will be saved in a report dataframe (report_ attribute)
        6. We output the clustering labels from the best model

        Parameters
        ----------
        X : pd.DataFrame shape (n_samples: timestamp, n_features: day)
            Daily Time Series Dataframe
        
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns a fitted instance of self.

        """        
        if not (type(X) == pd.core.frame.DataFrame):
            raise ValueError("Supplied X is not a dataframe. X needs to be a daily time series dataframe of shape (n_samples: timestamp, n_features: day)")

        logger.info('Start auto clustering')
    
        report_dict = {}

        ## Feature Scaling for Minkowski Distances
        standard_scaled, standard_scale_factors = scaling_on_timestamp( X, 'standard')
        minmax_scaled, minmax_scale_factors = scaling_on_timestamp( X, 'minmax')
        scaler_dict = {'standard': {'scaled':standard_scaled, 'scale_factors': standard_scale_factors},
                    'minmax': {'scaled':minmax_scaled, 'scale_factors': minmax_scale_factors}}

        ## Distance Matrices Calculation
        if self.verbose > 0:
            logger.debug(f"---- Distance Matrix Calculation")
        
        dist_mat_dict = {} 
        for distance_measure in self.dist_list:
            if distance_measure == 'dtw':
                if self.verbose > 1:
                    logger.debug(f"-- dtw")
                dist_matrix = dtw_dist(X)
                dist_mat_dict['dtw'] = dist_matrix
                
            elif distance_measure == 'minkowski':
                for scaler in scaler_dict.keys():
                    scaled = scaler_dict[scaler]['scaled']
                    for p in self.minkowski_p_list:
                        if self.verbose > 1:
                            logger.debug(f"-- minkowski (p={p},scaler={scaler})")
                        dist_matrix = minkowski_dist(scaled, p)
                        dist_mat_dict[f"minkowski (p={p},scaler={scaler})"] = dist_matrix

        ## Algorithm Detection
        if self.verbose > 0:
            logger.debug(f"---- Algorithm Detection")
        
        for algorithm in self.algorithm_list:
            if self.verbose > 1:
                logger.debug(f"-- {algorithm}")

            ## K-Means Algorithm
            if algorithm == 'kmeans': 
                report_dict[algorithm] = {}
                
                for distance_measure in self.dist_list:                    
                    if distance_measure == 'dtw':
                        report_dict[algorithm][distance_measure] = {}
                            
                        for n_cluster in self.num_of_clusters_list:
                            if self.verbose > 1:
                                logger.debug(f"- Distance Measure: {distance_measure}, Num of clusters: {n_cluster}")
                            
                            model, labels = kmeans_detection(distance_measure, X.T, n_cluster) # Detection
                            score = evaluate(algorithm, distance_measure, X.T, labels) # Evaluation     
                            report_dict[algorithm][distance_measure][f"num of clusters = {n_cluster}"] = {'eval_score': score, 'labels': labels}
                                                            
                    elif distance_measure == 'euclidean':
                        for scaler in scaler_dict.keys():
                            report_dict[algorithm][f"{distance_measure} (scaler={scaler})"] = {}
                            scaled = scaler_dict[scaler]['scaled']
                            
                            for n_cluster in self.num_of_clusters_list:
                                if self.verbose > 1:
                                    logger.debug(f"- Distance Measure: {distance_measure} (scaler={scaler}), Num of clusters: {n_cluster}")
                                model, labels = kmeans_detection(distance_measure, scaled.T, n_cluster) # Detection
                                score = evaluate(algorithm, distance_measure, scaled.T, labels) # Evaluation                         
                                report_dict[algorithm][f"{distance_measure} (scaler={scaler})"][f"num of clusters = {n_cluster}"] = {'eval_score': score, 'labels': labels}

            ## HC Algorithm
            elif algorithm == 'hc':
                report_dict[algorithm] = {}
                
                ## Load Distance Matrix
                for distance_measure in dist_mat_dict.keys():
                    dist_matrix = dist_mat_dict[distance_measure]
                    report_dict[algorithm][distance_measure] = {}
                        
                    ## Search Space
                    for n_cluster in self.num_of_clusters_list:
                        for linkage_method in self.linkage_list:
                            labels = []
                            
                            if linkage_method != 'ward':
                                model, labels = hc_detection(distance_measure.split(' ')[0], dist_matrix, linkage_method, n_cluster) ## Detection
                                score = evaluate(algorithm, distance_measure, dist_matrix, labels)
                                
                            elif (linkage_method == 'ward') and ('p=2' in distance_measure):
                                scaler = distance_measure.split('scaler=')[-1][:-1] ## obtain scaler for scaled
                                scaled = scaler_dict[scaler]['scaled']
                                model, labels = hc_detection('euclidean', scaled.T, 'ward', n_cluster)
                                score = evaluate(algorithm, 'euclidean',  scaled.T, labels)
                            
                            if len(labels) > 0:
                                report_dict[algorithm][distance_measure][f"num of clusters = {n_cluster}, linkage = {linkage_method}"] = {'eval_score': score, 'labels': labels}

            ## DBScan Algorithm
            elif algorithm == 'dbscan':
                report_dict[algorithm] = {}
                
                ## Distance Matrix
                for distance_measure in dist_mat_dict.keys():
                    dist_matrix = dist_mat_dict[distance_measure]
                    report_dict[algorithm][distance_measure] = {}
                    
                    ## Search Space
                    if self.k_list is None: 
                        self.k_list = range(2, int(len(dist_matrix)/3))
                    parameter_list = dbscan_find_optimal_epsilon_and_min_samples(dist_matrix, self.k_list)
                    for eps, min_sample_list in parameter_list:
                        for min_sample in min_sample_list:
                            model, labels = dbscan_detection(dist_matrix, eps, min_sample)
                            score = evaluate(algorithm, distance_measure, dist_matrix, labels)
                            report_dict[algorithm][distance_measure][f"eps = {eps}, min_samples = {min_sample}"] = {'eval_score': score, 'labels': labels}
        
        logger.info('Auto clustering completed')

        # Save Evaluation Report CSV 
        temp = pd.concat({k: pd.DataFrame(v).T for k, v in report_dict.items()}, axis=0)
        report_df = pd.DataFrame(temp.stack())
        report_df = report_df.reset_index()
        report_df.columns = ['algorithm','distance','hyperparameters', 'Values']
        report_df = pd.concat([report_df.drop(['Values'], axis=1), report_df['Values'].apply(pd.Series)], axis=1)
        report_df = report_df.sort_values(by=['eval_score','algorithm','distance','hyperparameters'], ascending = False)
        report_df.reset_index(inplace = True, drop = True)
        self.report_ = report_df

        self.clustering_labels_  = report_df['labels'][0]
        self.best_model_metric = {
            'Silhouette Score': round(report_df['eval_score'][0], 2)
        }
        self.best_model_params = {
            report_df['algorithm'][0]: {
                'params': report_df['hyperparameters'][0],
                'distance': report_df['distance'][0]
            }
        }

        logger.info(f'Best model: {self.best_model_params}')
        
        if self.verbose > 0:
            logger.debug("Silhouette Score of best model: ", round(report_df['eval_score'][0], 2))
        
        return self.clustering_labels_

    def plot(self, X):
        """ Plots the following visualisations:

        1) Plots the overall daily profiles for each cluster
        2) Plots the days of each cluster on the entire time series

        Parameters
        ----------
        X : pd.DataFrame shape (n_samples, n_features)
            Daily Time Series Dataframe
        """
        if not hasattr(self, 'clustering_labels_'):
            raise ValueError("AutomatedClustering has not been fitted. Use fit_predict(X) function before using plot(X)")
        
        if not (type(X) == pd.core.frame.DataFrame):
            raise ValueError("Supplied X is not a dataframe. X needs to be a daily time series dataframe of shape (n_samples: timestamp, n_features: day)")

        ## Check if labels and daily TS match
        if len(X.T) != len(self.clustering_labels_):
            raise ValueError(f"Number of days in X and labels do not have the same number of days. X has {len(X.T)} days but labels has {len(self.clustering_labels_)} samples")

        ## Associate color for each cluster for plotting
        unique_clusters = np.unique(self.clustering_labels_)
        color = cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        colour_dict = {}
        for idx, cluster in enumerate(unique_clusters):
            colour_dict[cluster] = color[idx]

        label_df = pd.DataFrame({'day':X.columns, 'cluster': self.clustering_labels_ })
        
        ## Plots
        plot_overall_daily_profiles(X, label_df, colour_dict)

        df = convert_daily_time_series_into_univariate(X)
        plot_detected_days(df, label_df, colour_dict)