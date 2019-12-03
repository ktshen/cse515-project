import numpy as np
from cmath import nan

class KMeans(object):
    def __init__(self, n_clusters = 8, max_iter = 500, min_rimpr = 1e-4, rpt = 15):
        self.k = n_clusters
        self.max_iter = max_iter
        self.min_impr = min_rimpr
        self.rpt = rpt
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, data):
        best_labels = None
        best_centroids = None
        min_pot = float('inf')
        
        for _ in range(self.rpt):
            centroids = self.centroid_picker(data)
            centroids_prev = centroids
            labels = self.assign_centroid(centroids, data)
            centroids = self.update_cetroid(labels, data)           
            
            for _ in range(self.max_iter-1):
                if np.allclose(centroids_prev, centroids, rtol = self.min_impr):
                    break
                centroids_prev = centroids
                labels = self.assign_centroid(centroids, data)
                centroids = self.update_cetroid(labels, data)
                if len(centroids) < self.k:
                    break
            cur_pot = self.compute_pot_func(data, centroids, labels)
            if min_pot > cur_pot:
                min_pot = cur_pot
                best_labels = labels
                best_centroids = centroids
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
                
        return self

    def assign_centroid(self, centroids, data):
        dis_data_2_centroids = np.zeros(shape=(len(centroids),np.size(data,0)))
        for i in range(len(centroids)):
            squared = np.square(centroids[i] - data)
            summed = np.sum(squared, axis=1)
            dis_data_2_centroids[i] = np.sqrt(summed)
            
        dt_cluster_assignment = np.argmin(dis_data_2_centroids.transpose(), axis=1)
        return dt_cluster_assignment
    
    def update_cetroid(self, dt_cluster_assignment, data):
        new_centroids = []
        for i in range(self.k):
            data_belonging_to_i = data[dt_cluster_assignment == i]
            if data_belonging_to_i.size == 0:
                break
            new_centroid = np.mean(data_belonging_to_i, axis=0)
            new_centroids.append(np.array(new_centroid))
        return new_centroids
    
    def compute_pot_func(self, data, centroids, labels):
        fc = 0
        for i in range(len(centroids)):
            data_belonging_to_i = data[np.where(labels == i)]
            squared = np.square(centroids[i] - data_belonging_to_i)
            summed = np.sum(squared)
            fc += summed
        return fc
    
    def centroid_picker(self, data):
        
        randindex = np.random.permutation(data.shape[0])
        return data[randindex[:self.k]]
