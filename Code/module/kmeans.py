import numpy as np
import matplotlib.pyplot as plt
import random

class K_means(object):
    def __init__(self):
        # need to rewrite the way to get data!!!
        self.data = None

    def kmeans(self, k):
        centroids = self.centroid_picker(k)
        centroids_prev = None
        counter =0
        dt_cluster_assignment=None
        while True:
            if np.array_equal(centroids_prev, centroids):
                break
            centroids_prev = centroids
            dt_cluster_assignment = self.assign_centroid(centroids)
            centroids = self.update_cetroid(k, dt_cluster_assignment)
            counter+=1
​
        return centroids,dt_cluster_assignment
​
​
​
    def assign_centroid(self, centroids):
        dis_data_2_centroids = np.zeros(shape=(len(centroids),np.size(self.data,0)))
        for i in range(0,len(centroids)):
            squared = np.square(centroids[i] - self.data)
            summed = np.sum(squared, axis=1)
            sqrted = np.sqrt(summed)
            dis_data_2_centroids[i] = sqrted
​
        dt_cluster_assignment= np.argmin(dis_data_2_centroids.transpose(), axis=1)
        return dt_cluster_assignment
​
​
    def update_cetroid(self,k, dt_cluster_assignment):
        new_centroids = []
        for i in range(0,k):
            data_belonging_to_i = self.data[np.where(dt_cluster_assignment == i)]
            new_centroid = np.mean(data_belonging_to_i, axis=0)
            new_centroids.append(np.array(new_centroid))
        return new_centroids
​
​
    def compute_pot_func(self, centroids, dt_cluster_assignment):
        fc = 0
        for i in range(0,len(centroids)):
            data_belonging_to_i = self.data[np.where(dt_cluster_assignment == i)]
            squared = np.square(centroids[i] - data_belonging_to_i)
            summed = np.sum(squared)
            fc += summed
        return fc
​
​
    def centroid_picker(self, k):
        centroids = []
        for i in range(0,k):
            randindex = random.randint(0,np.size(self.data,0))
            centroids.append(np.array(self.data[randindex]))
        return centroids
​
​
​
    def plt_curve(self, ks,fc):
        plt.title('K V.S Potential Function value')
        plt.plot(ks, fc, 'b')
        plt.ylabel('Value of Potential Function')
        plt.xlabel('K')
        plt.show()
​
ks = [2,3,4,5,6,7,8]
f_c = []
obj = K_means()
​
for k in ks:
    centriods, cluster_assignment = obj.kmeans(k)
    f_c.append(obj.compute_pot_func(centriods, cluster_assignment))
obj.plt_curve(ks,f_c)
