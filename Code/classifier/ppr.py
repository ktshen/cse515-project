from .classifier import Classifier
import numpy as np
from collections import Counter
class PPR(Classifier):
    def __init__(self, img_sim_graph, image_list, lk=50 ,alpha = 0.8):
        self.transition = np.zeros((len(img_sim_graph)+1,len(img_sim_graph)+1))
        self.tele =  np.zeros((len(img_sim_graph)+1,1))
        self.alpha = alpha
        self.lk = lk

        self.image_list = image_list

        for key,item in img_sim_graph.items():
            transit_index = image_list.index(key)
            sim_weight_lst = item['sim_weight']
            normalized_sim_weight_lst = sim_weight_lst/np.sum(sim_weight_lst)
            sim_node_lst = item['sim_node_index']
            self.transition[transit_index][sim_node_lst] = [normalized_sim_weight_lst]


    def get_steady_state(self,query_list):
        self.transition = self.transition.T

        for query in query_list:
            self.tele[self.image_list.index(query)] = 1/len(query_list)

        self.pi_final = np.dot(np.linalg.inv(np.eye(len(self.tele)) - self.alpha * self.transition),
                               (1 - self.alpha) * self.tele)
        self.pi_final = self.pi_final.reshape((1, -1))
        ind = self.pi_final[0].argsort()[-self.lk:][::-1]
        return ind

    def fit(self, data= None, gt = None):
        self.train_data = data
        self.gt = gt

    def predict(self, data):

        result_list = []

        for each_test_data in data:
            self.transition[-1,:] = 0
            self.transition[:,-1] = 0

            img_img_sim = []
            for tran_dd  in self.train_data:
                img_img_sim.append(np.dot(each_test_data, tran_dd.T))
            img_img_sim.append(np.dot(each_test_data, each_test_data.T))


            a = np.array(img_img_sim)
            ind = a.argsort()[-20:][::-1]
            normalized_sim_weight_lst = a[ind] / np.sum(a[ind])

            self.transition[-1][ind] = [normalized_sim_weight_lst]

            for i, index in enumerate(ind):
                self.transition[index][-1] = normalized_sim_weight_lst[i]

            self.transition = self.transition.T

            self.tele[-1] = 1

            self.pi_final = np.dot(np.linalg.inv(np.eye(len(self.tele)) - self.alpha * self.transition),
                                   (1 - self.alpha) * self.tele)
            self.pi_final = self.pi_final.reshape((1, -1))
            ind = self.pi_final[0].argsort()[-self.lk:][::-1]

            if ind[0]==100:
                c = Counter(np.array(self.gt)[ind[1:]])
                result_list.append(c.most_common(1)[0][0])
            else:
                c = Counter(np.array(self.gt)[ind[:-1]])
                result_list.append(c.most_common(1)[0][0])
        return result_list

















