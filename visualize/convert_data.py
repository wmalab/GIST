import torch

"""
index from dataset
prediction[index] = {'structures': pred_X, 
                    'structures_weights':weights,
                    'predict_cluster': [pred_dist_cluster_mat, pdcm_list], 
                    'predict_distance': [pred_dist_mat, pdm_list],
                    'true_cluster': true_cluster_mat}
prediction['mixture model'] = dis_gmm
"""

def load_prediction():
    pass