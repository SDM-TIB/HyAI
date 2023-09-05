import os, sys, glob
from os import listdir
from os.path import isfile, join
from shutil import copyfile


def get_measure(cls_folder, f_model):
    current_path = os.path.dirname(os.path.realpath(__file__))
    measure = '/cma ' + cls_folder + '/clusters/ ' + f_model + '/donor.txt ' + f_model + '/matrix_sim.txt > ' + cls_folder+'/'+f_model+'.txt'
    print(current_path)
    print('commd: ', current_path + measure)
    os.system(current_path + measure)


threshold = [45, 47, 50, 52, 55, 57, 60, 62, 65, 67, 70, 72, 75, 77, 80]

for th in threshold:
    cls_address = 'SemEP_' + str(th)
    cls_address_metis = 'METIS_' + str(th)
    cls_address_kmeans = 'Kmeans_' + str(th)

    """Compute Cluster-Measures"""
    get_measure('relationalData/'+cls_address, 'relationalData')
    get_measure('relationalData/'+cls_address_metis, 'relationalData')
    get_measure('relationalData/' + cls_address_kmeans, 'relationalData')

"""Execute Kmeans"""

"""Compute Cluster-Measures"""
# get_measure('relationalData/Kmeans', 'relationalData')

