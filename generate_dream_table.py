import pickle
import os
import numpy as np

import pickle
from scripts.helpers import aupr
from scripts.helpers.dream_reader import dream_data
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp




for dream_nbr in range(1,6):
    print ("#################### DREAM NBR", dream_nbr, " ##################")
    dataset_loader = dream_data(dream_nbr = dream_nbr, path_to_data = "./DREAM4")

    for arch in ["", "architecture_50_25_", "architecture_75_50_", "architecture_75_50_25_"]:
        print ("-----------", arch, "---------------")
        print (arch)
        r_d1 = pickle.load(open('DreamResults/' + arch + str(dream_nbr), 'rb'))
        types = ['gradient', 'lrp', 'selec', 'rf']

        adj = np.transpose(dataset_loader.adjacence_matrix)
        for t in types:
            m = r_d1[t]
            # sums = np.sum(m, axis=1)
            # sums[np.where(sums == 0)] = 1
            # m = m/sums
            print(t, " : ", aupr.compute_gene_area(m, adj))

