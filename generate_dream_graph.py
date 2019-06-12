import pickle
import os
import numpy as np

import pickle
from scripts.helpers import aupr
from scripts.helpers.dream_reader import dream_data
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

t = ["gradient", 'lrp', 'selec', 'rf']
t_n = ["SL+GRAD", "SL+LRP", "SL", "RF"]
c = ['green', 'orange', 'brown', 'red']
aupr_rf = [0.155, 0.153, 0.225, 0.208, 0.199]
for dream_nbr in range(1,6):
    dataset_loader = dream_data(dream_nbr = dream_nbr, path_to_data = "./DREAM4")
    adj = np.transpose(dataset_loader.adjacence_matrix)
    for architecture in [[50, 25], [75,50], [75,50,25]]:
        f = plt.figure()
        ticks = []
        handles = []
        auprs = {el:[] for el in t}
        xs = [10.0, 30.0, 100.0]
        ys = [10000.0, 30000.0, 100000.0]
            
        xs = xs * len(ys)
        ys = [ys[0]] * len(ys) + [ys[1]] * len(ys) + [ys[2]] * len(ys)

        xs = [10.0, 30.0, 100.0, 0.0, 0.0, 0.0] + xs
        ys = [0.0, 0.0, 0.0, 10000.0, 30000.0, 100000.0] + ys
        for a, a2 in zip(xs, ys):
            #for a2 in [0.0]:
            print (architecture, " ", a, " ", a2)
            ticks.append(r'$\alpha_1$:' + '%.0E' % Decimal(str(a)) + "\n" + 
                         r'$\alpha_2$:'+'%.0E' % Decimal(str(a2)))
            ticks[-1] =  ticks[-1].replace("+0", "")
            
            r_d1 = pickle.load(open('dream_full_aupr/retrain_architecture' + str(architecture) 
                                    + "_" + str(a) + "_" + str(a2) + str(dream_nbr), 'rb'))

            for cur_t in t:
                if cur_t == 'rf':
                    aupr_ = aupr_rf[dream_nbr-1]
                else:
                    m = r_d1[cur_t]
                    aupr_ =  aupr.compute_gene_area(m, adj)
                auprs[cur_t].append(aupr_)

        locs, labels = plt.xticks(np.arange(15), ticks, fontsize = 6)
        for cur_t, c_ in zip(t, c):
            x, = plt.plot(np.arange(3), auprs[cur_t][:3], c = c_)
            x, = plt.plot(np.arange(3, 6), auprs[cur_t][3:6], c = c_)
            x, = plt.plot(np.arange(6, 15), auprs[cur_t][6:], c = c_)
            handles.append(x)

        for i in [2.5, 5.5]:
            plt.axvline(x=i, c="black")

        plt.ylim([0.0, 0.185])
        locs, labels = plt.yticks()
        plt.text(1-0.25, 0.165, "L1", bbox=dict(facecolor='red', alpha=0.3))
        plt.text(4-0.25, 0.165, "L2", bbox=dict(facecolor='red', alpha=0.3))
        plt.text(7-0.25, 0.165, "L1L2", bbox=dict(facecolor='red', alpha=0.3))
        plt.ylabel("AUPR")

        plt.legend(handles, t_n)
        plt.savefig("figures/retrain_l1_" + str(dream_nbr) + "_" + str(architecture) + ".pdf")
                    

