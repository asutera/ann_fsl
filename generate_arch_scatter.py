import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from decimal import Decimal
t = "classification"
t_ = "nl"
build_selec = True
# Generate figure
for t3 in ["classification"]:
    for t6 in ['1']:
        def get_mean_std(regul, n_layers):
            auprs, perfs = {'gradient':[],'lrp':[],'selec':[]}, []
            for i in range(5):
                best_score = np.infty
                a2 = 0.0
                best_d = None
                for a in [10.0, 100.0, 1000.0]:
                    if a == 0.0 and a2 == 0.0:
                        n = t + "_" + t_ + "_5000_" + str(False) + "_none_" + str(i) + "_" + str(a) + "_" + str(a2) + "_" + str(n_layers)
                    else:
                        n = t + "_" + t_ + "_5000_" + str(build_selec) + "_" + regul + "_" + str(i) + "_" + str(a) + "_" + str(a2) + "_" + str(n_layers)
                    d = pickle.load(open("./ArtificialResults/" + n, "rb"))
                    if a == 0.0 and a2 == 0.0:
                        d['aupr']['selec'] = d['aupr']['lrp']
                    if d['score'] < best_score:
                        best_d = d
                        best_score = d['score']
                d = best_d
                for el in d['aupr']:
                    auprs[el].append(1-d['aupr'][el])
                perfs.append(d['score'])
            to_return_means_aupr = {el: auprs[el] for el in auprs}
            to_return_std_aupr = {el: np.std(auprs[el]) for el in auprs}

            to_return_means_perf = perfs
            to_return_std_perf = np.std(perfs)
            to_return = {'means':{'auprs':to_return_means_aupr, 'perf':to_return_means_perf},
                         'std': {'auprs': to_return_std_aupr, 'perf': to_return_std_perf}}

            return to_return
 
        handles = []
        types = ['selec', 'lrp', 'gradient']
        for tmp, tmp_n in zip(types, ["SL", "SL+LRP", "SL+GRAD"]):
            f = plt.figure()
            grad = []
            xs = []
            handles = []
            for n_l in [1,2,3,4,5,6]:
                d = get_mean_std("l1l2", n_l)
                grad.append(d['means']['auprs'][tmp])
                xs.append(d['means']['perf'])
            for i in range(5):
               handles.append( plt.scatter([el[i] for el in xs], [el[i] for el in grad], s=100))
               for j, txt in enumerate([el[i] for el in xs]):
                    plt.annotate(j+1, (xs[j][i], grad[j][i]), ha='center', va='center')
            
            plt.xlabel("Error rate")
            plt.ylabel("(1-AUPR)")
            perf_name = "Error rate" if t == "classification" else "MSE"
            plt.legend(handles, ["Dataset {}".format(el) for el in [1,2,3,4,5]])
            plt.title(tmp_n)
            plt.savefig("figures/dataset_scatter_" + t + "_" + t_ + tmp + ".pdf")

