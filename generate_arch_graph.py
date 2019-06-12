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
tmp_ = [4]
for t3 in ["classification"]:
    for t6 in ['1']:
        def get_mean_std(regul, n_layers):
            auprs, perfs = {'gradient':[],'lrp':[],'selec':[]}, []
            for i in tmp_:
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
            to_return_means_aupr = {el: np.mean(auprs[el]) for el in auprs}
            to_return_std_aupr = {el: np.std(auprs[el]) for el in auprs}

            to_return_means_perf = np.mean(perfs)
            to_return_std_perf = np.std(perfs)
            to_return = {'means':{'auprs':to_return_means_aupr, 'perf':to_return_means_perf},
                         'std': {'auprs': to_return_std_aupr, 'perf': to_return_std_perf}}

            return to_return
 
        f = plt.figure()
        handles = []
        types = ['selec', 'lrp', 'gradient', 'perf']
        styles = ['--', '-.', ':', '']
        colors = ['green', 'orange', 'brown', 'red']
        for tmp, sty, clr in zip(types, styles, colors):
            grad = []
            std = []
            for n_l in [1,2,3,4,5,6]:
                d = get_mean_std("l1l2", n_l)
                if not tmp == "perf":
                    grad.append(d['means']['auprs'][tmp])
                    std.append(d['std']['auprs'][tmp])
                else:
                    grad.append(d['means'][tmp])
                    std.append(d['std'][tmp])
                print (grad)
                    
            if not tmp == "perf":
                x = plt.errorbar(np.arange(1,7), grad, std, linestyle=sty, c= clr)
            else:
                x = plt.errorbar(np.arange(1,7), grad, std, c=clr)
            x[-1][0].set_linestyle('-.')
            
            handles.append(x)
        plt.xlabel("Number of hidden layers")
        perf_name = "Error rate" if t == "classification" else "MSE"
        plt.legend(handles, [el_n + " (1-AUPR)" if not el == "perf" else perf_name for el, el_n in zip(types, ["SL", "SL+LRP", "SL+GRAD", ""])])
        plt.savefig("figures/arch_" + t + "_" + t_ + "all_in_one" + str(tmp_) + ".pdf")

