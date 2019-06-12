import pickle
from scripts.helpers import aupr
from scripts.helpers.dream_reader import dream_data
import numpy as np
import matplotlib.pyplot as plt

dream_nbr = 3
dataset_loader = dream_data(dream_nbr = dream_nbr, path_to_data = "./DREAM4")


r_d1 = pickle.load(open('DreamResults/architecture_75_50_25_'+str(dream_nbr), 'rb'))
types = ['gradient', 'lrp', 'selec', 'rf']
#types = ['deriv']

adj = np.transpose(dataset_loader.adjacence_matrix)
# adj = dataset_loader.adjacence_matrix

auprs = {}
max_nn = []
max_tot = []
for t in types:
    auprs[t] = []

for t in types:
   r_d1[t] = (r_d1[t] - np.min(r_d1[t]))/(np.max(r_d1[t]) - np.min(r_d1[t]))

for l in range(100):
    f = plt.figure()
    grad_ax = f.add_subplot(221)
    lrp_ax = f.add_subplot(222)
    selec_ax = f.add_subplot(223)
    rf_ax = f.add_subplot(224)

    axes = [grad_ax, lrp_ax, selec_ax, rf_ax]

    cur_max_nn = 0.0
    cur_max_tot = 0.0
    as_ = []
    for t, ax in zip(types, axes):
        m = r_d1[t]
        sums = np.sum(m, axis = 1)
        sums[np.where(sums==0)] = 1
        # m = m/sums
        as_.append(m)
        a_ = aupr.get_aupr(m[l], adj[l])
        ax.title.set_text(t + " " + str(a_))
        ax.hist(m[l])
        if not np.isnan(a_):
            auprs[t].append(a_)
            if t in ["gradient", "lrp", "selec"] and a_ > cur_max_nn:
                cur_max_nn = a_
            if a_ > cur_max_tot:
                cur_max_tot = a_

    if not np.isnan(a_):
        max_nn.append(cur_max_nn)
        max_tot.append(cur_max_tot)

    # plt.show()
    print ('SCORE OF : ', aupr.compute_gene_area(np.max([r_d1[t] for t in ["gradient", "rf"]], axis=0), adj))

for t in types:
    print ("--------------------------")
    print (t)
    print (np.mean(auprs[t]))
    print (aupr.compute_gene_area(r_d1[t], adj))

print ("BEST OF ALL")
print (np.mean(max_nn))
print (np.mean(max_tot))