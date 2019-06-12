import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from decimal import Decimal
t = "regression"
t_ = "l"
build_selec = True
# Generate figure
for t in ["regression", "classification"]:
    for t_ in ["l", "nl"]:

        def get_mean_std(regul, a, a2):
            if a == 0.0 and a2 == 0.0:
                regul = "none"
            elif a == 0.0:
                regul = "l2"
            elif a2 == 0.0:
                regul = "l1"

            auprs, perfs = {'gradient':[],'lrp':[],'selec':[]}, []
            for i in range(5):
                if a == 0.0 and a2 == 0.0:
                    n = t + "_" + t_ + "_" + str(False) + "_none_" + str(i) + "_" + str(a) + "_" + str(a2)
                else:
                    n = t + "_" + t_ + "_" + str(build_selec) + "_" + regul + "_" + str(i) + "_" + str(a) + "_" + str(a2)
                d = pickle.load(open("./ArtificialResults/" + n, "rb"))
                if a == 0.0 and a2 == 0.0:
                    d['aupr']['selec'] = d['aupr']['lrp']

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


        # Do L1
        for a2 in [0.0, 10000.0, 100000.0, 1000000.0]:
            alphas = [10.0, 100.0, 1000.0]
            regul = "l1l2" if not a2 == 0.0 else "l1"

            means = {'gradient':[], 'selec':[], 'lrp':[], 'perf':[]}
            for a in alphas:
                d = get_mean_std(regul, a, a2)
                for el in d['means']['auprs']:
                    means[el].append(d['means']['auprs'][el])
                means['perf'].append(d['means']['perf'])

            stds = {'gradient':[], 'selec':[], 'lrp':[], 'perf':[]}
            for a in alphas:
                d = get_mean_std(regul, a, a2)
                for el in d['std']['auprs']:
                    stds[el].append(d['std']['auprs'][el])
                stds['perf'].append(d['std']['perf'])

            f = plt.figure()
            p1 = f.add_subplot(111)
            handles = []
            names = [el + " (1-AUPR)" if not el == "perf" else el + " ER" for el in means]
            for el in means:
                if not el == "perf":
                    x = p1.errorbar(alphas, means[el], stds[el], linestyle = "--")
                else:
                    x = p1.errorbar(alphas, means[el], stds[el])
                x[-1][0].set_linestyle('-.')
                handles.append(x)
            plt.xlabel(r'$\alpha_1$')
            plt.legend(handles, names)

            plt.savefig("figures/L1_alpha_analysis_alphas2_" + str(a2) + ".pdf")

        # Do L2
        for a in [0.0, 10.0, 100.0, 1000.0]:
            alphas2 = [10000.0, 100000.0, 1000000.0]
            regul = "l1l2" if not a == 0.0 else "l2"

            means = {'gradient': [], 'selec': [], 'lrp': [], 'perf': []}
            for a2 in alphas2:
                d = get_mean_std(regul, a, a2)
                for el in d['means']['auprs']:
                    means[el].append(d['means']['auprs'][el])
                means['perf'].append(d['means']['perf'])

            stds = {'gradient': [], 'selec': [], 'lrp': [], 'perf': []}
            for a2 in alphas2:
                d = get_mean_std(regul, a, a2)
                for el in d['std']['auprs']:
                    stds[el].append(d['std']['auprs'][el])
                stds['perf'].append(d['std']['perf'])

            f = plt.figure()
            p1 = f.add_subplot(111)
            handles = []
            names = [el + " (1-AUPR)" if not el == "perf" else el + " ER" for el in means]
            for el in means:
                if not el == "perf":
                    x = p1.errorbar(alphas2, means[el], stds[el], linestyle="--")
                else:
                    x = p1.errorbar(alphas2, means[el], stds[el])
                x[-1][0].set_linestyle('-.')
                handles.append(x)
            plt.xlabel(r'$\alpha_2$')
            plt.legend(handles, names)

            plt.savefig("figures/L2_alpha2_analysis_alphas1_" + str(a) + ".pdf")



        xs = [10.0, 100.0, 1000.0]
        ys = [10000.0, 100000.0, 1000000.0]

        xs, ys = np.meshgrid(xs, ys)
        z = []
        grad = []

        for x, y in zip(xs.flatten(), ys.flatten()):
            d = get_mean_std("l1l2", x, y)
            z.append(d['means']['perf'])
            grad.append(d['means']['auprs']['selec'])

        print (z)
        z = np.array(z).reshape(xs.shape)

        f = plt.figure()
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(xs, ys, z)
        ax.set_xlabel(r'$\alpha_1$')
        ax.set_ylabel(r'$\alpha_2$')
        ax.set_zlabel('Error rate')
        plt.savefig('figures/surf_3d_l1l2_error_rate.pdf')

        for tmp in ['selec', 'lrp', 'gradient']:
            grad = []
            xs = [10.0, 100.0, 1000.0]
            ys = [10000.0, 100000.0, 1000000.0]

            xs, ys = np.meshgrid(xs, ys)

            for x, y in zip(xs.flatten(), ys.flatten()):
                d = get_mean_std("l1l2", x, y)
                grad.append(d['means']['auprs'][tmp])
            print (grad)
            grad = np.array(grad).reshape(xs.shape)

            f = plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(xs, ys, grad)
            ax.set_xlabel(r'$\alpha_1$')
            ax.set_ylabel(r'$\alpha_2$')
            ax.set_zlabel(tmp + ' (1-AUPR)')
            plt.savefig('figures/surf_3d_l1l2_' + tmp + '_aupr.pdf')









        f = plt.figure()
        handles = []
        types = ['selec', 'lrp', 'gradient', 'perf']
        styles = ['--', '-.', ':', '']
        colors = ['green', 'orange', 'brown', 'red']
        for tmp, sty, clr in zip(types, styles, colors):
            grad = []
            std = []
            xs = [10.0, 100.0, 1000.0]
            ys = [10000.0, 100000.0, 1000000.0]
            
            xs = xs * len(ys)
            ys = [ys[0]] * len(ys) + [ys[1]] * len(ys) + [ys[2]] * len(ys)

            xs = [10.0, 100.0, 1000.0, 0.0, 0.0, 0.0] + xs
            ys = [0.0, 0.0, 0.0, 10000.0, 100000.0, 1000000.0] + ys
            # xs, ys = np.meshgrid(xs, ys)
        
            ticks = []
            for x, y in zip(xs, ys):
                if x == 0.0 and y == 0.0:
                    continue
                ticks.append(r'$\alpha_1$:' + '%.0E' % Decimal(str(x)) + "\n" + r'$\alpha_2$:'+'%.0E' % Decimal(str(y)))
                ticks[-1] =  ticks[-1].replace("+0", "")
            locs, labels = plt.xticks()
            plt.xticks(np.arange(len(xs)), ticks, fontsize = 6)

            for x, y in zip(xs, ys):
                if x == 0.0 and y == 0.0:
                    continue
                d = get_mean_std("l1l2", x, y)
                if not tmp == "perf":
                    grad.append(d['means']['auprs'][tmp])
                    std.append(d['std']['auprs'][tmp])
                else:
                    grad.append(d['means'][tmp])
                    std.append(d['std'][tmp])

            for s,e in zip([0,3,6],[3,6,len(xs)]):
                if not tmp == "perf":
                    x = plt.errorbar(np.arange(s,e), grad[s:e], std[s:e], linestyle=sty, c= clr)
                else:
                    x = plt.errorbar(np.arange(s,e), grad[s:e], std[s:e], c=clr)
                x[-1][0].set_linestyle('-.')
            
            handles.append(x)

        #for i, i_prime, c in zip([-0.5, 2.5, 5.5], [2.5, 5.5, 14.5], [(0.5,0,0), 'green', 'blue']):
        #        plt.axvspan(i, i_prime, facecolor=c, alpha=0.5)
        for i in [2.5, 5.5]:
            plt.axvline(x=i, c="black")

        locs, labels = plt.yticks()
        plt.text(1, (locs[-3] + locs[-2])/2, "L1", bbox=dict(facecolor='red', alpha=0.3))
        plt.text(4, (locs[-3] + locs[-2])/2, "L2", bbox=dict(facecolor='red', alpha=0.3))
        plt.text(7, (locs[-3] + locs[-2])/2, "L1L2", bbox=dict(facecolor='red', alpha=0.3))
        plt.xlim([-0.5, 14.5])
        perf_name = "Error rate" if t == "classification" else "MSE"
        plt.legend(handles, [el_n + " (1-AUPR)" if not el == "perf" else perf_name for el, el_n in zip(types, ["SL", "SL+LRP", "SL+GRAD", perf_name])])
        plt.savefig("figures/" + t + "_" + t_ + "all_in_one.pdf")




