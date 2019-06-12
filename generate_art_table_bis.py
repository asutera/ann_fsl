import numpy as np
import pickle

methods = ["none", "l1", "l2", "l1l2"]

with open("global_results.tex", "w") as f:
    for t in ["classification", "regression"]:
        for t_ in ["l", "nl"]:
            f.write("\\begin{table}\\begin{tabular}")
            f.write("{|c||")
            for el in methods:
                f.write("c|c|c||")
            f.write("c||}")
            f.write("\\hline \n")
            print("############# Problem : ", t, " ", t_, "##################")
            f.write("SCORE")
            for mn in methods:
                f.write(" & \\multicolumn{3}{||c||}{" + mn + "}")
            f.write("& RF \\\\ \\hline \n")
            to_write_mean = []
            to_write_std = []
            to_write_perf_mean = []
            for method in methods:
                build_selec = True
                possible_alphas = [10.0, 100.0, 1000.0]
                possible_alphas2 = [10000.0, 100000.0, 1000000.0]
                if method == "none":
                    build_selec = False
                    possible_alphas = [0.0]
                    possible_alphas2 = [0.0]
                if method == "l1":
                    possible_alphas2 = [0.0]
                if method == "l2":
                    possible_alphas = [0.0]

                best_params = []
                my_range = range(5)
                for test in my_range:
                    best_a = 0
                    best_a2 = 0
                    best_loss = np.infty
                    best_final_d = None
                    for a in possible_alphas:
                        for a2 in possible_alphas2:
                            d_name = "./ArtificialResults/" + t + "_" + t_ + "_" + str(build_selec) + "_" + \
                                               method + "_" + str(test) + "_" + str(a) + "_" + str(a2)
                            d = pickle.load(open(d_name, "rb"))
                            if d['results']['performance_loss'] < best_loss:
                                best_a = a
                                best_a2 = a2
                                best_loss = d['results']['performance_loss']
                                best_final_d = d
                    best_params.append({'alpha':best_a, 'alpha2':best_a2, 'd':best_final_d})
                mean_aupr_deriv = np.mean([el['d']['aupr']['gradient'] for el in best_params])
                mean_aupr_activ = np.mean([el['d']['aupr']['lrp'] for el in best_params])
                to_write_mean.append(mean_aupr_deriv)
                to_write_mean.append(mean_aupr_activ)
                to_write_std.append(np.std([el['d']['aupr']['gradient'] for el in best_params]))
                to_write_std.append(np.std([el['d']['aupr']['lrp'] for el in best_params]))
                if build_selec:
                    mean_aupr_selec = np.mean([el['d']['aupr']['selec'] for el in best_params])
                    to_write_mean.append(mean_aupr_selec)
                    to_write_std.append(np.std([el['d']['aupr']['selec'] for el in best_params]))
                else:
                    to_write_mean.append(0.0)

                if t == "classification":
                    perfs = [el['d']['score'] for el in best_params]
                else:
                    perfs = [el['d']['score'] for el in best_params]

                to_write_perf_mean.append(np.mean(perfs))


            for test in range(5):
                best_rf = []
                if t == "regression":
                    best_perf = np.infty
                else:
                    best_perf = 0.0
                for K in ['sqrt', 'log2', int(5000 / 3), int(5000 / 2)]:
                    current_rf_name = "./ArtificialResults/" + t + "_" + t_ + "_RF_" + str(K) + "_" + str(test)
                    d = pickle.load(open(current_rf_name, "rb"))
                    if d['results'] > best_perf and t == "classification":
                        best_perf = d['results']
                        best_d = d
                    if d['results'] < best_perf and t == "regression":
                        best_perf = d['results']
                        best_d = d
                best_rf.append(d)
            types = ["grad", "lrp", "selec"]
            for mn in methods:
                for type_name in types:
                    f.write(" & " + type_name)

            f.write("\\\\ \\hline \n")

            f.write("AUPR ")
            for mean in to_write_mean:

                if not mean == np.max(to_write_mean):
                    f.write(" & " + "{0:.2f}".format(mean))
                else:
                    f.write(" & " + "\\textbf{" + "{0:.2f}".format(mean) + "}")
            f.write(" & " + str([el['aupr'] for el in best_rf]) + "\\\\ \\hline \n")

            f.write("PERF ")
            for perf in to_write_perf_mean:
                f.write(" & " + "\\multicolumn{3}{||c||}{" + "{0:.2f}".format(perf) + "}")


            f.write("&"+ str(np.mean([el['results'] for el in best_rf])) +"\\\\ \\hline \n")
            f.write("")
            f.write("\\end{tabular}\\caption{" + t + "\\_" + t_ +"}\\end{table}\n")
