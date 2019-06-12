import numpy as np
import pickle

methods = ["none", "l1l2"]

with open("global_results_features.tex", "w") as f:
    t = "classification"
    t_ = "nl"
    for n_features in [50, 2500, 10000]:
        for t3_ in ["l"]:
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
            to_write_perf_std = []
            for method in methods:
                build_selec = True
                possible_alphas = [0.0, 10.0, 100.0, 1000.0]
                possible_alphas2 = [0.0]
                if method == "none":
                    build_selec = False
                    possible_alphas = [0.0]
                    possible_alphas2 = [0.0]

                best_params = []
                my_range = range(5)
                for test in my_range:
                    best_a = 0
                    best_a2 = 0
                    best_loss = np.infty
                    best_final_d = None
                    for a in possible_alphas:
                        for a2 in possible_alphas2:
                            method_nn = method
                            if method == "l1l2":
                                if a == 0.0:
                                    method_nn = "l2"
                                if a2 == 0.0:
                                    method_nn = "l1"
                                if a == 0.0 and a2 == 0.0:
                                    break
                                method_nn = "l1l2"
                            d_name = "./ArtificialResults/" + t + "_" + t_ + "_" + str(n_features) + "_" + str(build_selec) + "_" + \
                                method_nn + "_" + str(test) + "_" + str(a) + "_" + str(a2)
                            print (d_name)
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
                    to_write_std.append(0.0)

                if t == "classification":
                    perfs = [el['d']['score'] for el in best_params]
                else:
                    perfs = [el['d']['score'] for el in best_params]

                to_write_perf_mean.append(np.mean(perfs))
                to_write_perf_std.append(np.std(perfs))

            best_rf = []
            for test in range(5):
                if t == "regression":
                    best_perf = np.infty
                else:
                    best_perf = 0.0
                for K in ['sqrt', 'log2', int(n_features / 3), int(n_features / 2)]:
                    current_rf_name = "./ArtificialResults/" + t + "_" + t_ + "_" + str(n_features) +"_RF_" + str(K) + "_" + str(test)
                    d = pickle.load(open(current_rf_name, "rb"))
                    if d['results'] > best_perf and t == "classification":
                        best_perf = d['results']
                        best_d = d
                    if d['results'] < best_perf and t == "regression":
                        best_perf = d['results']
                        best_d = d
                best_rf.append(best_d)

            to_write_mean.append(np.mean([el['aupr'] for el in best_rf]))
            to_write_std.append(np.std([el['aupr'] for el in best_rf]))
            to_write_perf_mean.append(np.mean([el['results'] for el in best_rf]))
            to_write_perf_std.append(np.std([el['results'] for el in best_rf]))

            types = ["grad", "lrp", "selec"]
            for mn in methods:
                for type_name in types:
                    f.write(" & " + type_name)

            f.write("\\\\ \\hline \n")

            f.write("AUPR ")
            for mean, std in zip(to_write_mean, to_write_std):

                if not mean == np.max(to_write_mean):
                    f.write(" & " + "{0:.3f}".format(mean) + " +- " + "{0:.3f}".format(std))
                else:
                    f.write(" & " + "\\textbf{" + "{0:.3f}".format(mean) + " +- " + "{0:.3f}".format(std) + "}")
            f.write("\\\\ \\hline \n")

            f.write("PERF ")
            for perf, std in zip(to_write_perf_mean[:-1], to_write_perf_std[:-1]):
                f.write(
                    " & " + "\\multicolumn{3}{||c||}{" + "{0:.3f}".format(perf) + " +- " + "{0:.3f}".format(std) + "}")

            f.write(" & " +" {" + "{0:.3f}".format(to_write_perf_mean[-1]) + " +- " + "{0:.3f}".format(to_write_perf_std[-1]) + "}")
            f.write("\\\\ \\hline \n")
            f.write("")
            f.write("\\end{tabular}\\caption{" + t + "\\_" + t_ +"}\\end{table}\n")
