#!/usr/bin/env python3
import dream_reader
import derivative_network
import numpy as np
import aupr
import pickle

# il = False
il = True

for dream_data in [1,2,3,4,5]:
    data_giver = dream_reader.dream_data(dream_data)

    genes = data_giver.get_full_dataset()

    deriv_importances = np.zeros(shape = (100,100))
    il_importances = np.zeros(shape = (100,100))
    activ_importances = np.zeros(shape = (100,100))

    # validate to get best architecture

    best_accuracy = 100000
    parameters_dict = []
    best_alpha = 0
    best_neurons = 50
    best_layer = 2
    training_iter_per_layer = 10000
    # training_iter_per_layer = 1
    # for nbr_layers in [2,3]:
    for nbr_layers in [0]:
        # for neurons_per_layer in [50, 150]:
        for neurons_per_layer in [1]:
            for alpha in [0, 5, 60, 300, 800, 1500]:
            # for alpha in [0]:
                layer_sizes = [99]
                for i in range(nbr_layers):
                    layer_sizes.append(neurons_per_layer)
                layer_sizes.append(1)
                accuracies = []
                for i in range(5):
                # for i in range(1):
                    print ('########## FOLD ', i, ' ###########')
                    print (' For alpha ', alpha, ' neurons ', neurons_per_layer, ' layers ', nbr_layers)
                    for current_gene in range(i*20, (i+1)*20):
                        current_data = np.array([np.concatenate([x[:current_gene], x[current_gene + 1:]]) for x in genes])
                        current_answer = np.array([x[current_gene] for x in genes]).reshape((-1, 1))
                        train_data = np.copy(np.concatenate([current_data[:i*20], current_data[(i+1)*20:]]))
                        train_answers = np.copy(np.concatenate([current_answer[:i*20], current_answer[(i+1)*20:]]))
                        test_data = current_data[i * 20:(i + 1) * 20]
                        test_answers = current_answer[i * 20:(i + 1) * 20]

                        model = derivative_network.ImportanceNetwork(layer_sizes=layer_sizes, problem_type = 'regression',
                                                                     invisible_layer=il)
                        model.fit(train_data, train_answers, alpha = alpha, val_data=test_data, val_answers=test_answers,
                                  nbr_iteration=training_iter_per_layer, batch_size=35)
                        accuracies.append(model.score(data=test_data, answers=test_answers))
                        model.close()
                parameters_dict.append({'nbr_layer': nbr_layers, 'neurons': neurons_per_layer, 'alpha': alpha,
                                        'avg_accuracy':np.mean(accuracies)})
                if (np.mean(accuracies) < best_accuracy):
                    print ('NEW BEST ARCHITECTURE')
                    best_accuracy = np.mean(accuracies)
                    best_alpha = alpha
                    best_neurons = neurons_per_layer
                    best_layer = nbr_layers
                print ('####################')
                print()
                print('Obtained average accuracy of ', np.mean(accuracies))
                pickle.dump(parameters_dict, open('./cv_dream_10/dream' + str(dream_data) + '_kfold_history', 'wb'))

    pickle.dump([best_alpha, best_layer, best_neurons, best_accuracy], open('./cv_dream_10/best_dream'
                                                                            + str(dream_data) + '_parameters', 'wb'))

    layer_sizes = [99]
    for i in range(best_layer):
        layer_sizes.append(best_neurons)
    layer_sizes.append(1)
    for i in range(100):
        print ('################## GENE ', i, ' ##################### ')
        model = derivative_network.ImportanceNetwork(layer_sizes=layer_sizes, problem_type='regression',
                                                     invisible_layer=il)
        current_data = np.array([np.concatenate([x[:i], x[i + 1:]]) for x in genes])
        current_answer = np.array([x[i] for x in genes]).reshape((-1, 1))
        model.fit(current_data, current_answer, val_data=current_data[:20],
                  val_answers=current_answer[:20], batch_size=35, nbr_iteration=training_iter_per_layer,
                  alpha = best_alpha)
        current_output = [1.0]
        tmp_deriv = model.get_deriv_importances(current_data, current_output)
        line = np.concatenate([tmp_deriv[:i], [0], tmp_deriv[i:]])
        deriv_importances[i] = line
        tmp_deriv = model.get_activ_importances(current_data, current_output)
        line = np.concatenate([tmp_deriv[:i], [0], tmp_deriv[i:]])
        activ_importances[i] = line
        if il:
            tmp_deriv = model.get_il_importances()
            line = np.concatenate([tmp_deriv[:i], [0], tmp_deriv[i:]])
        else:
            tmp_deriv = np.zeros(line.shape)
            line = tmp_deriv
        il_importances[i] = line
        print (line)
        model.close()




    aupr_activ = aupr.get_aupr(np.transpose(activ_importances), data_giver.adjacence_matrix)
    aupr_il = aupr.get_aupr(np.transpose(il_importances), data_giver.adjacence_matrix)
    aupr_deriv = aupr.get_aupr(np.transpose(deriv_importances), data_giver.adjacence_matrix)

    print (aupr_activ, aupr_deriv, aupr_il)

    pickle.dump([deriv_importances, aupr_deriv,
                 activ_importances, aupr_activ,
                 il_importances, aupr_il], open('./cv_dream_10/dream_' + str(dream_data) + '_best_architecture_results', 'wb'))