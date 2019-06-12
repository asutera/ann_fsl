import tensorflow as tf
import numpy as np
from threading import Thread
from queue import Queue
import pickle

class DataProcessor():
    def __init__(self, data, answers, queue_max = 10):
        self.data = data
        self.answers = answers
        self.q_max = queue_max

    def process_random(self, batch_size):
        nbr_samples = self.data.shape[0]
        indexes = np.array([np.random.randint(nbr_samples) for _ in range(batch_size)])
        self.to_send = [self.data[indexes], self.answers[indexes]]\

    def produce_random(self, batch_size, nbr_batch):
        self.q = Queue(maxsize = self.q_max)
        while (nbr_batch != 0):
            self.process_random(batch_size)
            self.q.put(self.to_send, block = True)
            nbr_batch -= 1

    def produce_ordered(self, batch_size, nbr_batch):
        self.q = Queue(maxsize = self.q_max)
        start_index = 0
        while (nbr_batch != 0):
            if (start_index + batch_size < np.array(self.data).shape[0]):
                self.q.put([self.data[start_index:start_index+batch_size],
                            self.answers[start_index:start_index+batch_size]])
                start_index = start_index + batch_size
            else:
                remaining = batch_size - (np.array(self.data).shape[0] - start_index)
                data_n = np.concatenate([self.data[start_index:], self.data[:remaining]], axis = 0)
                answers_n = np.concatenate([self.answers[start_index:], self.answers[:remaining]], axis = 0)
                self.q.put([data_n, answers_n])
                start_index = remaining
            nbr_batch -= 1


    def start_producing(self, batch_size, nbr_batch, random = True):
        if random:
            self.t = Thread(target = self.produce_random, args = (batch_size, nbr_batch))
        else:
            self.t = Thread(target = self.produce_ordered, args = (batch_size, nbr_batch))
        self.t.start()

    def get_batch(self):
        tmp = self.q.get(block = True)
        return tmp[0], tmp[1]

    def check_stop_producing(self):
        self.t.join()


class Network:
    def __init__(self, layer_sizes = None, file_to_load = None, invisible_layer = True, problem_type = 'regression'):
        if not file_to_load == None:
            self.load_model(file_to_load)
            return
        else:
            self.to_pickle = {}
            self.to_pickle['invisible_layer'] = invisible_layer
            self.to_pickle['dropout'] = 0.5
            self.to_pickle['problem_type'] = problem_type
            self.create_network(layer_sizes)

    def create_network(self, layer_sizes):
        self.operations_dict = {}
        input_size = layer_sizes[0]
        output_size = layer_sizes[-1]

        self.operations_dict['weights'] = []
        self.operations_dict['bias'] = []
        self.operations_dict['after_dropout'] = []
        self.operations_dict['relu_output'] = []
        self.operations_dict['inputs'] = []
        self.operations_dict['outputs'] = []
        self.operations_dict['keep_prob'] = []
        self.operations_dict['cost'] = []
        self.operations_dict['training_step'] = []
        self.operations_dict['predict'] = []
        self.operations_dict['alpha'] = []
        self.operations_dict['il_weights'] = []

        self.operations_dict['inputs'].append(tf.placeholder(tf.float32, shape=(None, input_size)))
        tf.add_to_collection('inputs', self.operations_dict['inputs'][-1])
        self.operations_dict['outputs'].append(tf.placeholder(tf.float32, shape=(None, output_size)))
        tf.add_to_collection('outputs', self.operations_dict['outputs'][-1])
        self.operations_dict['keep_prob'].append(tf.placeholder(tf.float32))
        tf.add_to_collection('keep_prob', self.operations_dict['keep_prob'][-1])
        self.operations_dict['alpha'].append(tf.placeholder(tf.float32))
        tf.add_to_collection('alpha', self.operations_dict['alpha'][-1])

        previous_inputs = self.operations_dict['inputs'][-1]
        # Create one to one connected layer
        if self.to_pickle['invisible_layer']:
            self.operations_dict['il_weights'].append(tf.Variable(tf.constant(shape = [input_size], value = 1.0)))
            tf.add_to_collection('il_weights', self.operations_dict['il_weights'][-1])
            previous_inputs = tf.multiply(previous_inputs, self.operations_dict['il_weights'][-1])

        # create the invisible layers
        for i in range(1,len(layer_sizes)-1):
            current_weights = self.weight_tensor(shape=(layer_sizes[i-1], layer_sizes[i]))
            current_bias = self.bias_tensor(shape=[layer_sizes[i]])
            current_relu = self.create_neuron(previous_inputs, current_weights, current_bias)
            self.operations_dict['after_dropout'].append(tf.nn.dropout(current_relu, keep_prob=self.operations_dict['keep_prob'][-1]))
            tf.add_to_collection('after_dropout', self.operations_dict['after_dropout'][-1])
            previous_inputs = self.operations_dict['after_dropout'][-1]

        final_weights = self.weight_tensor(shape = [layer_sizes[-2], layer_sizes[-1]])
        final_bias = self.bias_tensor(shape=[layer_sizes[-1]])
        previous_inputs = tf.add(tf.matmul(previous_inputs, final_weights),final_bias)

        if self.to_pickle['problem_type'] == 'regression':
            cost_mse = tf.reduce_mean(tf.square(self.operations_dict['outputs'][-1] - previous_inputs))
            self.operations_dict['score'] = [cost_mse]
            tf.add_to_collection('score', cost_mse)
            self.operations_dict['network_answer'] = [previous_inputs]
            tf.add_to_collection('network_answer', previous_inputs)
        else:
            after_softmax = tf.nn.softmax(previous_inputs)
            cost_mse = -tf.reduce_mean(self.operations_dict['outputs'][-1]
                                       *tf.log(tf.clip_by_value(after_softmax,1e-10,1.0)))
            prediction = tf.arg_max(after_softmax, 1)
            good_predictions = tf.equal(tf.argmax(self.operations_dict['outputs'][-1], 1), prediction)
            self.operations_dict['score'] = [tf.reduce_mean(tf.cast(good_predictions, tf.float32))]
            self.operations_dict['network_answer'] = [prediction]
            tf.add_to_collection('network_answer', prediction)

        self.operations_dict['predict'].append(previous_inputs)
        tf.add_to_collection('predict', self.operations_dict['predict'][-1])

        if self.to_pickle['invisible_layer']:
            cost_il = tf.reduce_mean(tf.abs(self.operations_dict['il_weights'][-1]))
        else:
            cost_il = 0
        self.operations_dict['cost_loss'] = [cost_mse]
        tf.add_to_collection('cost_loss', cost_mse)
        self.operations_dict['cost'].append(cost_mse + self.operations_dict['alpha'][-1] * cost_il)
        tf.add_to_collection('cost', self.operations_dict['cost'][-1])
        self.operations_dict['training_step'].append(tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(self.operations_dict['cost'][-1]))
        tf.add_to_collection('training_step', self.operations_dict['training_step'][-1])

        # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def fit(self, data, answers, nbr_iteration, batch_size = 50, val_data = [], val_answers = [], alpha = 1.0):
        self.to_pickle['train_info'] = {}
        self.to_pickle['train_info']['accuracy'] = []
        self.to_pickle['train_info']['training_step'] = []
        self.to_pickle['train_info']['aupr'] = []

        data_processor = DataProcessor(data, answers)
        data_processor.start_producing(batch_size, nbr_iteration)
        current_display_step = 1000
        if len(val_data) == 0:
            val_data = data[:batch_size]
            val_answers = answers[:batch_size]
        validation_dict = {self.operations_dict['inputs'][-1]:val_data[:100],
                           self.operations_dict['outputs'][-1]:val_answers[:100],
                           self.operations_dict['keep_prob'][-1]:1.0,
                           self.operations_dict['alpha'][-1]:alpha}
        for i in range(nbr_iteration):
            batch_data, batch_answers = data_processor.get_batch()
            dict = {self.operations_dict['inputs'][-1]:batch_data, self.operations_dict['outputs'][-1]:batch_answers,
                    self.operations_dict['keep_prob'][-1]:self.to_pickle['dropout'],
                    self.operations_dict['alpha'][-1]:alpha}
            self.sess.run(self.operations_dict['training_step'][-1], feed_dict=dict)
            if i % current_display_step == 0 or i == nbr_iteration-1:
                dict[self.operations_dict['keep_prob'][-1]] = 1.0
                cur_score = self.sess.run(self.operations_dict['score'][-1], feed_dict=validation_dict)
                print ('iteration ' ,i ,', training/validation score ',
                       self.sess.run(self.operations_dict['score'][-1], feed_dict=dict), '/',
                       cur_score)
                if i % (current_display_step * 10) == 0 and i != 0:
                    current_display_step *= 10
        data_processor.check_stop_producing()
        validation_dict = {self.operations_dict['inputs'][-1]:val_data,
                           self.operations_dict['outputs'][-1]:val_answers,
                           self.operations_dict['keep_prob'][-1]:1.0,
                           self.operations_dict['alpha'][-1]:alpha}
        print('final validation score : ', self.sess.run(self.operations_dict['score'][-1], feed_dict=validation_dict))

    def score(self, data, answers, batch_size = 50):
        data_giver = DataProcessor(data, answers)
        iteration_number = int(np.ceil(np.array(data).shape[0]/batch_size))
        data_giver.start_producing(batch_size = batch_size, nbr_batch = iteration_number)
        accuracies = []
        for i in range(iteration_number):
            batch_data, batch_ans = data_giver.get_batch()
            accuracies.append(self.sess.run(self.operations_dict['score'][-1],
                                            feed_dict={self.operations_dict['inputs'][-1]:batch_data,
                                                       self.operations_dict['outputs'][-1]:batch_ans,
                                                       self.operations_dict['keep_prob'][-1]:1.0}))
        return np.mean(accuracies)

    def cost(self, data, answers, batch_size = 50):
        data_giver = DataProcessor(data, answers)
        iteration_number = int(np.ceil(np.array(data).shape[0]/batch_size))
        data_giver.start_producing(batch_size = batch_size, nbr_batch = iteration_number)
        accuracies = []
        for i in range(iteration_number):
            batch_data, batch_ans = data_giver.get_batch()
            accuracies.append(self.sess.run(self.operations_dict['cost_loss'][-1],
                                            feed_dict={self.operations_dict['inputs'][-1]:batch_data,
                                                       self.operations_dict['outputs'][-1]:batch_ans,
                                                       self.operations_dict['keep_prob'][-1]:1.0}))
        return np.mean(accuracies)

    def save_model(self, file):
        saver = tf.train.Saver()
        saver.save(self.sess, file)
        pickle.dump(self.to_pickle, open(file + '.pickle', 'wb'))

    def load_model(self, file):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(file + '.meta')
        saver.restore(self.sess, file)
        self.to_pickle = pickle.load(open(file + '.pickle', 'rb'))

        self.operations_dict = {}
        for op in tf.get_default_graph().get_all_collection_keys():
            self.operations_dict[op] = tf.get_collection(op)

    def create_neuron(self, inputs, weight_tensor, bias_tensor):
        x = tf.nn.relu(tf.add(tf.matmul(inputs,weight_tensor),bias_tensor))
        self.operations_dict['relu_output'].append(x)
        tf.add_to_collection('relu_output', x)
        return x

    def weight_tensor(self, shape, std = 0.1):
        initial = tf.Variable(tf.truncated_normal(shape = shape, stddev = std))
        tf.add_to_collection('weights', initial)
        self.operations_dict['weights'].append(initial)
        return initial

    def bias_tensor(self, shape, init = 0.1):
        initial = tf.Variable(tf.constant(shape = shape, value = 0.1))
        tf.add_to_collection('bias', initial)
        self.operations_dict['bias'].append(initial)
        return initial

    def close(self):
        self.sess.close()
        tf.reset_default_graph()


class ImportanceNetwork(Network):
    def __init__(self, layer_sizes = None, file_to_load = None, invisible_layer = True, problem_type = 'regression'):
        super(ImportanceNetwork, self).__init__(layer_sizes, file_to_load, invisible_layer, problem_type)
        self.create_importance_metric_graph()

    def compute_layer_importances(self, data, outputs, batch_size = 50, compute_everything = False, type = 'deriv'):
        self.to_pickle['computed_importances'] = []
        if type == 'deriv':
            to_run = 'importances_delta_deriv'
        elif type == 'old':
            to_run = 'full_layers_importances'
        else:
            to_run = 'importances_delta_activ'
        # First compute importances for one output
        iteration_number = int(np.ceil(np.array(data).shape[0]/batch_size))
        data_giver = DataProcessor(data, data)
        first = True
        for o in range(np.array(outputs).shape[0]):
            if compute_everything:
                analysed_output = outputs
            else:
                analysed_output = np.zeros_like(outputs, dtype=np.float32)
                analysed_output[o] = outputs[o]
            if not np.sum(analysed_output) == 0:
                data_giver.start_producing(50, iteration_number, random = False)
                for i in range(iteration_number):
                    batch_data, batch_answers = data_giver.get_batch()
                    dict = {self.operations_dict['inputs'][-1]:batch_data,
                            self.operations_dict['keep_prob'][-1]:1.0,
                            }
                    if type != 'old':
                        dict[self.operations_dict['output_to_analyse'][-1]] = analysed_output
                    imps = self.sess.run(self.operations_dict[to_run][0], feed_dict=dict)
                    #print (imps)
                    if first:
                        self.to_pickle['computed_importances'] = imps
                        first = False
                    else:
                        self.to_pickle['computed_importances'] = \
                            np.sum([imps, self.to_pickle['computed_importances']], axis = 0)
                data_giver.check_stop_producing()
            if compute_everything:
                break
        return self.to_pickle['computed_importances']

    def create_importance_metric_graph(self):
        self.operations_dict['output_to_analyse'] = \
            [tf.placeholder(tf.float32, shape = (self.operations_dict['outputs'][-1].get_shape())[1])]
        tf.add_to_collection('output_to_analyse', self.operations_dict['output_to_analyse'][-1])
        self.create_activ_graph()
        self.create_deriv_graph()

    def get_il_importances(self):
        if self.to_pickle['invisible_layer']:
            return self.sess.run(tf.abs(self.operations_dict['il_weights'][-1]))
        else:
            return -1

    def get_deriv_importances(self, data, outputs, batch_size = 50):
        return self.compute_layer_importances(data, outputs, batch_size=batch_size, type = 'deriv')

    def get_activ_importances(self, data, outputs, batch_size = 50):
        return self.compute_layer_importances(data, outputs, batch_size = batch_size, compute_everything = True,
                                              type = 'activ')


    def create_deriv_graph(self):
        weights = self.operations_dict['weights']
        neurons_out = self.operations_dict['relu_output']
        self.operations_dict['importances_delta_deriv'] = []
        for i in range(len(weights)):
            if i == 0:
                previous_importances = tf.multiply(tf.ones_like(self.operations_dict['predict'][-1],tf.float32),
                                                   self.operations_dict['output_to_analyse'][-1])
            current_weights = weights[-1-i]
            previous_importances_adjusted_with_weights = tf.matmul(previous_importances, tf.transpose(current_weights))
            if not i == len(weights)-1:
                activated_neurons = tf.where(neurons_out[-1-i] > 0, tf.ones_like(neurons_out[-1-i]),
                                             tf.zeros_like(neurons_out[-1-i]))
            else:
                activated_neurons = tf.ones_like(self.operations_dict['inputs'][-1])
                if self.to_pickle['invisible_layer']:
                    activated_neurons = tf.multiply(activated_neurons, tf.abs(self.operations_dict['il_weights'][-1]))
            current_importances = tf.multiply(previous_importances_adjusted_with_weights, activated_neurons)
            importances_delta = tf.reduce_sum(tf.abs(current_importances), axis = 0)
            self.operations_dict['importances_delta_deriv'].append(importances_delta)
            previous_importances = current_importances

        self.operations_dict['importances_delta_deriv'].reverse()
        for op in self.operations_dict['importances_delta_deriv']:
            tf.add_to_collection('importances_delta_deriv', op)

    def create_activ_graph(self):
        weights = self.operations_dict['weights']
        neurons_out = self.operations_dict['relu_output']
        self.operations_dict['importances_delta_activ'] = []
        for i in range(len(weights)):
            if i == 0:
                previous_importances = tf.multiply(tf.abs(self.operations_dict['predict'][-1]),
                                                   self.operations_dict['output_to_analyse'][-1])
            if not i == len(weights)-1:
                current_input = neurons_out[-i-1]
            else:
                current_input = self.operations_dict['inputs'][-1]
                if self.to_pickle['invisible_layer']:
                    current_input = tf.multiply(current_input, self.operations_dict['il_weights'][-1])
            transposed_weights = tf.transpose(weights[-i-1])
            expanded_input = tf.expand_dims(current_input, axis = 1)
            times = tf.multiply(expanded_input, transposed_weights)
            times_with_function = tf.nn.relu(times)
            totals = tf.reduce_sum(times_with_function, axis = 2)

            map = tf.nn.relu(tf.multiply(expanded_input, transposed_weights))
            transposed_map = tf.transpose(map, [0, 2, 1])
            adjusted_transposed_map = tf.multiply(transposed_map, tf.expand_dims(previous_importances, axis = 1))

            expanded_totals = tf.tile(tf.expand_dims(totals, axis = 1),
                                      [1, adjusted_transposed_map.get_shape()[1].value,1])
            adjusted_divided_transposed_map = tf.where(tf.less(expanded_totals, 1e-10), tf.zeros_like(expanded_totals),
                                                       tf.divide(adjusted_transposed_map, expanded_totals))
            layer_importances = tf.reduce_sum(adjusted_divided_transposed_map, axis = 2)
            importances_delta = tf.reduce_sum(layer_importances, axis = 0)
            previous_importances = layer_importances
            self.operations_dict['importances_delta_activ'].append(importances_delta)

        self.operations_dict['importances_delta_activ'].reverse()
        for op in self.operations_dict['importances_delta_activ']:
            tf.add_to_collection('importances_delta_activ', op)

    def create_neurone(self, inputs, weights, bias):
        to_mult = tf.matmul(inputs, weights) + bias
        return tf.nn.relu(to_mult)

    def create_neurone_mul(self, inputs, weights, bias):
        to_mult = tf.multiply(inputs, weights) + bias
        return tf.nn.relu(to_mult)