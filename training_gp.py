from __future__ import print_function

# Import MNIST data
import sys
import getopt
import input_data
import os.path
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import scipy.io as sio
import random
import sys
# np.set_printoptions(threshold='nan')

class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

# Parameters
learning_rate = 1e-4
training_epochs = 300
# training_epochs = 2
batch_size = 128
display_step = 1

# Network Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

'''
pruning Parameters
'''
# sets the threshold
prune_threshold_cov = 0.08
prune_threshold_fc = 1
# Frequency in terms of number of training iterations
prune_freq = 100
ENABLE_PRUNING = 0


# Store layers weight & bias
# def initialize_tf_variables(first_time_training):
#     if (first_time_training):
def initialize_variables(parent_dir, model_number, weights_mask, rmask, profile = False, train = False):
    with open(parent_dir+ model_number +'.pkl','rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    if (profile == True):
        wc1 = wc1 * weights_mask['cov1']
        wc2 = wc2 * weights_mask['cov2']
        wd1 = wd1 * weights_mask['fc1']
        out = out * weights_mask['fc2']
        wc1 = wc1.astype(np.float32)
        wc2 = wc2.astype(np.float32)
        wd1 = wd1.astype(np.float32)
        out = out.astype(np.float32)
    elif (train == True):
        wc1 = wc1 * (1 - rmask['cov1'])
        wc2 = wc2 * (1 - rmask['cov2'])
        wd1 = wd1 * (1 - rmask['fc1'])
        out = out * (1 - rmask['fc2'])
        wc1 = wc1.astype(np.float32)
        wc2 = wc2.astype(np.float32)
        wd1 = wd1.astype(np.float32)
        out = out.astype(np.float32)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': tf.Variable(wc1,tf.float32),
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': tf.Variable(wc2,tf.float32),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': tf.Variable(wd1,tf.float32),
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': tf.Variable(out,tf.float32)
    }

    biases = {
        'cov1': tf.Variable(bc1,tf.float32),
        'cov2': tf.Variable(bc2,tf.float32),
        'fc1': tf.Variable(bd1,tf.float32),
        'fc2': tf.Variable(bout,tf.float32)
    }
    return (weights, biases)
# weights = {
#     'cov1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1)),
#     'cov2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
#     'fc1': tf.Variable(tf.random_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 1024])),
#     'fc2': tf.Variable(tf.random_normal([1024, NUM_LABELS]))
# }
# biases = {
#     'cov1': tf.Variable(tf.random_normal([32])),
#     'cov2': tf.Variable(tf.random_normal([64])),
#     'fc1': tf.Variable(tf.random_normal([1024])),
#     'fc2': tf.Variable(tf.random_normal([10]))
# }
#
#store the masks
# weights_mask = {
#     'cov1': tf.Variable(tf.ones([5, 5, NUM_CHANNELS, 32]), trainable = False),
#     'cov2': tf.Variable(tf.ones([5, 5, 32, 64]), trainable = False),
#     'fc1': tf.Variable(tf.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]), trainable = False),
#     'fc2': tf.Variable(tf.ones([512, NUM_LABELS]), trainable = False)
# }
    # else:
    #     with open('assets.pkl','rb') as f:
    #         (weights, biases, weights_mask) = pickle.load(f)

# weights_mask = {
#     'cov1': np.ones([5, 5, NUM_CHANNELS, 32]),
#     'cov2': np.ones([5, 5, 32, 64]),
#     'fc1': np.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]),
#     'fc2': np.ones([512, NUM_LABELS])
# }
# Create model
def conv_network(x, weights, biases, keep_prob):
    conv = tf.nn.conv2d(x,
                        weights['cov1'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov1']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')

    conv = tf.nn.conv2d(pool,
                        weights['cov2'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov2']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')
    '''get pool shape'''
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    hidden = tf.nn.dropout(hidden, keep_prob)
    output = tf.matmul(hidden, weights['fc2']) + biases['fc2']
    return output , reshape

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

'''
Prune weights, weights that has absolute value lower than the
threshold is set to 0
'''

# now i dont have recover precent, instead, fetch from profiled info
def dynamic_surgery(weight, pruning_th, recover_percent):
    threshold = np.percentile(np.abs(weight),pruning_th)
    weight_mask = np.abs(weight) > threshold
    recover_counts = int(np.sum(1 - weight_mask) * recover_percent)
    soft_weight_mask = (1 - weight_mask) * (np.random.rand(*weight.shape) > (1-recover_percent))
    return (weight_mask, soft_weight_mask)

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(p['cov1'] * 10))
    name += 'cov' + str(int(p['cov2'] * 10))
    name += 'fc' + str(int(p['fc1'] * 10))
    name += 'fc' + str(int(p['fc2'] * 10))
    return name

def prune_weights(weights, biases, org_masks, cRates, parent_dir):
    keys = ['cov1','cov2','fc1','fc2']
    new_mask = {}
    for key in keys:
        w_eval = weights[key].eval()
        threshold_off = 0.9*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        threshold_on = 1.1*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        # elements at this postion becomes zeros
        mask_off = np.abs(w_eval) < threshold_off
        # elements at this postion becomes ones
        mask_on = np.abs(w_eval) > threshold_on
        new_mask[key] = np.logical_or(((1 - mask_off) * org_masks[key]),mask_on).astype(int)
    file_name = compute_file_name(cRates)
    mask_file_name = parent_dir+'masks/'+ 'mask' + file_name +'.pkl'
    print("training done, save a mask file at "  + mask_file_name)
    with open(mask_file_name, 'wb') as f:
        pickle.dump(new_mask, f)
    mask_info(new_mask)

    weight_name = parent_dir + 'weights/' + 'weightpt'+ file_name +'.pkl'
    print("Pruning done, drop weights to {}".format(file_name))
    with open(weight_name, 'wb') as f:
        pickle.dump((
            weights['cov1'].eval(),
            weights['cov2'].eval(),
            weights['fc1'].eval(),
            weights['fc2'].eval(),
            biases['cov1'].eval(),
            biases['cov2'].eval(),
            biases['fc1'].eval(),
            biases['fc2'].eval()),f)
#
# def prune_weights(prune_thresholds, weights, weight_mask, biases, biases_mask, parent_dir, recover_percent):
#     keys = ['cov1','cov2','fc1','fc2']
#     next_threshold = {}
#     b_threshold = {}
#     soft_weight_mask = {}
#     soft_biase_mask = {}
#     for key in keys:
#         weight = weights[key].eval()
#         biase = biases[key].eval()
#         weight_mask[key], soft_weight_mask[key] = dynamic_surgery(weight, prune_thresholds[key], recover_percent)
#         biases_mask[key], soft_biase_mask[key] = dynamic_surgery(biase, prune_thresholds[key], recover_percent)
#         if (key == "fc1"):
#             print("-"*70)
#             print("Testing my ds")
#             print(np.array_equal(weight, weight_mask[key]))
#             (non_zeros, total) = calculate_non_zero_weights(weight_mask['fc1'])
#             print('fc1 hard mask {}'.format(non_zeros))
#             (non_zeros, total) = calculate_non_zero_weights(soft_weight_mask['fc1'])
#             print('fc1 soft mask {}'.format(non_zeros))
#             print("test end")
#             print("-"*70)
#     f_name = compute_file_name(prune_thresholds)
#     mask_file_name = parent_dir + 'mask/' + 'mask' +f_name + '.pkl'
#     print("pruning done, save a mask file at "  + mask_file_name)
#     with open(mask_file_name, 'wb') as f:
#         pickle.dump((weight_mask, biases_mask, soft_weight_mask, soft_biase_mask), f)
#     print("pruning done, save a weight file as well")
#     save_weights(weights, biases, parent_dir, f_name)
#
'''
mask gradients, for weights that are pruned, stop its backprop
'''
def mask_gradients(weights, grads_and_names, weight_masks):
    new_grads = []
    grad_values = []
    keys = ['cov1','cov2','fc1','fc2']
    for grad, var_name in grads_and_names:
        # flag set if found a match
        flag = 0
        index = 0
        for key in keys:
            if (weights[key]== var_name):
                # print('hi')
                new_grads.append((grad,var_name))
                grad_values.append(grad)
                flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append((grad,var_name))
            # grad_values.append(grad)
        # print(grad.get_shape())
    return (new_grads,grad_values)

'''
plot weights and store the fig
'''
def plot_weights(weights,pruning_info):
        keys = ['cov1','cov2','fc1','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval().flatten()
            # print (weight)
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            # print (weight)
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def roll_index(prob_list):
    index = 0
    rsum = 0
    shape = np.shape(prob_list)
    random_val = random.random()
    # prob_flatten = prob_list.flatten()
    for index, prob in np.ndenumerate(prob_list):
        rsum += prob
        if (random_val < rsum):
            s_index = index
            break
    return s_index

def recover_mask_gen(gradients, recover_rate):
    mask = np.zeros(gradients.shape)
    prob = gradients / float(np.sum(gradients))
    N_count = int(np.sum(gradients != 0) * recover_rate)
    run = 1
    i = 0
    index_list = []
    while (run):
        index = roll_index(prob)
        if (not(index in index_list)):
            index_list.append(index)
            mask[index] = 1
            i += 1
        if (i >= N_count):
            run = 0
    return mask


def recover_weights(weights_mask, grad_probs, recover_rates):
    keys = ['cov1','cov2','fc1','fc2']
    mask_info(weights_mask)
    prev = weights_mask['fc1']
    recover_mask = {}

    for key in keys:
        threshold = np.percentile(np.abs(grad_probs[key]),recover_rates[key])
        recover_mask[key] = np.abs(grad_probs[key]) > (threshold)
        recover_mask[key].astype(int)
    mask_info(recover_mask)
    return (recover_mask)
'''
Define a training strategy
'''
def main(argv = None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            # opts, args = getopt.getopt(argv[1:],'hp:tc1:tc2:tfc1:tfc2:')
            opts = argv
            keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']
            prune_thresholds = {}
            TRAIN = True
            PRUNE_ONLY = False
            parent_dir = './'
            first_read = False
            for key in keys:
                prune_thresholds[key] = 0.
            for item in opts:
                opt = item[0]
                val = item[1]
                if (opt == '-thresholds'):
                    cRates = val
                if (opt == '-file_name'):
                    file_name = val
                if (opt == '-train'):
                    TRAIN = val
                if (opt == '-prune'):
                    PRUNE_ONLY= val
                if (opt == '-profile'):
                    PROFILE = val
                if (opt == '-parent_dir'):
                    parent_dir = val
                if (opt == '-activate_rate'):
                    reactivate_rate = val
                if (opt == '-recover_rate'):
                    recover_rates = val
                if (opt == '-first_read'):
                    first_read = val
                if (opt == '-learning_rate'):
                    learning_rate = val
            print('pruning percentage for cov and fc are {}'.format(cRates))
        except getopt.error, msg:
            raise Usage(msg)

        # obtain all weight masks
        mask_file = parent_dir + 'masks/'+'mask'+ file_name+'.pkl'
        rmask_file = parent_dir + 'masks/'+'rmask'+ file_name+'.pkl'
        r_mask = {}
        if (PROFILE == True):
            with open(mask_file,'rb') as f:
                weights_mask = pickle.load(f)
        elif (TRAIN == True):
            with open(mask_file,'rb') as f:
                hard_mask = pickle.load(f)
            with open(rmask_file, 'rb') as f:
                r_mask = pickle.load(f)
            weights_mask = {}
            for key in keys:
                weights_mask[key] = np.logical_or(hard_mask, r_mask)
        else:
            weights_mask = {
                'cov1': np.ones([5, 5, NUM_CHANNELS, 20]),
                'cov2': np.ones([5, 5, 20, 50]),
                'fc1': np.ones([4 * 4 * 50, 500]),
                'fc2': np.ones([500, NUM_LABELS])
            }
            biases_mask = {
                'cov1': np.ones([20]),
                'cov2': np.ones([50]),
                'fc1': np.ones([500]),
                'fc2': np.ones([10])
            }

        mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])

        keep_prob = tf.placeholder(tf.float32)
        keys = ['cov1','cov2','fc1','fc2']

        x_image = tf.reshape(x,[-1,28,28,1])
        if (TRAIN == True):
            print("check r mask")
            (weights, biases) = initialize_variables(parent_dir + 'weights/', 'weightpt'+file_name, weights_mask, r_mask, PROFILE, TRAIN)
        elif (PROFILE == True):
            (weights, biases) = initialize_variables(parent_dir + 'weights/', 'weightpt'+file_name, weights_mask, r_mask, PROFILE, TRAIN)
        elif (PRUNE_ONLY == True):
            print(first_read)
            if (first_read == True):
                print(file_name)
                (weights, biases) = initialize_variables(parent_dir + 'weights/', 'weight'+file_name, weights_mask, r_mask, PROFILE, TRAIN)
            else:
                rfile_name = compute_file_name(cRates)
                (weights, biases) = initialize_variables(parent_dir + 'weights/', 'weight'+rfile_name, weights_mask, r_mask, PROFILE, TRAIN)
        else:
            (weights, biases) = initialize_variables(parent_dir + 'weights/', 'weight'+file_name, weights_mask, r_mask, PROFILE, TRAIN)

        # Construct model

        keys = ['cov1','cov2','fc1','fc2']
        new_weights = {}
        if (PRUNE_ONLY or PROFILE):
            for key in keys:
                new_weights[key] = weights[key]
        elif (TRAIN):
            for key in keys:
                weights_mask[key].astype(np.float32)
                new_weights[key] = weights[key] * weights_mask[key]
        else:
            for key in keys:
                new_weights[key] = weights[key] * weights_mask[key]

        pred, pool = conv_network(x_image, new_weights, biases, keep_prob)

        # Define loss and optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        org_grads = trainer.compute_gradients(cost, gate_gradients = trainer.GATE_OP)

        org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]
        (new_grads,grad_values) = mask_gradients(new_weights, org_grads, weights_mask)

        train_step = trainer.apply_gradients(new_grads)


        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            keys = ['cov1','cov2','fc1','fc2']


            prune_info(weights,1)
            # Training cycle
            training_cnt = 0
            pruning_cnt = 0
            train_accuracy = 0
            accuracy_list = np.zeros(30)
            accuracy_mean = 0
            c = 0
            train_accuracy = 0
            keys = ['cov1','cov2','fc1','fc2']

            if (PROFILE == True):
                print("profile for pruning...")
                prune_info(weights, 0)
                print("starts profile")

                total_batch = int(mnist.train.num_examples/batch_size)
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    _, fetched_grads = sess.run([train_step, grad_values], feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                keep_prob: 1.})
                    if (i == 0):
                        # print to log info
                        index = 0
                        collect_grads = {}
                        for key in keys:
                            collect_grads[key] = fetched_grads[index]
                            index = index + 1
                    else:
                        index = 0
                        for key in keys:
                            collect_grads[key] = fetched_grads[index] + collect_grads[key]
                            index = index + 1
                # weights inicated 1 in the hard mask can never be recovered
                grad_mask_val = {}
                keys = ['cov1','cov2','fc1','fc2']
                for key in keys:
                    grad_mask_val[key] = np.multiply (collect_grads[key],(1 - weights_mask[key]))
                non_zeros,size =calculate_non_zero_weights(1-weights_mask['cov2'])
                # print(weights_mask['cov2'].shape)
                print("profile done")
                prune_info(weights, 0)
                print('my grads')
                non_zeros,size =calculate_non_zero_weights(collect_grads['cov2'])
                print(non_zeros)
                # print(collect_grads['cov2'].shape)

                print('my masked grads')
                non_zeros,size =calculate_non_zero_weights(grad_mask_val['cov2'])
                print(non_zeros)
                print(grad_mask_val['fc1'].shape)

                print(collect_grads['fc1'])
                print(grad_mask_val['fc1'])

                recover_mask = recover_weights(weights_mask, grad_mask_val, recover_rates)
                print(file_name)
                with open(parent_dir + 'masks/' + 'rmask' + file_name + '.pkl','wb') as f:
                    pickle.dump(recover_mask, f)


            if (TRAIN == True):
                print('Training starts ...')
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    total_batch = int(mnist.train.num_examples/batch_size)
                    # Loop over all batches
                    for i in range(total_batch):
                        # execute a pruning
                        batch_x, batch_y = mnist.train.next_batch(batch_size)
                        _ = sess.run(train_step, feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                keep_prob: 0.8})
                        training_cnt = training_cnt + 1
                        if (training_cnt % 10 == 0):
                            [c, train_accuracy] = sess.run([cost, accuracy], feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                keep_prob: 1.})
                            accuracy_list = np.concatenate((np.array([train_accuracy]),accuracy_list[0:29]))
                            accuracy_mean = np.mean(accuracy_list)
                            if (training_cnt % 1000 == 0):
                                print(cRates)
                                print('accuracy mean is {}'.format(accuracy_mean))
                                print('Epoch is {}'.format(epoch))
                                weights_info(training_cnt, c, train_accuracy, accuracy_mean)
                                prune_info(new_weights, 0)
                                print('org weights')
                                prune_info(weights, 0)
                        # if (training_cnt == 10):
                        if (accuracy_mean > 0.99 or epoch > 300):
                            accuracy_list = np.zeros(30)
                            accuracy_mean = 0
                            print('Training ends')
                            test_accuracy = accuracy.eval({
                                    x: mnist.test.images[:],
                                    y: mnist.test.labels[:],
                                    keep_prob: 1.})
                            print('test accuracy is {}'.format(test_accuracy))
                            if (test_accuracy > 0.9936 or epoch > 300):
                                save_weights(weights, biases, parent_dir, file_name)
                                return test_accuracy
                            else:
                                pass
                        with open('log/data0118.txt',"a") as output_file:
                    		output_file.write("{},{},{}\n".format(training_cnt,train_accuracy, c))
                        # Compute average loss
                        avg_cost += c / total_batch
                    # Display logs per epoch step
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                print("Optimization Finished!")
                # Test model
            if (TRAIN == True):
                save_weights(weights, biases, parent_dir, file_name)
            if (PRUNE_ONLY == True):
                prune_weights(weights, biases, weights_mask, cRates, parent_dir)
                # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob : 1.0})
            print("Accuracy:", test_accuracy)
            with open('acc_log_10.txt','a') as f:
                f.write(str(test_accuracy)+'\n')
            return test_accuracy
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2

def save_weights(weights, biases, parent_dir, file_name):
    with open(parent_dir + 'weights/' + 'weight'+ file_name + '.pkl', 'wb') as f:
        pickle.dump((
            weights['cov1'].eval(),
            weights['cov2'].eval(),
            weights['fc1'].eval(),
            weights['fc2'].eval(),
            biases['cov1'].eval(),
            biases['cov2'].eval(),
            biases['fc1'].eval(),
            biases['fc2'].eval()),f)
def weights_info(iter,  c, train_accuracy, acc_mean):
    print('This is the {}th iteration, cost is {}, accuracy is {}, accuracy mean is {}'.format(
        iter,
        c,
        train_accuracy,
        acc_mean
    ))

def prune_info(weights, counting):
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('take fc1 as example, {} nonzeros, in total {} weights'.format(non_zeros, total))

def mask_info(weights):
    (non_zeros, total) = calculate_non_zero_weights(weights['cov1'])
    print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['cov2'])
    print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc1'])
    print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc2'])
    print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))

def write_numpy_to_file(data, file_name):
    # Write the array to disk
    with file(file_name, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        for data_slice in data:
            for data_slice_two in data_slice:
                np.savetxt(outfile, data_slice_two)
                outfile.write('# New slice\n')


if __name__ == '__main__':
    sys.exit(main())
