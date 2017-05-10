import os
import training_gp

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(p['cov1'] * 10))
    name += 'cov' + str(int(p['cov2'] * 10))
    name += 'fc' + str(int(p['fc1'] * 10))
    name += 'fc' + str(int(p['fc2'] * 10))
    return name

acc_list = []
run = 1
# rRates = {'cov1':0.,'cov2':4.,'fc1': 6.,'fc2':0.}
# cRates = {'cov1':0.,'cov2':1.5,'fc1': 4.4,'fc2':0.}
rRates = {'cov1':90.,'cov2': 95.,'fc1': 99.9,'fc2':90.}
cRates = {'cov1':0.,'cov2':0.,'fc1': 4.4,'fc2': 0.}
f_name = compute_file_name(cRates)
learning_rate = 1e-4
print(f_name)
while (cRates['cov2'] <= 2.):
    iter_cnt = 0
    # cRates['cov2'] = cRates['cov2'] + .1
    cRates['cov2'] = cRates['cov2'] + 0.1
    # cRates['cov1'] = cRates['cov1'] + 0.2
    # cRates['fc2'] = cRates['fc2'] + 0.2
    # rRates['fc1'] = rRates['fc1'] + 0.
    while (iter_cnt < 7):
        # Prune
        if (iter_cnt > 4):
            learning_rate = 1e-5
        else:
            learning_rate = 1e-4
        param = [
        ('-thresholds',cRates),
        ('-file_name',f_name),
        ('-train', False),
        ('-prune',True),
        ('-profile',False),
        ('-parent_dir', './'),
        ('-recover_rate', rRates),
        ('-first_read', iter_cnt == 0),
        ('-learning_rate', learning_rate)
        ]
        _ = training_gp.main(param)

        # compute the name
        f_name = compute_file_name(cRates)

        # Profile and redefine mask
        param = [
        ('-thresholds',cRates),
        ('-file_name',f_name),
        ('-train', False),
        ('-prune',False),
        ('-profile', True),
        ('-parent_dir', './'),
        ('-recover_rate', rRates),
        ('-learning_rate', learning_rate)
        ]
        _ = training_gp.main(param)

        # TRAIN
        param = [
        ('-thresholds',cRates),
        ('-file_name',f_name),
        ('-train', True),
        ('-prune',False),
        ('-profile', False),
        ('-parent_dir', './'),
        ('-recover_rate', rRates),
        ('-learning_rate', learning_rate)
        ]
        _ = training_gp.main(param)

        # TEST
        param = [
        ('-thresholds',cRates),
        ('-file_name',f_name),
        ('-train', False),
        ('-prune',False),
        ('-profile', False),
        ('-parent_dir', './'),
        ('-recover_rate', rRates),
        ('-learning_rate', learning_rate)
        ]
        acc = training_gp.main(param)
        if (acc >= 0.9936):
            break
        else:
            iter_cnt += 1

    acc_list.append(acc)
