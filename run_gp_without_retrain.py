import os
import training_gp_without_retrain

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
rRates = {'cov1':99.,'cov2': 99.,'fc1': 99,'fc2':99.}
cRates = {'cov1':0.,'cov2':0.,'fc1': 0.,'fc2': 0.}
f_name = compute_file_name(cRates)
learning_rate = 1e-4
print(f_name)
parent_dir = 'assets_no_prune/'
while (cRates['fc1'] <= 5.):

    # Prune
    param = [
    ('-thresholds',cRates),
    ('-file_name',f_name),
    ('-train', False),
    ('-prune',True),
    ('-profile',False),
    ('-parent_dir', parent_dir),
    ('-recover_rate', rRates),
    ('-learning_rate', learning_rate),
    ('-first_read', True)
    ]
    # _ = training_gp_without_retrain.main(param)

    # compute the name
    f_name = compute_file_name(cRates)

    # Profile and redefine mask
    param = [
    ('-thresholds',cRates),
    ('-file_name',f_name),
    ('-train', False),
    ('-prune',False),
    ('-profile', True),
    ('-parent_dir', parent_dir),
    ('-recover_rate', rRates),
    ('-learning_rate', learning_rate)
    ]
    # _ = training_gp_without_retrain.main(param)

    # # Profile and redefine mask
    param = [
    ('-thresholds',cRates),
    ('-file_name',f_name),
    ('-train', True),
    ('-prune',False),
    ('-profile', False),
    ('-parent_dir', parent_dir),
    ('-recover_rate', rRates),
    ('-learning_rate', learning_rate)
    ]
    # _ = training_gp_without_retrain.main(param)
    # TEST
    param = [
    ('-thresholds',cRates),
    ('-file_name',f_name),
    ('-train', False),
    ('-prune',False),
    ('-profile', False),
    ('-parent_dir', parent_dir),
    ('-recover_rate', rRates),
    ('-learning_rate', learning_rate)
    ]

    (acc,prune_percent) = training_gp_without_retrain.main(param)

    acc_list.append((acc,prune_percent))

    cRates['cov1'] = cRates['cov1'] + .2
    cRates['cov2'] = cRates['cov2'] + .2
    cRates['fc1'] = cRates['fc1'] + .2
    cRates['fc2'] = cRates['fc2'] + .2
print(acc_list)
