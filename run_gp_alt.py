import os
import training_gp_alt

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(p['cov1'] * 10))
    name += 'cov' + str(int(p['cov2'] * 10))
    name += 'fc' + str(int(p['fc1'] * 10))
    name += 'fc' + str(int(p['fc2'] * 10))
    return name

acc_list = []
run = 1
rRates = {'cov1':0.,'cov2':0.,'fc1': 0.,'fc2':0.}
cRates = {'cov1':0.,'cov2':0.,'fc1': 0.,'fc2':0.}
f_name = compute_file_name(cRates)
learning_rate = 1e-4
print(f_name)
while (cRates['fc1'] <= 5.):
    iter_cnt = 0
    # cRates['cov2'] = cRates['cov2'] + .1
    # rRates['cov2'] = rRates['cov2'] + .1
    cRates['fc1'] = cRates['fc1'] + 1.
    rRates['fc1'] = rRates['fc1'] + 1.
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
        _ = training_gp_alt.main(param)

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
        _ = training_gp_alt.main(param)

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
        _ = training_gp_alt.main(param)

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
        acc = training_gp_alt.main(param)
        if (acc >= 0.9936):
            break
        else:
            iter_cnt += 1

    acc_list.append(acc)
