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
cRates = {'cov1':0.,'cov2':0.,'fc1':0.,'fc2':0.}
f_name = compute_file_name(cRates)
print(f_name)
while (run == 1):
    iter_cnt = 0
    cRates['fc1'] = cRates['fc1'] + 0.2
    while (iter_cnt < 7):
        # Prune
        param = [
        ('-thresholds',cRates),
        ('-file_name',f_name),
        ('-train', False),
        ('-prune',True),
        ('-profile',False),
        ('-parent_dir', './'),
        ('-recover_rate', 0.01),
        ('-first_read', iter_cnt == 0)
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
        ('-recover_rate', 0.01)
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
        ('-recover_rate', 0.01)
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
        ('-recover_rate', 0.01)
        ]
        acc = training_gp.main(param)
        if (acc < 0.9936):
            break
        else:
            iter_cnt += 1

    acc_list.append(acc)
