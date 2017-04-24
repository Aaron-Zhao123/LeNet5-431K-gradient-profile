import os
import training_v5

acc_list = []
count = 0
pcov = 90
pfc = 90
retrain = 0
model_tag = 'pcov'+str(pcov)+'pfc'+str(pfc)
while (count < 10):
    # print(model_tag)
    # pfc = pfc+1
    pcov = pcov+1
    param = [
    ('-pcov',pcov),
    ('-pfc',pfc),
    ('-m',model_tag)
    ]
    acc = training_v5.main(param)
    model_tag = 'pcov'+str(pcov)+'pfc'+str(pfc)
    acc_list.append(acc)
    count = count + 1
    print (acc)

print('accuracy summary: {}'.format(acc_list))
