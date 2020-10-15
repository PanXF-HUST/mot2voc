# @Author   : Pan Xiaofeng
# @Email    : xiaofengpan@hust.edu.cn
# @FileName : test_result.py
# @DateTime : 2020/10/8 21:00

import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result

import pickle
import progressbar

# you should change the format_name and the epoch
format_name= 'fcos_mstrain_voc'

checkpoint_file = 'work_dirs/'+ format_name +'/epoch_24.pth'
config_file = 'configs/'+ format_name+ '.py'

img_dir = 'data/VOCdevkit/VOC2007/JPEGImages/'
out_dir = 'results/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

model = init_detector(config_file,checkpoint_file)

fp = open('data/VOCdevkit/VOC2007/ImageSets/Main/train1.txt','r')
test_list = fp.readlines()

imgs=[]
for test_1 in test_list:
    test_1 = test_1.replace('\n','')
    name = img_dir + test_1 + '.jpg'
    imgs.append(name)

results = []
# for i,result in enumerate(inference_detector(model,imgs)):
#     print('model is processing the {}/{} images.'.format(i+1,len(imgs)))
#     results.append(result)

count = 0

# initial the processbar
bar = progressbar.ProgressBar(maxval=len(imgs)).start()

for img in imgs:
    count += 1
    # print('model is processing the {}/{} images.'.format(count,len(imgs)))
    bar.update(count)
    result = inference_detector(model,img)
    results.append(result)

bar.finish()

print('\nwriting results to {}'.format(format_name+'.pkl'))
pkl_file = out_dir+format_name+'.pkl'
mmcv.dump(results,pkl_file)

#output the result
print('evaluate result has been written to {}'.format(pkl_file))
print('Loading...')
print()
print('transform the pkl-file to txt-file')


#process the pkl—file to the MOT style
fp_pkl = open(pkl_file,'rb')
inf = pickle.load(fp_pkl)
length = len(inf)

output_dir = out_dir+format_name+'/'+ 'train'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# for i in range (len(test_list)):

fp2 = open(output_dir+'/MOT17-02.txt','w')
fp4 = open(output_dir+'/MOT17-04.txt','w')
fp5 = open(output_dir+'/MOT17-05.txt','w')
fp9 = open(output_dir+'/MOT17-09.txt','w')
fp10 = open(output_dir+'/MOT17-10.txt','w')
fp11 = open(output_dir+'/MOT17-11.txt','w')
fp13 = open(output_dir+'/MOT17-13.txt','w')

for i in range(length):
    pic_info = test_list[i]
    file_fp = pic_info[:2]
    if file_fp=='02':
        fp = fp2
    if file_fp == '04':
        fp = fp4
    if file_fp=='05':
        fp = fp5
    if file_fp=='09':
        fp = fp9
    if file_fp=='10':
        fp = fp10
    if file_fp=='11':
        fp = fp11
    if file_fp=='13':
        fp = fp13
    fram = int(pic_info[2:])

    for pre in inf[i]:
        for one in pre:
            x1 = one[0]
            y1 = one[1]
            x2 = one[2]
            y2 = one[3]
            conf = one[4]
            w = x2-x1
            h = y2-y1
            result = str(fram) + ',' + str(-1) +',' + str(x1) + ',' + str(y1) +',' + str(w) + ',' + str(h) + ',' + str(conf)+'\n'
            fp.writelines(result)

fp2.close()
fp4.close()
fp5.close()
fp9.close()
fp10.close()
fp11.close()
fp13.close()
print('finish processing all the results')



