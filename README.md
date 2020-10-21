# MOT to VOC

**Function**: For research purpose, we create this project for others to simplify 
tranform the MOT17Det and MOT20Det to VOC format so that can be easily used in detection task.
 
We also press the Implementation details on [Zhihu](https://zhuanlan.zhihu.com/p/265984636).


## Introduction

The master branch works with python 3.5+.

```shell 
# example
linux:/xxx/mmdetection$ python mot2voc.py 20
100% (429 of 429) |#################################################################################################################################################################################| Elapsed Time: 0:00:25 Time:  0:00:25
100% (2782 of 2782) |###############################################################################################################################################################################| Elapsed Time: 0:09:31 Time:  0:09:31
100% (2405 of 2405) |###############################################################################################################################################################################| Elapsed Time: 0:14:32 Time:  0:14:32
100% (3315 of 3315) |###############################################################################################################################################################################| Elapsed Time: 0:39:28 Time:  0:39:28
folders [xxx,xxx,xxx,xxx] have been succeed checked
100% (2080 of 2080) |#################################################################################################################################################################################| Elapsed Time: 0:00:25 Time:  0:00:25
100% (1008 of 1008) |###############################################################################################################################################################################| Elapsed Time: 0:09:31 Time:  0:09:31
100% (585 of 585) |###############################################################################################################################################################################| Elapsed Time: 0:14:32 Time:  0:14:32
100% (806 of 806) |###############################################################################################################################################################################| Elapsed Time: 0:39:28 Time:  0:39:28
folders [xxx,xxx,xxx,xxx] have been succeed checked
```

![demo image](demo.png)



## Installation
get into ./mmdetection and download MOT dataset and place into ./mmdetection
```shell
# into mmdetection folder
cd mmdetection-master

# transform mot17 to voc format
python mot2voc.py 17

# transform mot20 to voc format
python mot2voc.py 20
```

## Get Started

```shell
# git clone
git clone https://github.com/PanXF-HUST/mot2voc.git


Refer to the example to place the file and run
```

## Folder
```

mmdetection
├── mmdet
├── tools
├── configs
├── MOT{17,20}
│   ├── train
│   │   ├── MOT{17,20}-01
│   │   │   ├── img1
│   │   │   ├── gt.txt
│   │   │   ├── seqinfo.ini
│   │   ├── ...
│   ├── test
│   │   ├── ...

├── data
│   ├── MOT{17,20}Det
│   │   ├── VOC2007
│   │   │   ├── Annotations
│   │   │   │   ├── 01000001.xml
│   │   │   │   ├── ...
│   │   │   ├── ImageSets
│   │   │   │   ├── Main
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   │   │   ├── test.txt
│   │   │   │   ├── train_all.txt
│   │   │   │   ├── test_val.txt
│   │   │   ├── JPEGImages
│   │   │   │   ├── 01000001.jpg
│   │   │   │   ├── ...
```

