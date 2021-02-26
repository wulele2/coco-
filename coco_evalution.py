# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 2021/02/25
# @Author  : lele wu
# @Email   : 2541612007@qq.com
# @File    : coco_evalution.py
# @Comment: 本脚本用于研究cocoeval.py
# ======================================================

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab,json

if __name__ == "__main__":
    gt_path = "/home/wujian/WLL/mmdet-master/data/coco/annotations/instances_val2017.json"  # 存放真实标签的路径
    dt_path = "/home/wujian/WLL/mmdet-master/tools/work_dirs/yolo_result/my_result.json"    # 存放检测结果的路径
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")                                             #
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()