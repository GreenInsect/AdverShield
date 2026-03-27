import torch
import numpy as np
import cv2
import os
from PIL import Image
from utils import *
from darknet import Darknet
import warnings

# 忽略 PyTorch 版本弃用警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 模拟 Ultralytics 结果结构 ---

class FakeBoxes:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = torch.tensor(data) if data else torch.empty((0, 6))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            yield FakeBoxes(self.data[i:i+1])

    @property
    def xyxy(self):
        return self.data[:, :4] if len(self.data) > 0 else torch.empty((0, 4))

    @property
    def conf(self):
        return self.data[:, 4] if len(self.data) > 0 else torch.empty((0))

    @property
    def cls(self):
        return self.data[:, 5] if len(self.data) > 0 else torch.empty((0))

    def cpu(self):
        return FakeBoxes(self.data.cpu())

    def numpy(self):
        return self.data.detach().cpu().numpy()

class FakeResults:
    def __init__(self, boxes_list, orig_shape):
        self.boxes = FakeBoxes(boxes_list)
        self.orig_shape = orig_shape 

# --- 主类封装 ---

class YOLO:
    def __init__(self, cfg_path, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = torch.cuda.is_available()
        
        self.model = Darknet(cfg_path)
        self.model.load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.names = self._load_names()
        

    def _load_names(self):
        """模拟加载类别名称映射表"""
        # 默认尝试查找常见路径，如果找不到则返回通用标签
        names_file = 'data/coco.names' if self.model.num_classes == 80 else 'data/voc.names'
        
        if os.path.exists(names_file):
            with open(names_file, 'r') as f:
                names_list = [line.strip() for line in f.readlines()]
            return {i: name for i, name in enumerate(names_list)}
        else:
            # 如果找不到文件，创建一个通用的字典防止报错
            return {i: f"class_{i}" for i in range(self.model.num_classes)}

    def __call__(self, frame, conf=0.4, classes=None, verbose=False):
        if isinstance(frame, np.ndarray):
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            orig_h, orig_w = frame.shape[:2]
        else:
            img_pil = frame.convert('RGB')
            orig_w, orig_h = img_pil.size

        processed_img = img_pil.resize((self.model.width, self.model.height))
        
        with torch.no_grad():
            boxes = do_detect(self.model, processed_img, conf, 0.4, self.use_cuda)
        
        results_list = []
        for b in boxes:
            cls_id = int(b[6])
            score = b[4] * b[5]
            
            if classes is not None and cls_id not in classes:
                continue
            if score < conf:
                continue

            # 坐标转换
            x_c, y_c, w, h = b[0], b[1], b[2], b[3]
            x1 = (x_c - w / 2.0) * orig_w
            y1 = (y_c - h / 2.0) * orig_h
            x2 = (x_c + w / 2.0) * orig_w
            y2 = (y_c + h / 2.0) * orig_h
            
            results_list.append([x1, y1, x2, y2, score, cls_id])

        return [FakeResults(results_list, (orig_h, orig_w))]