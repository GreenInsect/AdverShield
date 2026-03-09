import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from patch_detector import PatchDetector

# 1. 自动检测设备 (GPU vs CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载对抗图片
img_name = "crop_000001_p.png"
image_0 = cv2.imread(img_name)

if image_0 is None:
    raise FileNotFoundError(f"无法读取图片: {img_name}")

image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB) 
image = np.stack([image_0], axis=0).astype(np.float32) / 255.0 
image = image.transpose(0, 3, 1, 2)
image = torch.tensor(image).to(device)  # 将输入数据送入设备

# 3. 加载 SAC 处理器
SAC_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[150, 100, 75, 50, 25], n_patch=1)

# 加载权重
ckpt_path = "models/patch_processor.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

SAC_processor.unet.load_state_dict(torch.load(ckpt_path, map_location=device))
SAC_processor.unet.to(device) 
SAC_processor.unet.eval()      

# 4. 补丁检测与修复
with torch.no_grad():
    x_processed, _, _ = SAC_processor(image, bpda=True, shape_completion=False)
    print(x_processed)

# 5. 后处理显示
image_sac = np.asarray(x_processed[0].cpu().detach())
image_sac = image_sac.transpose(1, 2, 0)
image_sac = np.clip(image_sac, 0, 1) 

# 6. 展示结果
f, axarr = plt.subplots(1, 2, figsize=(15, 10))
axarr[0].imshow(image_0)
axarr[0].axis("off")
axarr[0].set_title('Adv image')

axarr[1].imshow(image_sac)
axarr[1].axis("off")
axarr[1].set_title('SAC image')

plt.savefig('sac_result.png')
print("结果已保存至 sac_result.png")
plt.show()