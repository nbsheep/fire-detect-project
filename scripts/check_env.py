import torch
print("PyTorch 版本 :", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())   # 必须是 True
if torch.cuda.is_available():
    print("显卡型号    :", torch.cuda.get_device_name(0))
    print("显存(GB)   :", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))