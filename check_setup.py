
import torch
import datasets
import matplotlib

print("PyTorch version:", torch.__version__)
print("Datasets version:", datasets.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
print("Setup successful!")
