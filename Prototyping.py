import torch
# for i in range(torch.cuda.device_count()):
#    print(torch.cuda.get_device_properties(i).name)
numGPUs = torch.cuda.device_count()
print("Number of GPUs: ", numGPUs)