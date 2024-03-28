from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import torch

print(dir(models))

alexnet = models.alexnet(pretrained=True)

# You will see a similar output as below
# Downloading: "https://download.pytorch.org/models/alexnet-owt- 4df8aa71.pth" to /home/hp/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth

print(alexnet)

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])


2
3
# Import Pillow
from PIL import Image
img = Image.open("automotive.jpg")

#img.show()

img_t = transform(img)
print(img_t.size())

plt.imshow(img_t.numpy()[0])
#plt.show()

#arr = np.ndarray((3,224,224))#This is your tensor
#arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
#plt.imshow(arr_)
#plt.show()

batch_t = torch.unsqueeze(img_t, 0)

alexnet.eval()

out = alexnet(batch_t)
print("Out Shape:"+str(out.shape))

with open('imagenet_classes.txt') as f:
 classes = [line.strip() for line in f.readlines()]

_,index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(classes[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

print("#--------------------------------------------------------------------------------------------")
print("Resnet")

# First, load the model
resnet = models.resnet101(pretrained=True)

# Second, put the network in eval mode
resnet.eval()

# Third, carry out model inference
out = resnet(batch_t)

# Forth, print the top 5 classes predicted by the model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]






