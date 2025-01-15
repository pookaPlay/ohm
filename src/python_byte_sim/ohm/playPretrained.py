import torch
import urllib
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)


#url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
#try: urllib.URLopener().retrieve(url, filename)
#except: urllib.request.urlretrieve(url, filename)

filename = "TCGA_CS_4944.png"
input_image = Image.open(filename)
m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

print(torch.round(output[0]))

myOut = output[0].cpu()
binOut = torch.round(myOut)
tensor_image = torch.squeeze(binOut.permute(1,2,0))
plt.imshow(tensor_image)
plt.show()

#plt.imshow(  tensor_image.permute(1, 2, 0)  )
#or 
#tensor_image = tensor_image.view(tensor_image.shape[1], tensor_image.shape[2], tensor_image.shape[0])
#print(type(tensor_image), tensor_image.shape)
