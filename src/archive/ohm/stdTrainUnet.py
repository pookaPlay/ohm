import torch
import torch.nn.functional as F
from unet import UNet
import QuanSynData
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=2, depth=3, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())
epochs = 10

Xsyn, Ysyn, XTsyn, YTsyn = QuanSynData.GetData(20)
#input_image = Image.open(filename)
#m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
#preprocess = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=m, std=s),
#])
#input_tensor = preprocess(input_image)

Ysyn = (Ysyn != 0)
Ysyn = np.squeeze(Ysyn.astype(np.longlong))

X = torch.tensor(Xsyn)
y = torch.tensor(Ysyn)
print("Tensor input shapes")
print(X.shape)
print(y.shape)

#tensor_image = torch.squeeze(X[0].permute(1,2,0))
#plt.imshow(tensor_image)
#plt.figure()
#plt.imshow(y[0].squeeze())
#plt.show()
#exit()

X = X.to(device)  # [N, C, H, W]
y = y.to(device)  # [N, H, W] with class indices (0, 1)

for epoch in range(epochs):

    prediction = model(X)  # [N, 2, H, W]
    #print(prediction.shape)
    #print(y.shape)
    loss = F.cross_entropy(prediction, y)

    optim.zero_grad()
    loss.backward()
    optim.step()
    
    myLoss = loss.item()
    print('Epoch %5d loss: %.3f' % (epoch+1, myLoss))


# Now test
XT = torch.tensor(XTsyn)
XT = XT.to(device)  # [N, C, H, W]
pred = model(XT)  # [N, 2, H, W]

myOut = pred.cpu()
npPred = myOut.detach().numpy()
predImage = npPred[0,0,:,:]
print(predImage.shape)
#plt.figure()
plt.imshow(predImage)
plt.show()
#exit()
