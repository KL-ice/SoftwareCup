import torch
#from torch.nn import Linear, Module, MSELoss
import torch.nn as nn
#from torch.optim import SGD
import numpy as np
import pandas as pd
#import matplotlib
#import matplotlib.pyplot as plt
#import seaborn as sns
import csv
import cv2
import torch.functional as F
import torchvision
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms

'''x = np.linspace(0,20,500)
y = 5*x + 7
#plt.plot(x,y)


x = np.random.rand(256)
noise = np.random.randn(256) / 4
y = x * 5 + 7 + noise
#df = pd.DataFrame()
#df['x'] = x
#df['y'] = y
#sns.lmplot(x='x', y='y', data=df)
#plt.show()

model = Linear(1,1)
criterion = MSELoss()
optim = SGD(model.parameters(), lr = 0.01)
epochs = 3000

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

for i in range(epochs):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    outputs = model(inputs)
    optim.zero_grad()
    loss=criterion(outputs, labels)
    loss.backward()
    optim.step()
    if (i%100 == 0):
        print('epoch {}, loss {:1.4f}'.format(i,loss.data.item()))


predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label = 'data', alpha = 0.3)
plt.plot(x_train, predicted, label = 'predicted', alpha = 1)
plt.legend()
plt.show()'''
def drawLines(img, x1, y1, width, height):     #返回画出了矩形框的img
    #xBegin = min(int((x1/256)*len(img)), int((x2/256)*len(img)))
    #xEnd = max(int((x1/256)*len(img)), int((x2/256)*len(img)))
    #yBegin = min(int((y1 / 256) * len(img[0])), int((y2 / 256) * len(img[0])))
    #yEnd = max(int((y1 / 256) * len(img[0])), int((y2 / 256) * len(img[0])))
    for i in range(width):
        '''img[int(((x1+i)/256)*len(img))][int((y1 / 256) * len(img[0]))][0] = 0
        img[int(((x1 + i) / 256) * len(img))][int((y1 / 256) * len(img[0]))][0] = 0
        img[int(((x1 + i) / 256) * len(img))][int((y1 / 256) * len(img[0]))][0] = 250'''
        img[y1][x1 + i][0] = 0
        img[y1][x1 + i][1] = 0
        img[y1][x1 + i][2] = 250
        img[y1 + height][x1 + i][0] = 0
        img[y1 + height][x1 + i][1] = 0
        img[y1 + height][x1 + i][2] = 250
    for i in range(height):
        img[y1 + i][x1][0] = 0
        img[y1 + i][x1][1] = 0
        img[y1 + i][x1][2] = 250
        img[y1 + i][x1 + width][0] = 0
        img[y1 + i][x1 + width][1] = 0
        img[y1 + i][x1 + width][2] = 250
    return img

def printSize(filenames):    #将all_labels.csv文件中，每个图片的大小输出到img_sizes.csv，用作label坐标的变换
    out = open('image_sizes.csv', 'w', newline='')
    # 设定写入模式
    csv_write = csv.writer(out, dialect='excel')
    for i in range(len(filenames)):

        img = cv2.imread("E:\\Grade3\\Software_Competition\\SoftwareCup\\all_images\\"+filenames[i])
        #img = cv2.imdecode(np.fromfile("E:\\Grade3\\Software_Competition\\SoftwareCup\\all_images\\"+filenames[i], dtype=np.uint8), -1)
        if img is None:
            print(filenames[i])
            continue
        #print("E:\\Grade3\\Software_Competition\\SoftwareCup\\all_images\\"+filenames[i])
        content = []
        content.append(filenames[i])
        content.append(len(img))
        content.append(len(img[0]))
        csv_write.writerow(content)


def getLabels(csvName, resizeWidth, iffill):   #获取拉伸后的labels， resizeWidth：resize后的像素宽度， iffill如resizeImage函数一样   依赖于上一个函数
    csv = pd.read_csv(csvName, usecols=['filename', 'region_shape_attributes'])
    csv = np.array(csv)
    sizes = pd.read_csv("image_sizes.csv")
    sizes = np.array(sizes)
    filenames = []
    labels = []
    for i in range(len(csv)):
        resizeRate = resizeWidth / sizes[i][2]
        filenames.append(csv[i][0])
        j = 19
        x = 0
        y = 0
        width = 0
        height = 0
        while (csv[i][1][j]>='0' and csv[i][1][j]<='9'):
            x = x*10 + int(csv[i][1][j])
            j = j+1
        j = j+5
        while (csv[i][1][j]>='0' and csv[i][1][j]<='9'):
            y = y*10 + int(csv[i][1][j])
            j = j+1
        j = j+9
        while (csv[i][1][j]>='0' and csv[i][1][j]<='9'):
            width = width*10 + int(csv[i][1][j])
            j = j+1
        j = j+10
        while (csv[i][1][j]>='0' and csv[i][1][j]<='9' and j<len(csv[i][1])):
            height = height*10 + int(csv[i][1][j])
            j = j+1
        if iffill:
            labels.append([int(x*resizeRate), int(y*resizeRate), int(width*resizeRate), int(height*resizeRate)])
        else:
            labels.append([int(x * resizeWidth / sizes[i][2]), int(y * resizeWidth / sizes[i][1]), int(width * resizeWidth / sizes[i][2]), int(height * resizeWidth / sizes[i][1])])
    return filenames, labels


def resizeImage(img, width, iffill):         #img:原图片，   width:填充后像素值（正方形）   iffill: True：原比例变换，缺的部分填充黑色  False:拉伸成正方形
    if iffill:
        sizeY = int(len(img) * (width / len(img[0])))
        img = cv2.resize(img, (width, sizeY))
        fill = np.zeros(((width - sizeY), width, 3), dtype='uint8')
        img = np.concatenate((img, fill), axis=0)
    else:
        img = cv2.resize(img, (width, width))
    return img


class CustomDataset(data.Dataset):
    def __init__(self, labelcsv, sizecsv):
        self.names, self.labels = getLabels(labelcsv, 256, False)
        self.trasfer_2_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        #print(idx)
        #print("all_images" + str(self.names[idx]))
        img = cv2.imread(".\\all_images\\" + str(self.names[idx]))
        #sizeY = int(len(img) * (256 / len(img[0])))
        #img = cv2.resize(img, (256, sizeY))
        img = resizeImage(img, 256, False)
        img_tensor = self.trasfer_2_tensor(img)
        label = np.array(self.labels[idx])
        label = label.astype('float32')
        label = torch.from_numpy(label)
        #print(np.shape(img))
        return (img_tensor, label)

    def __len__(self):
        return len(self.names)


def openImg(filePath):         #以tensor的形式打开图片，可以直接传给神经网络，方便查看神经网络对于某一张图片的检测结果
    img = cv2.imread(filePath)
    img = resizeImage(img, 256, False)
    tt = transforms.ToTensor()
    img = tt(img)
    img = img.unsqueeze(0)

    print(img.shape)

    return img


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = crit(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#f, l = getLabels("all_labels.csv", 256, False)
#print(f[0], l[0])
'''img = cv2.imread(".\\all_images\\000001.jpg")
img = cv2.resize(img, (256, 256))
img = drawLines(img, 41, 128, 171, 22)
cv2.imshow("1", img)
cv2.waitKey(0)'''

if __name__ == '__main__':
    #====================训练代码===============================
    resnet = torchvision.models.resnet50(pretrained=False).cuda()
    resnet.fc = nn.Linear(resnet.fc.in_features, 4).cuda()
    custom_dataset = CustomDataset("all_labels.csv", "image_sizes.csv")
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=40,
                                               shuffle=True,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                              batch_size=40)
    optimizer = optim.Adam(resnet.parameters())
    crit = nn.MSELoss().cuda()
    #for i in range(50):
        #train(resnet, train_loader, optimizer, i)
    #torch.save(resnet, 'adam_MSE_40_50.pkl')

    #======================训练代码结束（要取消for循环注释才能用）========================================


    #==============检测一张图片的输出结果======================================
    mod = torch.load("adam_MSE_40_50.pkl")
    img = openImg(".\\all_images\\1001108.jpg")
    a = mod(img.cuda()).detach().cpu().numpy()[0]
    print(mod(img.cuda()).detach().cpu().numpy())
    img = drawLines(cv2.resize(cv2.imread(".\\all_images\\1001108.jpg"), (256,256)), int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    cv2.imshow("!", img)
    cv2.waitKey(0)
    #==============================================================================