import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_mnist import *
from network import MyModel

address = r"/media/wang/文件/Project/Material/ML/dataset/MNIST/self/data"
train_dataname = r"train-images.idx3-ubyte"
train_labelname = r"train-labels.idx1-ubyte"
test_dataname = r"t10k-images.idx3-ubyte"
test_labelname = r"t10k-labels.idx1-ubyte"

# log
log = SummaryWriter("logs")

# 设置训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])
# train_set = DataMNIST(address, train_dataname, train_labelname, transform)
# test_set = DataMNIST(address, test_dataname, test_labelname, transform)
train_set = datasets.MNIST(root=r"./data/",
                           train=True,
                           transform=transform,
                           target_transform=None,
                           download=True)

test_set = datasets.MNIST(root=r"./data/",
                           train=False,
                           transform=transform,
                           target_transform=None,
                           download=True)

train_set_size = len(train_set)
test_set_size = len(test_set)
print("Size of Train set： {}".format(train_set_size))
print("Size of Test set： {}".format(test_set_size))


# 数据加载
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)
i = 0
for data in train_loader:
    imgs, targets = data
    log.add_images("train_img", imgs, i)
    i = i + 1
i = 0
for data in test_loader:
    imgs, targets = data
    log.add_images("test_img", imgs, i)
    i = i + 1


# 加载网络模型
model = MyModel()
test = torch.ones((64, 1, 28, 28))
testout = model(test)

log.add_graph(model, test)
log.close()
model = model.to(device)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss(reduction='mean')
loss_fn = loss_fn.to(device)

# 定义优化器
learning_rate = 2e-3
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma = 0.8)

# 训练
epoch = 100
train_step = 0
# 添加 TensorBoard
writer = SummaryWriter("logs")

best_model = model
min_loss = 1000000
max_acc = 0
for i in range(epoch):
    print("----------Epoch({}/{})----------".format(i+1, epoch))

    model.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step = train_step + 1
        if train_step % 100 == 0:
            print("Training times: {}，Loss: {}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)

    # 调整学习率
    scheduler.step()

    # 测试
    total_test_loss = 0
    total_test_acc = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss

            acc = (outputs.argmax(1) == targets).sum()
            total_test_acc = total_test_acc + acc

    total_test_acc = 100*total_test_acc/test_set_size
    print("Loss on test set: {}".format(total_test_loss))
    print("Acc on test set: {:.2f}%".format(total_test_acc))
    writer.add_scalar("test_loss", total_test_loss.item(), i+1)
    writer.add_scalar("test_Acc", (total_test_acc).item(), i + 1)

    if total_test_acc > max_acc:
        min_loss = total_test_loss
        max_acc = total_test_acc
        best_model = model

    if (i+1) % 10 == 0:
        torch.save(model, "./model/model_{}.pth".format(i+1))
        print("Model Saved")

print("----------Best Model----------")
print("Loss of best model of test set: {}".format(min_loss))
print("Acc of best model on test set: {:.2f}%".format(max_acc))
torch.save(best_model, "./model/model_best.pth")
print("Best Model Saved")

writer.close()
