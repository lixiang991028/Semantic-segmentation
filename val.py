##  从验证集中获取一个batch的数据
from data_generator import*
from model2 import*
device='cpu'
model = torch.load('unet.pkl').cpu()
val_loss_all = []
val_acc_all = []
val_loss = 0.0
val_num = 0
criterion = nn.NLLLoss()
since=time.time()
model.eval()  ## 设置模型为训练模式评估模式
for step, (b_x, b_y) in enumerate(test_loader):
    if step == 0:
        b_x = b_x.float().to(device)
        b_y = b_y.long().to(device)
        out = model(b_x)
        out = F.log_softmax(out, dim=1)
        pre_lab = torch.argmax(out, 1)
        loss = criterion(out, b_y)
        val_loss += loss.item() * len(b_y)
        val_num += len(b_y)

    ## 计算一个epoch在训练集上的损失和精度
val_loss_all.append(val_loss / val_num)
print('Val Loss: {:.4f}'.format(val_loss_all[-1]))
    ## 没个epoch的花费时间
time_use = time.time() - since
print("val complete in {:.0f}m {:.0f}s".format(
    time_use // 60, time_use % 60))
## 对验证集中一个batch的数据进行预测，并可视化预测效果
b_x_numpy = b_x.cpu().data.numpy()
b_x_numpy = b_x_numpy.transpose(0,2,3,1)
b_y_numpy = b_y.cpu().data.numpy()
pre_lab_numpy = pre_lab.cpu().data.numpy()
plt.figure(figsize=(16,9))
for ii in range(4):
    plt.subplot(3,4,ii+1)
    plt.imshow(inv_normalize_image(b_x_numpy[ii]))
    plt.axis("off")
    plt.subplot(3,4,ii+5)
    plt.imshow(label2image(b_y_numpy[ii],colormap))
    plt.axis("off")
    plt.subplot(3,4,ii+9)
    plt.imshow(label2image(pre_lab_numpy[ii],colormap))
    plt.axis("off")
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()


for step, (b_x, b_y) in enumerate(val_loader):
    if step == 1:
        b_x = b_x.float().to(device)
        b_y = b_y.long().to(device)
        out = model(b_x)
        out = F.log_softmax(out, dim=1)
        pre_lab = torch.argmax(out, 1)
        loss = criterion(out, b_y)
        val_loss += loss.item() * len(b_y)
        val_num += len(b_y)

    ## 计算一个epoch在训练集上的损失和精度
val_loss_all.append(val_loss / val_num)
print('Val Loss: {:.4f}'.format(val_loss_all[-1]))
    ## 每个epoch的花费时间
time_use = time.time() - since
print("val complete in {:.0f}m {:.0f}s".format(
    time_use // 60, time_use % 60))
## 对验证集中一个batch的数据进行预测，并可视化预测效果
b_x_numpy = b_x.cpu().data.numpy()
b_x_numpy = b_x_numpy.transpose(0,2,3,1)
b_y_numpy = b_y.cpu().data.numpy()
pre_lab_numpy = pre_lab.cpu().data.numpy()
plt.figure(figsize=(16,9))
for ii in range(4):
    plt.subplot(3,4,ii+1)
    plt.imshow(inv_normalize_image(b_x_numpy[ii]))
    plt.axis("off")
    plt.subplot(3,4,ii+5)
    plt.imshow(label2image(b_y_numpy[ii],colormap))
    plt.axis("off")
    plt.subplot(3,4,ii+9)
    plt.imshow(label2image(pre_lab_numpy[ii],colormap))
    plt.axis("off")
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()