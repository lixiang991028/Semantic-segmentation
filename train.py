## 网络的训练函数
from data_generator import*
from model2 import*
def train_model(model, criterion, optimizer, traindataloader,
                valdataloader, num_epochs=25):
    """
    model:网络模型；criterion：损失函数；optimizer：优化方法；
    traindataloader:训练数据集，valdataloader:验证数据集
    num_epochs:训练的轮数
    """
    since = time.time()#时间的时间戳
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10#先设置最佳损失
    train_loss_all = []#训练损失
    train_acc_all = []#训练准确率
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0
        # 每个epoch包括训练和验证阶段
        model.train()  ## 设置模型为训练模式
        for step, (b_x, b_y) in enumerate(traindataloader):
            optimizer.zero_grad()#梯度清零
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)

            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)  # 预测的标签
            loss = criterion(out, b_y)  # 计算损失函数值
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
            train_num += len(b_y)
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        print('{} Train Loss: {:.4f}'.format(epoch, train_loss_all[-1]))

        ## 计算一个epoch的训练后在验证集上的损失
        model.eval()  ## 设置模型为训练模式评估模式
        for step, (b_x, b_y) in enumerate(valdataloader):
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)
            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)
            out=out.transpose(0,3,2,1)
            loss = criterion(out, b_y)
            val_loss += loss.item() * len(b_y)
            val_num += len(b_y)
        ## 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / val_num)
        print('{} Val Loss: {:.4f}'.format(epoch, val_loss_all[-1]))
        ## 保存最好的网络参数
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        ## 没个epoch的花费时间
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(
            time_use // 60, time_use % 60))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "val_loss_all": val_loss_all})
    ## 输出最好的模型
    model.load_state_dict(best_model_wts)
    return model, train_process
    ## 定义损失函数和优化器
LR = 0.0003
criterion = nn.NLLLoss()
optimizer = optim.Adam(fcn8s.parameters(), lr=LR, weight_decay=1e-4)
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
unet, train_process = train_model(
    unet, criterion, optimizer, train_loader,
    val_loader, num_epochs=3)
## 保存训练好的网络fcn8s
torch.save(unet, "fcn8s.pkl")

## 可视化模型训练过程中
plt.figure(figsize=(10,6))
plt.plot(train_process.epoch,train_process.train_loss_all,
         "ro-",label = "Train loss")
plt.plot(train_process.epoch,train_process.val_loss_all,
         "bs-",label = "Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()