#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/6 11:40
# @Author  : LingJiaXiaoHu
# @File    : 草稿1.py
# @Software: win11 pytorch(GPU版本） python3.9.16



# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# *添加Tensorboard
writer = SummaryWriter(".logs_train")
start_time = time.time()  #记录这个时候的时间赋值给开始时间
for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))

    # 训练步骤开始
    tudui.train()  #不用也可以  设置为训练模式
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)      #控制是gpu还是cpu
        targets = targets.to(device)
        outputs = tudui(imgs)       #前向传播，计算输出值
        loss = loss_fn(outputs,targets)     #计算损失值

        # 优化器优化模型
        optimizer.zero_grad()    #优化器梯度清零
        loss.backward()     #反向传播，计算梯度值
        optimizer.step()    #更新模型参数

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:  # 使输出更好看  逢百记录
            end_time = time.time()      #每一次结束时间
            print(end_time - start_time)
            print("训练次数:{}，Loss:{}".format(total_train_step,loss.item()))     # item()将tensor数据类型转换成数字
            writer.add_scalar("train_loss",loss.item(),total_train_step)         # *

    # 测试步骤开始
    tudui.eval()   #不用也可以
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)      #控制是gpu还是cpu
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()      # *
            accuracy = (outputs.argmax(1) == targets).sum()   #argmax(), 1 取矩阵横向最大值 ， 0 取矩阵纵向最大值
            toatl_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_step))
    print("整体测试集上的正确率:{}".format(toatl_accuracy / test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)   # *
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1   # *

    # 保存模型
    torch.save(tudui,"tudui_{}.pth".format(i))     #后缀可以改，通常pth
    #官方 torch.save(tudui.state_dict(),"dudui_{}_gpu.pth".format(i))
    print("模型已保存")

writer.close()  # *