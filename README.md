# CNN-based-autoencoder-for-fmri
本仓库是作者为处理fmri数据而编写的自编码器。

<br>

## 项目描述
encoder可以将Pereira数据集中大小为88*128*85的fmri数据用三层卷积映射到1000的一维向量（用于降维），decoder可以将其尽可能地还原成原向量。

在编写过程中，最后存放降维数据的时候忘记了dataloader设置成了shuffle=True，浪费了很多时间找bug...真是血泪教训...

<br>

