#此处是为了显示以下dataloader中的图片，以防我之前数据划分的时候分错了
import torch
import matplotlib.pyplot as plt
def image_show(train_iter,batch_size):
    images, labels = next(iter(train_iter))
    print(images.size())
    plt.figure(figsize=(9, 9))
    for i in range(batch_size):
        plt.subplot(3, 3, i + 1)
        plt.title(labels[i].item())
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis('off')
    plt.show()
