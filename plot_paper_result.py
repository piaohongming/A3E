import numpy as np
import matplotlib.pyplot as plt
def hyper1():
    

    # 数据
    x = [0, 0.5, 1]
    y = [0, 0.0625, 0.125]
    data = np.array([[44, 50, 52], [46, 58, 56], [46, 56, 58]])



    # 绘制数据的热力图
    plt.imshow(data, cmap='Blues', vmin=data.min(), vmax=data.max())
    '''
    # 在每个方格中显示数字
    for i in range(len(y)):
        for j in range(len(x)):
            ax.text(j, i, str(data[i, j]), va='center', ha='center', color='black' if data[i, j] > 50 else 'gray')
    '''
    # 设置坐标轴
    #ax.set_xticks(np.arange(len(x)))
    #ax.set_yticks(np.arange(len(y)))

    #ax.set_xticklabels(x)
    #ax.set_yticklabels(y)

    # 加上颜色条
    #fig.colorbar(cax)

    # 显示方格图
    #plt.show()
    plt.tight_layout()
    plt.savefig("hyper1.pdf",dpi=300, format="pdf")

hyper1()