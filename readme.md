from_bag_get_data文件定时从bag中获取深度和rgb数据，发送到ros，可以后续使用订阅节点代替
需要在原始python环境运行。

cvdataprocess文件把ros里的数据转化为mmseg可以处理的格式，存储在临时硬盘空间，需要在原始python环境运行

data_process的最后一个cell进行语义分割，点云和标签的产生，需要在conda环境运行

使用mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .      命令下载模型

模型文件放在mmsegmentation/config文件夹
