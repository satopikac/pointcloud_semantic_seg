from mmseg.apis import inference_model, init_model,show_result_pyplot
import mmcv
config_file = '/home/mei123/workspace/cv/mmsegmentation/config/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = '/home/mei123/workspace/cv/mmsegmentation/config/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 从一个 config 配置文件和 checkpoint 文件里创建分割模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 测试一张样例图片并得到结果
img = '/home/mei123/workspace/cv/mmsegmentation/demo/demo.png'  # 或者 img = mmcv.imread(img), 这将只加载图像一次．
# 确保在文件开头导入了 show_result_pyplot
# from mmseg.apis import inference_model, init_model, show_result_pyplot 

result = inference_model(model, img)

# 在新的窗口里可视化结果 (对应 model.show_result(img, result, show=True))
# 注意：你需要传入 model 对象来获取类别元数据
show_result_pyplot(
    model, 
    img, 
    result, 
    show=True,
    out_file=None # 不保存文件
)

# 或者保存图片文件的可视化结果 (对应 model.show_result(img, result, out_file='result.jpg', opacity=0.5))
# 您可以改变 segmentation map 的不透明度(opacity)，在(0, 1]之间。
show_result_pyplot(
    model, 
    img, 
    result, 
    out_file='result.jpg', 
    opacity=0.5,
    show=False # 保存文件时通常不显示窗口
)
# 测试一个视频并得到分割结果
#video = mmcv.VideoReader('video.mp4')
#for frame in video:
#   result = inference_model(model, frame)
#   model.show_result(frame, result, wait_time=1)