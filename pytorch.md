# transforms

## transforms.Compose()

~~~python
self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

~~~

> torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起：
> 比如说：
> transforms.Compose([
> 	transforms.CenterCrop(10),
> 	transforms.ToTensor(),
> ]}
> 这样就把两个步骤整合到了一起。
>
> ==接下来介绍transforms中的函数：==
>
> Resize：把给定的图片resize到given size
>
> Normalize：Normalized an tensor image with mean and standard deviation
>
> ToTensor：convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
>
> ToPILImage: convert a tensor to PIL image
>
> Scale：目前已经不用了，推荐用Resize
>
> **CenterCrop**：在图片的中间区域进行裁剪
>
> **RandomCrop**：在一个随机的位置进行裁剪
>
> RandomHorizontalFlip：以0.5的概率水平翻转给定的PIL图像
>
> RandomVerticalFlip：以0.5的概率竖直翻转给定的PIL图像
>
> RandomResizedCrop：将PIL图像裁剪成任意大小和纵横比
>
> Grayscale：将图像转换为灰度图像
>
> RandomGrayscale：将图像以一定的概率转换为灰度图像
>
> FiceCrop：把图像裁剪为四个角和一个中心
>
> TenCrop
>
> Pad：填充
>
> ColorJitter：随机改变图像的亮度对比度和饱和度。
> ————————————————
> 版权声明：本文为CSDN博主「岁月神偷小拳拳」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/u013925378/article/details/103363232/

# 计算模型的参数量

1. 示例：

   ~~~python
   plfd_backbone = PFLDInference() # init model
   
   total = sum([param.nelement() for param in plfd_backbone.parameters()])
   print("Number of parameters: %.2fM"%(total/1e6))
   ~~~

2. 示例：

   ~~~python
   dummy_input = torch.randn(1, 3, 112, 112)
   plfd_backbone = PFLDInference() # init model
   
   from thop import profile
   macs, p = profile(model=plfd_backbone, inputs=(dummy_input, ), verbose=False)
   print(f"macs: {macs / 1000000.0}, params: {p / 1000000.0}")
   
   
   ~~~

   

# DataLoader

示例：

~~~python
dataloader = DataLoader(lapaDataset, batch_size=128, shuffle=True, num_workers=0, drop_last=False)
~~~

注意事项：

==batchsize 个图片的尺寸必须统一，否则程序出错， 如下：==

**RuntimeError: stack expects each tensor to be equal size, but got [450, 450, 3] at entry 0 and [642, 600, 3] at entry 1**

> 原因是，stack 的所有图片 的尺寸不一致。



# gpu

> os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"
>
> torch.cuda.is_available()
>
> torch.cuda.device_count()
>
> x = [torch.cuda.get_device_properties(i) for i in range(ng)] #获取每个gpu的属性

# torch.cat

~~~python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
>>> torch.cat((x, x, x), -1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
~~~



# pytorch 保存、加载模型

>`torch.save`：将对象序列化到硬盘上，该对象可以是 Models, tensors和 dictionaries 等。实际上是使用了python的 `pickle`方法。
>
>`torch.load`：将硬盘上序列化的对象加载设备中。实际是使用了`pickle`的解包方法。
>
>`torch.nn.Module.load_state_dict`：通过反序列化`state_dict`加载模型参数字典。



==***`state_dict`是一个从每一个层的名称映射到这个层的参数`Tesnor`的字典对象。***==

==注意，只有具有可学习参数的层(卷积层、线性层等)和注册缓存`(batchnorm’s running_mean)`才有`state_dict`中的条目。优化器`(torch.optim)`也有一个`state_dict`，其中包含关于优化器状态以及所使用的超参数的信息。==

## 模型参数

在pytorch中`torch.nn.Module`模型的参数存放在模型的`parameters`中(`model.parameters()`),而==`state_dict`是参数tensor的字典。仅当该层有可学习的参数才会有属性`state_dict`。==`torch.optim`也有`state_dict`属性，其中包含的是优化器的参数，即网络所使用的超参数。

~~~python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
model = TheModelClass()
# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 打印模型参数
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# 打印优化器参数
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
~~~



## 保存和加载模型

保存的文件结尾一般使用`.pt`或`.pth`。最后的`model.eval()`表示将`drop`和`batch nromalization`层设置为测试模式。可通过`mode.train()`转化为训练模式。

* ==模型参数==

  **说明：也是官方推荐的方法，只保存和恢复模型中的参数。使用这种方法，需要我们自己导入模型的结构信息。**

  ~~~python
  #保存模型参数
  torch.save(model.state_dict(), PATH)
  #加载模型参数
  model = TheModelClass(*args, **kwargs)
  model.load_state_dict(torch.load(PATH))
  model.eval()
  ~~~

* ==模型==

  **说明： 使用这种方法，将会保存模型的参数和结构信息。**

  ~~~python
  #保存模型
  torch.save(model, PATH)
  #加载模型
  model = torch.load(PATH)
  model.eval()
  ~~~

* ==保存多个模型==

  ~~~python
  torch.save({
              'modelA_state_dict': modelA.state_dict(),
              'modelB_state_dict': modelB.state_dict(),
              'optimizerA_state_dict': optimizerA.state_dict(),
              'optimizerB_state_dict': optimizerB.state_dict(),
              ...
              }, PATH)
  ~~~

* ==加载多个模型==

  ~~~python
  modelA = TheModelAClass(*args, **kwargs)
  modelB = TheModelBClass(*args, **kwargs)
  optimizerA = TheOptimizerAClass(*args, **kwargs)
  optimizerB = TheOptimizerBClass(*args, **kwargs)
  
  checkpoint = torch.load(PATH)
  modelA.load_state_dict(checkpoint['modelA_state_dict'])
  modelB.load_state_dict(checkpoint['modelB_state_dict'])
  optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
  optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
  
  modelA.eval()
  modelB.eval()
  # - or -
  modelA.train()
  modelB.train()
  ~~~

* ==加载不同的模型==

  ==加载他人训练的模型，可能需要忽略部分层。则将`load_state_dict`方法的`strict`参数设置为`False`。则模型金价在modelB中名称对应的层的参数。==

  **说明：它直接忽略那些没有的dict，有相同的就复制，没有就直接放弃赋值！**

  ~~~python
  torch.save(modelA.state_dict(), PATH)
  modelB = TheModelBClass(*args, **kwargs)
  modelB.load_state_dict(torch.load(PATH), strict=False)
  ~~~

* ==加载不同设备的模型==

  1. 将由GPU保存的模型加载到CPU上。==将`torch.load()`函数中的`map_location`参数设置为`torch.device('cpu')`==

  ~~~python
  device = torch.device('cpu')
  model = TheModelClass(*args, **kwargs)
  model.load_state_dict(torch.load(PATH, map_location=device))
  ~~~

  2. 将由GPU保存的模型加载到GPU上。确保对输入的`tensors`调用`input = input.to(device)`方法。

  ~~~python
  device = torch.device("cuda")
  model = TheModelClass(*args, **kwargs)
  model.load_state_dict(torch.load(PATH))
  model.to(device)
  ~~~

  3. 将由CPU保存的模型加载到GPU上。确保对输入的`tensors`调用`input = input.to(device)`方法。`map_location`是将模型加载到GPU上，`model.to(torch.device('cuda'))`是将模型参数加载为CUDA的tensor。最后保证使用`.to(torch.device('cuda'))`方法将需要使用的参数放入CUDA。

  ~~~python
  device = torch.device("cuda")
  model = TheModelClass(*args, **kwargs)
  model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
  model.to(device)
  ~~~




# 是否冻结权重

示例：

~~~python
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad_(False)
~~~





# torch.stack, torch.as_tensor

~~~python
>>> import torch
>>> import numpy as np
>>> image_a = np.random.randn(3, 4, 5)
>>> image_b = np.random.randn(3, 4, 5)
>>> image_c = np.random.randn(3, 4, 5)
>>> image_a
array([[[ 0.91387459,  0.00538825,  1.45887352, -0.74785712,
          1.32300517],
        [-1.31764841, -0.3718417 , -1.35850879, -0.58140172,
         -1.44131811],
        [ 0.11085216, -0.1662203 ,  0.91846831,  0.56486866,
          0.51557138],
        [-0.8683517 , -1.02183699, -0.49120935, -0.57746753,
          0.7309601 ]],

       [[ 3.49278773,  0.85399508,  0.66705887,  0.12482549,
          0.29586813],
        [-1.35609752, -0.37834719,  0.80206437,  0.39177243,
         -2.39092106],
        [-0.60915936,  1.0605545 ,  0.42477012,  0.3591753 ,
         -1.08361277],
        [-0.42222022,  0.04530008, -0.2779951 , -0.95785066,
         -0.23571577]],

       [[-1.06975387,  0.35138859,  1.6942104 , -0.96313545,
          0.34580783],
        [-0.30407329,  0.87900515,  1.43994098, -0.80934319,
          0.88444079],
        [ 1.17017925, -2.45368125,  1.19189383,  1.27514577,
         -0.84628511],
        [-0.34801086, -0.73013543,  0.62219335, -1.40078809,
          1.30786796]]])
>>>
>>> tensor_image_a = torch.from_numpy(image_a)
>>> tensor_image_b = torch.from_numpy(image_b)
>>> tensor_image_c = torch.from_numpy(image_c)
>>> tensor_image_a
tensor([[[ 0.9139,  0.0054,  1.4589, -0.7479,  1.3230],
         [-1.3176, -0.3718, -1.3585, -0.5814, -1.4413],
         [ 0.1109, -0.1662,  0.9185,  0.5649,  0.5156],
         [-0.8684, -1.0218, -0.4912, -0.5775,  0.7310]],

        [[ 3.4928,  0.8540,  0.6671,  0.1248,  0.2959],
         [-1.3561, -0.3783,  0.8021,  0.3918, -2.3909],
         [-0.6092,  1.0606,  0.4248,  0.3592, -1.0836],
         [-0.4222,  0.0453, -0.2780, -0.9579, -0.2357]],

        [[-1.0698,  0.3514,  1.6942, -0.9631,  0.3458],
         [-0.3041,  0.8790,  1.4399, -0.8093,  0.8844],
         [ 1.1702, -2.4537,  1.1919,  1.2751, -0.8463],
         [-0.3480, -0.7301,  0.6222, -1.4008,  1.3079]]], dtype=torch.float64)
>>>
>>> image_tuple = (tensor_image_a, tensor_image_b, tensor_image_c)
>>> type(image_tuple)
<class 'tuple'>
>>>
>>> images_batch = torch.stack(image_tuple, dim=0)
>>> type(images_batch)
<class 'torch.Tensor'>
>>> images_batch.shape
torch.Size([3, 3, 4, 5])
>>> images_batch
tensor([[[[ 0.9139,  0.0054,  1.4589, -0.7479,  1.3230],
          [-1.3176, -0.3718, -1.3585, -0.5814, -1.4413],
          [ 0.1109, -0.1662,  0.9185,  0.5649,  0.5156],
          [-0.8684, -1.0218, -0.4912, -0.5775,  0.7310]],

         [[ 3.4928,  0.8540,  0.6671,  0.1248,  0.2959],
          [-1.3561, -0.3783,  0.8021,  0.3918, -2.3909],
          [-0.6092,  1.0606,  0.4248,  0.3592, -1.0836],
          [-0.4222,  0.0453, -0.2780, -0.9579, -0.2357]],

         [[-1.0698,  0.3514,  1.6942, -0.9631,  0.3458],
          [-0.3041,  0.8790,  1.4399, -0.8093,  0.8844],
          [ 1.1702, -2.4537,  1.1919,  1.2751, -0.8463],
          [-0.3480, -0.7301,  0.6222, -1.4008,  1.3079]]],


        [[[ 0.5074, -1.9472,  0.0290,  0.2692,  1.6226],
          [-0.2067,  0.1147, -0.3367,  0.7179,  0.0895],
          [ 1.1212, -0.8893, -1.2598, -0.6275, -0.6027],
          [-0.9702, -1.9714,  0.1273, -2.3799, -1.1117]],

         [[ 0.4177,  0.9798, -0.6690, -0.1984, -0.1814],
          [ 0.5176, -0.2734, -2.5594, -0.7119, -1.0188],
          [ 0.0110,  0.8594,  0.5214,  0.2521,  2.2344],
          [-0.0314,  1.4781, -1.2913, -0.9982,  0.7245]],

         [[-0.1547, -0.1445, -1.2792, -1.0396,  0.2761],
          [-1.4257, -0.5512,  1.5132, -0.5712, -1.4612],
          [ 0.0440,  1.0859, -1.1249, -0.8507,  1.5424],
          [-0.2897,  0.5751,  0.0321, -0.0373, -0.2950]]],


        [[[-0.3194,  1.4513, -0.4586, -1.1734, -0.4004],
          [ 1.5177, -0.6316,  0.0318, -0.6979,  0.0805],
          [-0.0942,  1.3797, -1.8379, -0.5290,  2.6085],
          [ 0.8566,  2.2184, -1.0791, -1.3464, -1.7244]],

         [[-2.0953, -1.3378, -0.0179, -0.4864, -0.4528],
          [-0.4663, -0.2079,  0.4242,  1.1032,  1.1951],
          [ 0.5238, -0.5592, -1.6922,  1.2687,  1.4926],
          [ 0.6145,  0.2707,  0.9933, -0.1689,  1.8934]],

         [[-0.6015, -0.7122,  0.5170,  0.6356, -0.1784],
          [-0.7986,  0.7967,  0.2425,  0.4052,  1.3132],
          [-0.9203,  0.1695,  0.1957, -0.2961,  1.2645],
          [ 0.0646, -0.2755,  0.4801,  1.0499,  0.3506]]]],
       dtype=torch.float64)
>>> labels = (1, 0, 1)
>>> type(labels)
<class 'tuple'>
>>> labels = torch.as_tensor(labels)
>>> type(labels)
<class 'torch.Tensor'>
>>> labels
tensor([1, 0, 1])
>>>
~~~

