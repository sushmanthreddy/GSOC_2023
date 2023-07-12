# GSOC 2023
This repository represents work done in gsoc 2023 for INCF and openworm foundation.

The project has two subprojects based on Instance Segmentation:
1. SAM Fine-Tuning(Segment Anything Model)
2. DevoNet (Fully Convolutional Based Network)

## Segment Anything Model

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

The main problem around here is with the that sam is trained on the RGB images and dataset we have is grey scale images and we need to do image processing on grey scale images before the fine tuning the model.

For finetuning  of the images we need the bounding boxes as prompt for training , so my approach would be extracting the prompt from the masks.
So, my approach converting the mask to bounding boxes .

### mask to bounding boxes

```
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

```

* The images for training should be in the the shape of [3,1024,1024] represents  [channel,height,width].
* The masks should be in the shape of [1,256,256] represents [channel,height,width]
* The prompt will be in the shape of [number_of_objects,4].

As part of evaluation was able to convert mask to bounding boxes and due to various number of bounding boxes in each image ,we have problem with the dataloader. To solve this issue by using the collate function , We can pad with the zeroes of each image with max number of bounding boxes .

### next phase work of sam

Is training the sam model by only fine tuning the mask decoder and loss function with dice loss from monai library .
hyperparametric tuning and converting to onnx model and hosting it on hugging face spaces.

# DevoNet

The model is based on NSN and NDN , here we are are trying to convert 3d tiff file for the training the dataset we used is cell tracking challenge dataset.


## Dataset description

* man_seg_T_Z.tif (gold segmentation truth only) - 16-bit multi-page tiff file (segmented objects have unique positive labels that are not necessarily propagated over time, background has zero label). It contains reference segmentation for the Z-th slice from the corresponding original image tT.tif. Not all objects have to be segmented. The man_seg_T_Z.tif file does not have to be provided for every slice of each tT.tif file. Only the slices with non-empty reference segmentation are released.

* man_segT.tif - 16-bit multi-page tiff file (segmented objects have unique positive labels that are not necessarily propagated over time, background has zero label). It contains reference segmentation for the corresponding original image tT.tif. In the case of gold segmentation truth, only selected frames are annotated (i.e., the man_segT.tif file does not have to be provided for every tT.tif file). However, in those frames, all objects are segmented. In the case of silver segmentation truth, all frames tend to be completely annotated. Nevertheless, some objects may be missing there due to their difficulty in being segmented automatically. If that involves all objects in a particular frame, the reference segmentation annotation is not released at all.

* man_trackT.tif - 16-bit multi-page tiff file (markers have unique positive labels propagated over time, background has zero label). It contains ground truth markers for the corresponding original image tT.tif. The man_trackT.tif file is provided for every tT.tif file.


## Model architecture of DevoNet

### NSN Model architecture 

```
NSN Model architecture 
class Model_L2(nn.Module):
    def __init__(
            self,
            ndim=3,
            n_class=2,
            init_channel=2,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=-1,
            loss_func='nn.CrossEntropyLoss'
        ):
        super(Model_L2, self).__init__()
        self.gpu = gpu
        self.pool_size = pool_size
        self.phase = 'train'
        self.loss_func = eval(loss_func)()

        self.c0=nn.Conv3d(1, init_channel, kernel_size, 1, int(kernel_size/2))
        self.c1=nn.Conv3d(init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))

        self.c2=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.c3=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.c4=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.c5=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))

        self.dc0=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0)
        self.dc1=nn.Conv3d(int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.dc2=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.dc3=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0)
        self.dc4=nn.Conv3d(int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.dc5=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))

        self.dc6=nn.Conv3d(int(init_channel * (ap_factor ** 1)), n_class, 1, 1)

        self.bnc0=nn.BatchNorm3d(init_channel)
        self.bnc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.bnc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bnc3=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))

        self.bnc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bnc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))

        self.bndc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bndc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.pool = nn.MaxPool3d(pool_size, pool_size)

    def _calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        syn0 = F.relu(self.bnc1(self.c1(e0)))
        del e0
        e1 = self.pool(syn0)
        e2 = F.relu(self.bnc2(self.c2(e1)))
        syn1 = F.relu(self.bnc3(self.c3(e2)))
        del e1, e2
        e3 = self.pool(syn1)
        e4 = F.relu(self.bnc4(self.c4(e3)))
        e5 = F.relu(self.bnc5(self.c5(e4)))
        del e3, e4
        d0 = torch.cat([self.dc0(e5), syn1], dim=1)
        del e5, syn1
        d1 = F.relu(self.bndc1(self.dc1(d0)))
        d2 = F.relu(self.bndc2(self.dc2(d1)))
        del d0, d1
        d3 = torch.cat([self.dc3(d2), syn0], dim=1)
        del d2, syn0
        d4 = F.relu(self.bndc4(self.dc4(d3)))
        d5 = F.relu(self.bndc5(self.dc5(d4)))
        del d3, d4
        d6 = self.dc6(d5)
        del d5
        return d6

    def forward(self, x, t=None, seg=True):
        h = self._calc(x)
        if seg:
            pred = F.softmax(h, dim=1)
            del h
            return pred.data
        else:
            loss = self.loss_func(h, t.long())
            pred = F.softmax(h, dim=1)
            del h
            return loss, pred.data


```
### NSN Model
This code defines a neural network model called `Model_L2` for a 3D image segmentation task. Let's go through the code and explain its structure and components:

1. The model is implemented as a subclass of `nn.Module`, which is the base class for all neural network modules in PyTorch.

2. The constructor `__init__` defines the initial setup and architecture of the model. It takes several parameters:
   - `ndim`: The number of dimensions in the input data (default: 3).
   - `n_class`: The number of classes for segmentation (default: 2).
   - `init_channel`: The number of initial channels for the first convolutional layer (default: 2).
   - `kernel_size`: The size of the convolutional kernel (default: 3).
   - `pool_size`: The size of the pooling kernel for max pooling (default: 2).
   - `ap_factor`: A factor used to determine the number of channels in each layer (default: 2).
   - `gpu`: Specifies the GPU device to use (-1 for CPU, default: -1).
   - `loss_func`: The loss function to use for training, specified as a string (default: 'nn.CrossEntropyLoss').

3. The model contains various convolutional (`nn.Conv3d`), transposed convolutional (`nn.ConvTranspose3d`), and batch normalization (`nn.BatchNorm3d`) layers. These layers are defined as attributes of the model in the constructor.

4. The `_calc` method is a helper function that performs the forward computation of the model. It takes an input tensor `x` and returns the output tensor after passing through the layers. The method applies a series of convolutions, activations (ReLU), batch normalizations, and max pooling operations to process the input tensor and generate feature maps.

5. The `forward` method implements the forward pass of the model. It takes an input tensor `x`, an optional target tensor `t` (used for computing the loss), and a boolean flag `seg` (indicating whether to perform segmentation or not). The method calls the `_calc` method to obtain the output tensor `h`. If `seg` is True, the output tensor is softmax normalized and returned as `pred`. Otherwise, the method calculates the loss using the specified loss function and returns both the loss and softmax normalized `pred`.

Overall, this code defines a model with an encoder-decoder architecture for 3D image segmentation. It performs convolutions and pooling operations to extract features from the input data and then uses transposed convolutions to upsample the features and generate the final segmentation output.

### NDN model

```
class Model_L4(nn.Module):
    def __init__(
            self,
            ndim=3,
            n_class=2,
            init_channel=2,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=-1,
            loss_func='nn.CrossEntropyLoss'
        ):
        super(Model_L4, self).__init__()
        self.gpu = gpu
        self.pool_size = pool_size
        self.phase = 'train'
        self.loss_func = eval(loss_func)()

        self.c0=nn.Conv3d(1, init_channel, kernel_size, 1, int(kernel_size/2))
        self.c1=nn.Conv3d(init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.c2=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.c3=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.c4=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.c5=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))

        self.c6=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))
        self.c7=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))

        self.c8=nn.Conv3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))
        self.c9=nn.Conv3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 5)), kernel_size, 1, int(kernel_size/2))

        self.dc0=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 5)), int(init_channel * (ap_factor ** 5)), self.pool_size, self.pool_size, 0)
        self.dc1=nn.Conv3d(int(init_channel * (ap_factor ** 4) + init_channel * (ap_factor ** 5)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))
        self.dc2=nn.Conv3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))

        self.dc3=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), self.pool_size, self.pool_size, 0)
        self.dc4=nn.Conv3d(int(init_channel * (ap_factor ** 3) + init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))
        self.dc5=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))

        self.dc6=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0)
        self.dc7=nn.Conv3d(int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.dc8=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.dc9=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0)
        self.dc10=nn.Conv3d(int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.dc11=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))

        self.dc12=nn.Conv3d(int(init_channel * (ap_factor ** 1)), n_class, 1, 1)

        self.bnc0=nn.BatchNorm3d(init_channel)
        self.bnc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.bnc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bnc3=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))

        self.bnc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bnc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))

        self.bnc6=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bnc7=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))

        self.bnc8=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))
        self.bnc9=nn.BatchNorm3d(int(init_channel * (ap_factor ** 5)))
        self.bndc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))
        self.bndc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))
        self.bndc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bndc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bndc7=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc8=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc10=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bndc11=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.pool = nn.MaxPool3d(pool_size, pool_size)

    def _calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        syn0 = F.relu(self.bnc1(self.c1(e0)))
        del e0
        e1 = self.pool(syn0)
        e2 = F.relu(self.bnc2(self.c2(e1)))
        syn1 = F.relu(self.bnc3(self.c3(e2)))
        del e1, e2
        e3 = self.pool(syn1)
        e4 = F.relu(self.bnc4(self.c4(e3)))
        syn2 = F.relu(self.bnc5(self.c5(e4)))
        del e3, e4
        e5 = self.pool(syn2)
        e6 = F.relu(self.bnc6(self.c6(e5)))
        syn3 = F.relu(self.bnc7(self.c7(e6)))
        del e5, e6
        e7 = self.pool(syn3)
        e8 = F.relu(self.bnc8(self.c8(e7)))
        e9 = F.relu(self.bnc9(self.c9(e8)))
        del e7, e8
        d0 = torch.cat([self.dc0(e9), syn3], dim=1)
        del e9, syn3
        d1 = F.relu(self.bndc1(self.dc1(d0)))
        d2 = F.relu(self.bndc2(self.dc2(d1)))
        del d0, d1
        d3 = torch.cat([self.dc3(d2), syn2], dim=1)
        del d2, syn2
        d4 = F.relu(self.bndc4(self.dc4(d3)))
        d5 = F.relu(self.bndc5(self.dc5(d4)))
        del d3, d4
        d6 = torch.cat([self.dc6(d5), syn1], dim=1)
        del d5, syn1
        d7 = F.relu(self.bndc7(self.dc7(d6)))
        d8 = F.relu(self.bndc8(self.dc8(d7)))
        del d6, d7
        d9 = torch.cat([self.dc9(d8), syn0], dim=1)
        del d8, syn0
        d10 = F.relu(self.bndc10(self.dc10(d9)))
        d11 = F.relu(self.bndc11(self.dc11(d10)))
        del d9, d10

        d12 = self.dc12(d11)
        del d11
        return d12

    def forward(self, x, t=None, seg=True):
        h = self._calc(x)
        if seg:
            pred = F.softmax(h, dim=1)
            del h
            return pred.data
        else:
            loss = self.loss_func(h, t.long())
            pred = F.softmax(h, dim=1)
            del h
            return loss, pred.data


```

The code defines a neural network model called `Model_L4`, which is an extension of the `Model_L2` architecture. It is designed for 3D image segmentation tasks and has a deeper architecture with additional convolutional and transposed convolutional layers.

Let's break down the code and understand its structure and components:

1. The `Model_L4` class is a subclass of `nn.Module`, which is the base class for all neural network modules in PyTorch.

2. The constructor `__init__` initializes the model's attributes and defines its architecture. It takes similar parameters as in `Model_L2` and assigns them to corresponding attributes.

3. The model consists of a series of convolutional (`nn.Conv3d`), transposed convolutional (`nn.ConvTranspose3d`), and batch normalization (`nn.BatchNorm3d`) layers. These layers are defined as attributes of the model in the constructor.

4. The `_calc` method is a helper function that performs the forward computation of the model. It takes an input tensor `x` and applies a sequence of convolutions, activations (ReLU), batch normalizations, and max pooling operations to produce the output tensor.

5. The `forward` method implements the forward pass of the model. It takes an input tensor `x`, an optional target tensor `t` (used for computing the loss), and a boolean flag `seg` (indicating whether to perform segmentation or not). The method calls the `_calc` method to obtain the output tensor `h`. If `seg` is True, the output tensor is softmax normalized and returned as `pred`. Otherwise, the method calculates the loss using the specified loss function and returns both the loss and softmax normalized `pred`.

The key difference between `Model_L2` and `Model_L4` is the additional convolutional and transposed convolutional layers in `Model_L4`, resulting in a deeper and more complex architecture. These additional layers aim to capture more intricate patterns and features in the input data, potentially improving the model's segmentation performance.

Overall, `Model_L4` is a more sophisticated variant of the original model, offering increased representational capacity through the inclusion of additional layers while maintaining the general architecture and functionality of `Model_L2`.

## Loss Function :

The most commonly used loss functions for segmentation are based on either the cross entropy loss, Dice loss or a combination of the two. We propose the Unified Focal loss, a new hierarchical framework that generalises Dice and cross entropy-based losses for handling class imbalance

My choice would be the Dice loss.

# Next phase DevoNet

* Training the model and converting it into onnx format and hosting it in hugging face spaces.