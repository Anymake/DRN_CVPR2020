
# DRN and SKU110K-R
#### Xingjia Pan, Yuqiang Ren, Kekai Sheng, Weiming Dong, Haolei Yuan, Xiaowei Guo, Chongyang Ma, Changsheng Xu

Code and Dataset for CVPR2020 "Dynamic Reﬁnement Network for Oriented and Densely Packed Object Detection" [[Paper link]](https://arxiv.org/pdf/temp.pdf)

<!---[alt text](figures/teaser_width.jpg)--->
<img src="images/sku110k_r.png" width="1000">

Figure 1. Some sample images from SKU110K. The images in top row are annotated with horizontal bounding boxes while the images in bottom row are with oriented bounding boxes.

On the basis of SKU110K, we propose an extensive variant, namely SKU110K-R, of which each instance is annotated by an oriented bounding box. In the original SKU110K, the orientation angle ranges mainly in [-15 ◦ , 15 ◦ ]. To enrich the orientation, we further do some rotation augmentation from 6 angles (-45 ◦ , -30 ◦ , -15 ◦ , 15 ◦ , 30 ◦ ,45 ◦ ). Fig. 1 shows the statistics of orientation distribution of instances in SKU110k and SKU110K-R. To be compatible with the setting of CenterNet, we use a tuple(cx,cy,w,h,θ) to depict a oriented bounding box. cx,cy are the coordinates of the center point. w,h are the width and height of the object and θ is the orientation angle. Note that we start with y-axis, positive in clockwise direction and negative in counterclockwise direction. All the angles ranges from -90 ◦ to 90 ◦ .

### Our novel contributions are:
1. **Soft-IoU layer**, added to an object detector to estimate the Jaccard index between the detected box and the (unknown) ground truth box.
2. **EM-Merger unit**, which converts detections and Soft-IoU scores into a MoG (Mixture of Gaussians), and resolves overlapping detections in packed scenes.
3. **A new dataset and benchmark**, the store keeping unit, 110k categories (SKU-110K), for item detection in store shelf images from around the world.

## Introduction
In our SKU-110K paper[1] we focus on detection in densely packed scenes, where images contain many objects, often looking similar or even identical, positioned in close proximity. These scenes are typically man-made, with examples including retail shelf displays, traffic, and urban landscape images. Despite the abundance of such environments, they are under-represented in existing object detection benchmarks, therefore, it is unsurprising that state-of-the-art object detectors are challenged by such images.


## Method
We propose learning the Jaccard index with a soft Intersection over Union (Soft-IoU) network layer. This measure provides valuable information on the quality of detection boxes. Those detections can be represented as a Mixture of Gaussians (MoG), reflecting their locations and their Soft-IoU scores. Then, an Expectation-Maximization (EM) based method is then used to cluster these Gaussians into groups, resolving detection overlap conflicts. 

<img src="figures/system.jpg" width="750">

System diagram: (a) Input image. (b) A base network, with bounding box (BB) and objectness (Obj.) heads, along
with our novel Soft-IoU layer. (c) Our EM-Merger converts Soft-IoU to Gaussian heat-map representing (d) objects captured by
multiple, overlapping bounding boxes. (e) It then analyzes these box clusters, producing a single detection per object


## Dataset

<img src="figures/benchmarks_comparison.jpg" width="750">

We compare between key properties for related benchmarks. **#Img.**: Number of images. **#Obj./img.**: Average items per image. **#Cls.**: Number of object classes (more implies a harder detection problem due to greater appearance variations). **#Cls./img.**: Average classes per image. **Dense**: Are objects typically densely packed together, raising potential overlapping detection problems?. **Idnt**: Do images contain multiple identical objects or hard to separate object sub-regions?. **BB**: Bounding box labels available for measuring detection accuracy?.

The dataset is provided for the exclusive use by the recipient and solely for academic and non-commercial purposes. 

##  CVPR 2020 Challenge
The detection challenge will be held in CVPR 2020 Retail-Vision workshop.
Please visit our [workshop page](https://retailvisionworkshop.github.io/) for more information. The data and evaluation code are available in the [challenge page](https://retailvisionworkshop.github.io/detection_challenge_2020/).

## Qualitative Results
Qualitative detection results on SKU-110K. 

<img src="figures/qualitative.jpg" width="750">

## Notes

**Please note that the main part of the code has been released, though we are still testing it to fix possible glitches. Thank you.**

This implementation is built on top of https://github.com/fizyr/keras-retinanet.
The SKU110K dataset is provided in csv format compatible with the code CSV parser.

Dependencies include: `keras`, `keras-resnet`, `six`, `scipy`. `Pillow`, `pandas`, `tensorflow-gpu`, `tqdm`
This repository requires `Keras 2.2.4` or higher, and was tested using `Python 3.6.5`, `Python 2.7.6`  and `OpenCV 3.1`.

The output files will be saved under "$HOME"/Documents/SKU110K and have the same structure as in https://github.com/fizyr/keras-retinanet:
The weight h5 files will are saved in the "snapshot" folder and the tensorboard log files are saved in the "logs" folder.

Note that we have made several upgrades to the baseline detector since the beginning of this research, so the latest version can actually
achieve even higher results than the ones originally reported.

The EM-merger provided here is the stable version (not time-optimized). Some of the changes required for
optimization are mentioned in the TO-DO comments.

Contributions to this project are welcome.

## Usage

TODO

## References
[1] Eran Goldman*, Roei Herzig*, Aviv Eisenschtat*, Jacob Goldberger, Tal Hassner, [Precise Detection in Densely Packed Scenes](https://arxiv.org/abs/1904.00853), 2019.

[2] Tsung-Yi Lin, Priyal Goyal, Ross Girshick, Kaiming He, Piotr Dollar, [Focal loss for dense object detection](https://arxiv.org/abs/1708.02002), 2018.


## Citation

```
@inproceedings{goldman2019dense,
 author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
 title     = {Precise Detection in Densely Packed Scenes},
 booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
 year      = {2019}
}
```
