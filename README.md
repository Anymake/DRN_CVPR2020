
# DRN and SKU110K-R
#### Xingjia Pan, Yuqiang Ren, Kekai Sheng, Weiming Dong, Haolei Yuan, Xiaowei Guo, Chongyang Ma, Changsheng Xu

### Code and Dataset for CVPR2020 "Dynamic Reﬁnement Network for Oriented and Densely Packed Object Detection" will come soon !

 Dynamic Reﬁnement Network for Oriented and Densely Packed Object Detection[[Paper Link]](https://arxiv.org/abs/2005.09973)

<img src="images/drn.png" width="1000">

Figure 1. Overall framework of our Dynamic Reﬁnement Network. The backbone network is followed by two modules, i.e., feature selection module (FSM) and dynamic reﬁnement heads (DRHs). FSM selects the most suitable features by adaptively adjusting receptive ﬁelds. The DRHs dynamically reﬁne the predictions in an object-aware manner.

## Method
In this work, we adopt CenterNet, with an additional angle prediction head as our baseline and present dynamic reﬁnement network (DRN). Our DRN consists of two novel parts: feature selection module (FSM) and dynamic reﬁnement head (DRH). FSM empowers neurons with the ability to adjust receptive ﬁelds in accordance with the object shapes and orientations, thus passing accurate and denoised features to detectors. DRH enables our model to make ﬂexible inferences in an object-aware manner. Speciﬁcally, we propose two DRHs for classiﬁcation (DRHC) and regression (DRH-R) tasks. In addition, we carefully relabel oriented bounding boxes for SKU110K [9] and called them SKU110K-R; in this manner, oriented object detection is facilitated. To evaluate the proposed method, we conduct extensive experiments on the DOTA, HRSC2016, and SKU110K datasets.

In summary, our contributions include:

• We propose a novel FSM to adaptively adjust the receptive ﬁelds of neurons based on object shapes and orientations. The proposed FSM effectively alleviates the misalignment between receptive ﬁelds and objects.

• We present two DRHs, namely, DRH-C and DRHR, for classiﬁcation and regression tasks, respectively. These DRHs can model the uniqueness and particularity of each sample and reﬁne the prediction in an objectwise manner.

• We collect a carefully relabeled dataset, namely, SKU110K-R, which contains accurate annotations of oriented bounding boxes, to facilitate the research on oriented and densely packed object detection.

• Our method shows consistent and substantial gains across DOTA, HRSC2016, SKU110K, and SKU110KR on oriented and densely packed object detection.

## Dataset
<img src="images/sku110k_r.png" width="1000">

Figure 2. Some sample images from SKU110K. The images in top row are annotated with horizontal bounding boxes while the images in bottom row are with oriented bounding boxes.

On the basis of SKU110K, we propose an extensive variant, namely SKU110K-R, of which each instance is annotated by an oriented bounding box. In the original SKU110K, the orientation angle ranges mainly in [-15 ◦ , 15 ◦ ]. To enrich the orientation, we further do some rotation augmentation from 6 angles (-45 ◦ , -30 ◦ , -15 ◦ , 15 ◦ , 30 ◦ ,45 ◦ ). Fig. 1 shows the statistics of orientation distribution of instances in SKU110k and SKU110K-R. To be compatible with the setting of CenterNet, we use a tuple(cx,cy,w,h,θ) to depict a oriented bounding box. cx,cy are the coordinates of the center point. w,h are the width and height of the object and θ is the orientation angle. Note that we start with y-axis, positive in clockwise direction and negative in counterclockwise direction. All the angles ranges from -90 ◦ to 90 ◦ .
