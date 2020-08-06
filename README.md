
# DRN and SKU110K-R
#### Xingjia Pan, Yuqiang Ren, Kekai Sheng, Weiming Dong, Haolei Yuan, Xiaowei Guo, Chongyang Ma, Changsheng Xu

### Work in process.

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

## SKU110K-R
<img src="images/sku110k_r.png" width="1000">

Figure 2. Some sample images from SKU110K. The images in top row are annotated with horizontal bounding boxes while the images in bottom row are with oriented bounding boxes.

To use SKU110K-R,
0. Download the original SKU110K data set from [websit](https://github.com/eg4000/SKU110K_CVPR19) and extract images
1. Generate SKU110-R using our rotate augment script
```
   python rotate_augment.py path/to/images
```
2. Download the annotations for SKU110K-R from [website](https://drive.google.com/file/d/1_5JsVc_A5vWm-d-JXMJdX0Lx5FIlgAXJ/view?usp=sharing)
The annotation is in coco format.

## Citation

If you find this project useful for your research, please use the following BibTeX entry.
```
@article{pan2020dynamic,
  title={Dynamic Refinement Network for Oriented and Densely Packed Object Detection},
  author={Xingjia Pan and Yuqiang Ren and Kekai Sheng and Weiming Dong and Haolei Yuan and Xiaowei Guo and Chongyang Ma and Changsheng Xu},
  booktitle={CVPR},
  pages={1--8},
  year={2020}
}
```
## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.  
Xingjia Pan: xingjia.pan@nlpr.ia.ac.cn  
Yuqiang Ren: condiren@tencent.com
