
Current deep learning models benefit greatly from supervised training, especially in medical imaging where superior performance to human experts are observed recently. However, such approaches rely on expert annotations as well, to train the model. Self-supervised learning is a solution to address this problem and emerges in recent years. Though effective, the working principle behind is vague, which might be harmful especially for healthcare-related methods. As a result, this project is proposed to study the underlying principle and explain how modern self-supervised models work on real video data (and ideally on multi-modal data). As self-supervised methods aim at extracting knowledge from the data itself, there must be something explainable behind the ‘magic’. Specifically, modern self-supervised approaches will be analysed both qualitatively and quantitatively on real medical data. Visualisation of the learning process and statistical summarisation will be performed. On the other hand, for multi-modal data, another underlying constraint is added — the correlation among different modalities — so that stronger explainable knowledge could be revealed.

--------------------------------------------------

## General Pipeline

The core of the self-supervised method lies a framing called "pretext task" that is pre-degisned to solve for leaning visual representations by leaning the objective function of the "pretext task". And the representations learned by performing this task can be used as a starting point for our downstream supervised tasks. In some sense, we don't quite care about the final performance of this artifacial task, but what is really interested is the intermediate representation it learned that are expected to carry good semantic or structural meanings. If the learning model can learn representative semantic features, then it can be served as a good start point and leads to better performance on these high-level visual tasks. 

<img width="400" alt="1" src="https://user-images.githubusercontent.com/57115537/86545046-a9b20c00-bf23-11ea-98b8-e2d82959d500.png">

### Common Setup (Pre-training, Feature Extraction and Transfer):

1. Firstly, perform self-supervised pre-training using a self-supervised pretext method on a pre-training dataset
2. Secondly, extract features from various layers of the network. Say for AlexNet, we do this after every conv layer
3. Finally, evaluate quality of these features (from different self-supervised approaches) by transfer learning


## Image-Based

Many ideas have been proposed for self-supervised representation learning on images. A common workflow is to train a model on one or multiple pretext tasks with unlabelled images and then use one intermediate feature layer of this model to feed a multinomial logistic regression classifier on ImageNet classification. The final classification accuracy quantifies how good the learned representation is.

### Relative positioning
Doersch et al. (2015) train network to predict relative position of two regions in the same image since a model needs to understand the spatial context of objects in order to tell the relative position between parts. Randomly sample the first patch without any reference to image content. Considering that the first patch is placed in the middle of a 3x3 grid, and the second patch is sampled from its 8 neighboring locations around it. In order to avoid trivial solutions with which model only catch low-level trivial signals, additional noise is added by including a gap and jittering the patch locations.

<img width="500" alt="Screenshot 2020-07-06 at 10 53 47 AM" src="https://user-images.githubusercontent.com/57115537/86580758-0a1e6900-bf77-11ea-9eff-879edfd796f3.png">

Evaluation including: Predict the bounding boxes of all objects of a given class in an image (if any)/ Pre-train CNN using self-supervision (no labels) then Train CNN for detection in R-CNN object category detection pipeline

<img width="600" alt="Screenshot 2020-07-06 at 10 53 32 AM" src="https://user-images.githubusercontent.com/57115537/86580726-ff63d400-bf76-11ea-90ce-26ffedc100ed.png">

### Jigsaw puzzle
The model is trained to place 9 shuffled patches back to the original locations. A convolutional network processes each patch independently with shared weights and outputs a probability vector per patch index out of a predefined set of permutations. To control the difficulty of jigsaw puzzles, the paper proposed to shuffle patches according to a predefined permutation set and configured the model to predict a probability vector over all the indices in the set.



### Colourization (could not be transferred to medical image)
Train network to predict pixel colour from a monochrome input

<img width="600" alt="Screenshot 2020-07-06 at 10 33 15 AM" src="https://user-images.githubusercontent.com/57115537/86578729-2ff63e80-bf74-11ea-8d38-83b1e6072a72.png">


### Exemplar networks
Perturb/distort image patches, e.g. by cropping and affine transformations and train to classify these exemplars as same class

<img width="600" alt="Screenshot 2020-07-06 at 10 33 23 AM" src="https://user-images.githubusercontent.com/57115537/86578752-371d4c80-bf74-11ea-8d17-fdd4507bab15.png">


### Rotation - with applications to video



## Video-Based
Since we are going to work on vedio instead of image, here is some applications on this area.

### Unsupervised Learning of Video Representations using LSTMs


### Self-supervised Spatiotemporal Feature Learning by Video Geometric Transformations



## Self-supervised learning works in biomedical image analysis:
Moving from natural image, what we want to look at is biomedical image, and here is some applications on this area.

### Self-supervised Feature Learning for 3D Medical Images by Playing a Rubik’s Cube



### Ultrasound Image Representation Learning by Modeling Sonographer Visual Attention







