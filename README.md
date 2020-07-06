
Current deep learning models benefit greatly from supervised training, especially in medical imaging where superior performance to human experts are observed recently. However, such approaches rely on expert annotations as well, to train the model. Self-supervised learning is a solution to address this problem and emerges in recent years. Though effective, the working principle behind is vague, which might be harmful especially for healthcare-related methods. As a result, this project is proposed to study the underlying principle and explain how modern self-supervised models work on real video data (and ideally on multi-modal data). As self-supervised methods aim at extracting knowledge from the data itself, there must be something explainable behind the ‘magic’. Specifically, modern self-supervised approaches will be analysed both qualitatively and quantitatively on real medical data. Visualisation of the learning process and statistical summarisation will be performed. On the other hand, for multi-modal data, another underlying constraint is added — the correlation among different modalities — so that stronger explainable knowledge could be revealed.

--------------------------------------------------

## General Pipeline

The core of the self-supervised method lies a framing called "pretext task" that is pre-degisned to solve for leaning visual representations by leaning the objective function of the "pretext task". And the representations learned by performing this task can be used as a starting point for our downstream supervised tasks. In some sense, we don't quite care about the final performance of this artifacial task, but what is really interested is the intermediate representation it learned that are expected to carry good semantic or structural meanings. If the learning model can learn representative semantic features, then it can be served as a good start point and leads to better performance on these high-level visual tasks. 

<img width="400" alt="1" src="https://user-images.githubusercontent.com/57115537/86545046-a9b20c00-bf23-11ea-98b8-e2d82959d500.png">

### Common Setup (Pre-training, Feature Extraction and Transfer):

1. Firstly, perform self-supervised pre-training using a self-supervised pretext method on a pre-training dataset
2. Secondly, extract features from various layers of the network. Say for AlexNet, we could do this after every conv layer
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

The figure below illustrates how a puzzle is generated and solved. 
1. randomly crop a 225 × 225 pixel window from an image (red dashed box), divide it into a 3 × 3 grid, and randomly pick a 64 × 64 pixel tiles from each 75 × 75 pixel cell. 
2. These 9 tiles are reordered via a randomly chosen permutation from a predefined permutation set and are then fed to the Context Free Network (CFN). The task is to predict the index of the chosen permutation (technically, defines as output a probability vector with 1 at the 64-th location and 0 elsewhere). The CFN is a siamese-ennead CNN. For simplicity, we do not indicate the maxpooling and ReLU layers. These shared layers are implemented exactly as in AlexNet. 
3. In the transfer learning experiments we show results with the trained weights transferred on AlexNet (precisely, stride 4 on the first layer). The training in the transfer learning experiment is the same as in the other competing methods. Notice instead, that during the training on the puzzle task, we set the stride of the first layer of the CFN to 2 instead of 4.


<img width="800" alt="Screenshot 2020-07-06 at 1 54 02 PM" src="https://user-images.githubusercontent.com/57115537/86595171-3beff980-bf90-11ea-81e5-4a251475c1fb.png">


### Exemplar networks
Perturb/distort image patches, e.g. by cropping and affine transformations and train to classify these exemplars as same class. In this case, every individual image corresponds to its own class, multiple examples are aumentated and triplet loss is used in order to scale this pretext task to a large number of images and classes. Triplet loss is choicen here to encourage examples of the same image to have representations that are close in Euclidean space.

<img width="300" alt="Screenshot 2020-07-06 at 10 33 23 AM" src="https://user-images.githubusercontent.com/57115537/86578752-371d4c80-bf74-11ea-8d17-fdd4507bab15.png">


### Rotation - with applications to video (see in next section)

Gidaris et al. proposed to produce 4 copies of original image by rotating it by {0,90,180,270} degree and the model is trained to predict which ratation is appied, which is a four class classification job. In tuitatively, in order to identify the same image with different rotations, a good model should learn to recognize canonical orientations of objects in natural image, the level object parts, such as heads, noses, and eyes, and the relative positions of these parts, rather than local patterns. This pretext task drives the model to learn semantic concepts of objects in this way.

The RotNet used in the paper:
<img width="600" alt="Screenshot 2020-07-06 at 11 33 50 AM" src="https://user-images.githubusercontent.com/57115537/86600277-c12adc80-bf97-11ea-9414-cb1e0a61c548.png">


### Colourization (could not be transferred to medical image)
Train network to predict pixel colour from a monochrome input

<img width="300" alt="Screenshot 2020-07-06 at 10 33 15 AM" src="https://user-images.githubusercontent.com/57115537/86578729-2ff63e80-bf74-11ea-8d38-83b1e6072a72.png">


## Video-Based (A temporal sequence of frames)
Since we are going to work on vedio instead of image, here is some applications on this area.

### Self-supervised Spatiotemporal Feature Learning by Video Geometric Transformations (Rotation)

In the paper, they propose 3DRotNet: a fully self-supervised approach to learn spatiotemporal features from unlabeled videos. A set of rotations are applied to all videos, and a pretext task is defined as prediction of these rotations. When accomplishing this task, 3DRotNet is actually trained to understand the semantic concepts and motions in videos. In other words, it learns a spatiotemporal video representation, which can be transferred to improve video understanding tasks in small datasets. They test the effectiveness of the 3DRotNet on action recognition task

<img width="600" alt="Screenshot 2020-07-06 at 2 52 48 PM" src="https://user-images.githubusercontent.com/57115537/86603117-81fe8a80-bf9b-11ea-9079-b701440c3690.png">

### Shuffle and Learn: Unsupervised Learning using Temporal Order Verification (Shuffle and Learn)

In this paper, they define the pretext task as determining whether a sequence of frams from the video is in the correct trmporal order. In some sense could be understand as given the starting and the end, could a certaion point in the middle? Such a simple sequential verification task captures important spatiotemporal signals in videos, hence are used to learn powerful visual representation.

<img width="400" alt="Screenshot 2020-07-06 at 3 09 16 PM" src="https://user-images.githubusercontent.com/57115537/86602785-14525e80-bf9b-11ea-8f66-48ac311c5a6f.png">

<img width="600" alt="Screenshot 2020-07-06 at 3 10 55 PM" src="https://user-images.githubusercontent.com/57115537/86602794-187e7c00-bf9b-11ea-87b6-e980e74fc0ab.png">

The representation in fc7 contains complementary information and are transferred for action reconition and human pose reconition.

### Self-Supervised Video Representation Learning with Space-Time Cubic Puzzles (3DCubicPuzzle)

A new self-supervised task called as Space-Time Cubic Puzzles is introduced to train 3D CNNs using large scale video dataset. Given a randomly permuted sequence of 3D spatio-temporal pieces cropped from a video clip, the network is trained to predict their original arrangement.

(How to generate puzzle pieces: consider a spatio-temporal cuboid consisting of 2 × 2 × 4 grid cells for each video, hence there are 16! possible permutations. To avoid ambiguity of similar permutation, sample 4 crops instead of 16, in either spatial or temporal direction. More specifically, the 3D crops are extracted from a 4-cell grid of shape 2×2×1 (colored in blue) or 1 × 1 × 4 (colored in red) along the spatial or temporal dimension respectively. Finally, randomly permute them to make the input. The network must feed the 4 input crops through several convolutional layers, and produce an output probability to each of the possible permutations that might have been sampled.)

Space-time cuboid:
<img width="200" alt="Screenshot 2020-07-06 at 3 27 33 PM" src="https://user-images.githubusercontent.com/57115537/86606834-6b0e6700-bfa0-11ea-88e9-dd7d1f9db735.png">

Example spatial and temporal tuples.:
<img width="600" alt="Screenshot 2020-07-06 at 3 46 49 PM" src="https://user-images.githubusercontent.com/57115537/86606858-72ce0b80-bfa0-11ea-8240-ca73df969bf1.png">

Architecture:
<img width="400" alt="Screenshot 2020-07-06 at 3 27 51 PM" src="https://user-images.githubusercontent.com/57115537/86606848-6fd31b00-bfa0-11ea-97fb-bb3370970ec8.png">

The following analysis are done to demonstrate the effectiveness of the methods:
1) comparison with the random initialization and Kinetics-pretraining (supervised), 2) comparison with our alternative strategies, 3) ablation analysis, 4) comparison with the state-of-the-art methods, and 5) Visualization of the low-level filters and high-level activations.

### Generating Videos with Scene Dynamics

They propose to use generative adversarial networks for video with a spatio-temporal convolutional architecture that untangles the scene’s foreground from the background, which have been shown to have good performance on image generation. The main idea behind generative adversarial networks is to train two networks: a generator network G tries to produce a video, and a discriminator network D tries to distinguish between “real“ videos and “fake” generated videos.


(Video geneartor network: The input to the generator network is a low-dimensional latent code, which is usually sampled from Gaussian. There are two independent streams: a moving foreground pathway of fractionally-strided spatio-temporal convolutions, and a static background pathway of fractionally-strided spatial convolutions, both of which up-sample. These two pathways are combined to create the generated video using a mask from the motion pathway.)

<img width="600" alt="Screenshot 2020-07-06 at 4 01 51 PM" src="https://user-images.githubusercontent.com/57115537/86608131-0d7b1a00-bfa2-11ea-9967-a7aaf9882525.png">

Action classification is used for test the effectiveness. Pre-train the two-stream model with unlabeled videos and then fine-tune the discriminator on the task of interest (e.g., action recognition) using a relatively small set of labeled video. To do action classificationm just replace the last layer (which is a binary classifier) with a K-way softmax classifier and freeze the remaining layers. 


### Self-supervised Feature Learning for 3D Medical Images by Playing a Rubik’s Cube

A novel proxy task, i.e., Rubik’s cube recovery, is formulated to pre-train 3D neural networks. The proxy task involves two operations, i.e., cube rearrangement and cube rotation, which enforce networks to learn translational and rotational invariant features from raw 3D data. 

Rubik’s cube recovery: first partition it into a grid (e.g., 2×2×2) of cubes, and then permute the cubes with random rotations. Like playing a Rubik’s cube, the proxy task aims to recover the original configuration, i.e., cubes are ordered and orientated. This is similar to Jigsaw in 2D, but here cube rotation poeration is added to encourages deep learning networks to leverage more spatial information

<img width="600" alt="Screenshot 2020-07-06 at 4 10 08 PM" src="https://user-images.githubusercontent.com/57115537/86611151-284f8d80-bfa6-11ea-9170-ac66f5d3c4ab.png">

Adapting Pre-trained Weights for Pixel-wise Target Task: 
1. Classification: For the classification task, the pre-trained CNN can be directly used for finetuning
2. Segmentation: For the segmentation task, the pre-trained weights can only be adapted to the encoder part of the fully convolutional network (FCN), e.g. U-Net. The decoder of FCN still needs random initialization, which may wreck the pre-trained feature representation and neutralize the improvement generated by the pre-training.

### Unsupervised Learning of Video Representations using LSTMs










