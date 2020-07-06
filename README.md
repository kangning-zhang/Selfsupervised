
Current deep learning models benefit greatly from supervised training, especially in medical imaging where superior performance to human experts are observed recently. However, such approaches rely on expert annotations as well, to train the model. Self-supervised learning is a solution to address this problem and emerges in recent years. Though effective, the working principle behind is vague, which might be harmful especially for healthcare-related methods. As a result, this project is proposed to study the underlying principle and explain how modern self-supervised models work on real video data (and ideally on multi-modal data). As self-supervised methods aim at extracting knowledge from the data itself, there must be something explainable behind the ‘magic’. Specifically, modern self-supervised approaches will be analysed both qualitatively and quantitatively on real medical data. Visualisation of the learning process and statistical summarisation will be performed. On the other hand, for multi-modal data, another underlying constraint is added — the correlation among different modalities — so that stronger explainable knowledge could be revealed.

--------------------------------------------------

## General Pipeline

The core of the self-supervised method lies a framing called "pretext task" that is pre-degisned to solve for leaning visual representations by leaning the objective function of the "pretext task". And the representations learned by performing this task can be used as a starting point for our downstream supervised tasks. In some sense, we don't quite care about the final performance of this artifacial task, but what is really interested is the intermediate representation it learned that are expected to carry good semantic or structural meanings. If the learning model can learn representative semantic features, then it can be served as a good start point and leads to better performance on these high-level visual tasks.

<img width="300" alt="1" src="https://user-images.githubusercontent.com/57115537/86545046-a9b20c00-bf23-11ea-98b8-e2d82959d500.png">




# Literature Review

### Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey

the motivation, general pipeline, and terminologies of this field are described

the schema and evaluation metrics of self-supervised learning methods are reviewed


## Self-supervised learning based on image:




## Self-supervised learning works in biomedical image analysis:
Moving from natural image, what we want to look at is biomedical image, and here is some applications on this area.

### Self-supervised Feature Learning for 3D Medical Images by Playing a Rubik’s Cube




### Ultrasound Image Representation Learning by Modeling Sonographer Visual Attention



## Self-supervised learning works in video representation learning
Since we are going to work on vedio instead of image, here is some applications on this area.

### Unsupervised Learning of Video Representations using LSTMs




