
## Report for TIPR Assignment 3
### Sonu Dixit
### SR 14432

* python 3, tensorflow 1.13

#### PART 1 Fashion MNIST dataset

**Task 1** : Layer vs Accuracy:  
As the layers are increasing test accuracy is increasing ![part_1_task_1.png](attachment:part_1_task_1.png)

**Task 2**:Filter size versus accuracy:  
        best Result at 3x3 filter  
        ![part_1_task_2.png](attachment:part_1_task_2.png)

**Task 3** : different configurations:  
All relu is giving better accuracy than combination of sigmoid, tanh, relu, swish. All swish gives comparable result as all relu.

**Task 4** : Truncated Normal (Xavier initialization) is giving best accuracy.

**Task 5** : Clustering Accuracy versus Train data percentage
![part_1_task_5.png](attachment:part_1_task_5.png)
* with 10% data accuracy is 89%. Network is able to figure out the difference, but is not sure of which features belong to which particular class. Network is poor at mapping features to particular class, but it has learned to extract discriminating features.

**Task 6:** Different embedding dimensions and its TSNE 2 dimensional visualization  
32 dimensional embedding, and its visualization for different perplexity
![part_1_task_6_1.png](attachment:part_1_task_6_1.png)

![part_1_task_6_2.png](attachment:part_1_task_6_2.png)

![part_1_task_6_3.png](attachment:part_1_task_6_3.png)  


**64** dimensional embedding, and its visualization for different perplexity  
![part_1_task_6_1.png](attachment:part_1_task_6_1.png)
* clustering tendecy is there but regions are not well seperated
![part_1_task_6_2.png](attachment:part_1_task_6_2.png)
![part_1_task_6_3.png](attachment:part_1_task_6_3.png)
* There is not much improvement from perplexity 30 to 50. 
![part_1_task_6_4.png](attachment:part_1_task_6_4.png)
* all clusters are almost completely seperated at perplexity 50.

** Task 7** CNN is giving better performance than simple MLP.
* MLP hidden(64,64,10) after 40 iterations is giving 37 percent accuracy. In 3rd epoch itself it is giving 28% accuracy. But after that it has improved very slowly. MLP is very slow to learn spatial dependency in the images. After 20 epochs itself, accuracy is 36%, in epochs 20-40, accuracy is almost stagnant.


### Part 2 CIFAR 10 Dataset

**Task 1**: The plot is for 20 epochs of training. Each conv layer having 3x3 filter, having 64 filters. ![part_2_task_1.png](attachment:part_2_task_1.png)
* As the layer count increases performance should have improved, 
* i think, 20 epochs are not good enopugh to train deep networks, so the performance for deeper nets is less as compared to shallower nets.

**Task 2:** Best acuuracy is given by 3x3 filter size
![part_2_task_2.png](attachment:part_2_task_2.png) It shows the dependency is local, so if filter size is increased, model is trying to capture broader dependency, but the dependency is local, so _3x3_ filter is neither too small nor too big, hence gives best performance.

**Task 3** : different configurations:  
* All relu is giving better accuracy than combination of sigmoid, tanh, relu, swish. 
* All swish gives comparable result as all relu. 
* Relu models require less time as compared to that of combination of sigmoid,relu, swish, tanh.

**Task 4** : Truncated Normal (Xavier initialization) is giving best accuracy.

**Task 5** : Clustering Accuracy versus Train data percentage


**Task 6** : embedding Visualization using tsne  
* 100 dimensional embedding for different perplexity in tsne visualization:
![part_2_task_4_1.png](attachment:part_2_task_4_1.png)
![part_2_task_4_2.png](attachment:part_2_task_4_2.png)
![part_2_task_4_3.png](attachment:part_2_task_4_3.png)
* 64 dimensional embedding for differnt perplexity
![part_2_task_4.png](attachment:part_2_task_4.png)
![part_2_task_4_2.png](attachment:part_2_task_4_2.png)
![part_2_task_4_3.png](attachment:part_2_task_4_3.png)
![part_2_task_4_4.png](attachment:part_2_task_4_4.png)

**task 7:** CNN is giving better accuracy. 
* MLNN with 512,256,64,64 hidden units, all relu activation. Initial accuracy is 25%, after 25 epochs accuracy is 35%. After 60 epochs accuracy is 42%. Model is slowly improving. 
* CNN after 25 epochs gives 60% accuracy.MLNN has a lot more parameters than CNN.
