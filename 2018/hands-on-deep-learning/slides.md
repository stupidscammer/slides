class: middle
background-image: url(img/brain.png)

# Hands on .red[deep learning]

.footnote[Alexandre Boucaud  -  [@alxbcd][twitter]]

[twitter]: https://twitter.com/alxbcd
---
class: middle
background-image: url(img/brain.png)
.hidden[aa]
# Hands on .red[deep learning]
.small[with [Keras][keras] examples]
.footnote[Alexandre Boucaud  -  [@alxbcd][twitter]]

---
name: intro
background-image: url(img/news.png)

---
class: center, middle
<img src="img/arxiv-nov18.png" , width="800px" / >

---
## What does "deep" means ?

<img src="img/imagenet.png" , width="800px" / >


---
## A bumpy 60-year history

<img src="img/dl-history1.png" , width="800px" / >

--
<img src="img/dl-history2.png" , width="800px" / >

--
<img src="img/dl-history3.png" , width="800px" / >

---
## An unprecedented trend 

deep learning computation power .red[doubles every 3.5-month]

.center[
  <img src="img/dl_compute.png" , width="500px" / >
]

.footnote[[_AI and Compute_ blog post, Amodei & Hernandez, 16-05-2018](https://blog.openai.com/ai-and-compute/)]

---
exclude: true
class: center, middle

## QUESTION:


### Why this recent .red[trend] ?

---
## Why this recent trend ?

- .medium[specialized .blue[hardware]] .right[e.g. GPU, TPU, Intel Xeon Phi]

--
- .medium[.blue[data] availability] .right[_big data era_]

--
- .medium[.blue[algorithm] research] .right[e.g. adversarial or reinforcement learning]

--
- .medium[.blue[open source] tools] .right[huge ecosystem right now]

---
## Graphics Processing Unit (GPU)

- **< 2000** : "graphics cards" (video edition + game rendering)

.center[<img src="img/msi.jpg" , width="450px">]

---
count: false
## Graphics Processing Unit (GPU)

- **< 2000** : "graphics cards" (video edition + game rendering)
- .hidden[< ]**1999** : nviDIA coins the term "GPU" for the first time

--
- .hidden[< ]**2001** : floating point support on graphics processors

--
- .hidden[< ]**2005** : programs start to be faster on GPU than on CPU

--
- .hidden[< ]**2007** : first release of the [CUDA](https://developer.nvidia.com/cuda-zone) framework

.center[<img src="img/cuda.jpg" , width="400px">]

---
count: false
## Graphics Processing Unit (GPU)

- **< 2000** : "graphics cards" (video edition + game rendering)
- .hidden[< ]**1999** : nviDIA coins the term "GPU" for the first time
- .hidden[< ]**2001** : floating point support on graphics processors
- .hidden[< ]**2005** : programs start to be faster on GPU than on CPU
- .hidden[< ]**2007** : first release of the [CUDA](https://developer.nvidia.com/cuda-zone) framework
- .hidden[< ]**2018** : GPUs are part of our lives (phones, computers, cars, etc..)

.center[<img src="img/nvidia-titan-v.jpg" , width="300px", vspace="50px"/ >]
.footnote[credit: NviDIA Titan V]

---
## Computational power 
GPU architectures are .blue[excellent] for the kind of computations required by the training of NN

.center[<img src="img/tensor_core2.png" , width="600px", vspace="20px">]

| year | hardware | computation (TFLOPS) | price (K$) |
|------|:------:|:-----------:|:----:|
| 2000 | IBM ASCI White | 12 | 100 000 K |
| 2005 | IBM Blue Gene/L | 135 | 40 000 K |
| 2018 | NviDIA Titan V | 110 | 3 |

---
## Deep learning software ecosystem

.center[
  <img src="img/frameworks.png" width="800px" vspace="30px"/>
]

---
## Deep learning today

.left-column[
- translation
- image captioning
- speech synthesis
- style transfer
]

.right-column[
- cryptocurrency mining
- self-driving cars
- games 
- etc.
]

.reset-column[]
.center[
  <img src="img/dl_ex1.png" width="700px" vspace="30px"/>
]

---
## Deep learning today

.center[
  <img src="img/dl_ex2.png" width="800px"/>
]

???
But we are .red[still far]* from "Artificial Intelligence" 

.footnote[*[nice post](https://medium.com/@mijordan3/artificial-intelligence-the-revolution-hasnt-happened-yet-5e1d5812e1e7) by M. Jordan]

---
## Deep learning today

.center[
<img src="img/WaveNet.gif" style="width: 500px;" vspace="80px" />
]

.footnote[[Tacotron 2][tacotron] & [WaveNet][wavenet] - TTS with sound generation - DeepMind (2017)]

[tacotron]: https://google.github.io/tacotron/publications/tacotron2/index.html
[wavenet]: https://deepmind.com/blog/high-fidelity-speech-synthesis-wavenet/

---
class: center, middle

## QUESTION:


### Can you guess which sound is .red[generated] ?

1)  <audio controls><source src="img/lipstick_gen.wav"></audio>
</br> 
2) <audio controls><source src="img/lipstick_gt.wav"></audio>

---
exclude: true
class: center, middle

# For ML applications in astro see Emille's talk later

---

# Outline

.medium[[Neural nets](#nns)]

> hidden layers - activation - backpropagation - optimization

--

.medium[[Convolutional Neural Networks (CNN)](#cnn)]

> kernels - strides - pooling - loss - training


--
.medium[[In practice](#practice)]

> step-by-step - monitoring your training


--
.medium[[Common optimizations](#optim)]

> data augmentation - dropout - batch normalisation

---
## Foreword

The following slides provide examples of neural network models written in _Python_, using the [Keras][keras] library and [TensorFlow][tf] tensor ordering convention*. 

Keras provides a high level API to create deep neural networks and train them using numerical tensor libraries (_backends_) such as [TensorFlow][tf], [CNTK][cntk] or [Theano][theano].


[keras]: https://keras.io/
[tf]: https://www.tensorflow.org/
[cntk]: https://docs.microsoft.com/fr-fr/cognitive-toolkit/
[theano]: http://www.deeplearning.net/software/theano/

.center[
  <img src='img/kerastf.jpeg', width="300px", vspace="30px"/>
]

.footnote[*channels last]

---
class: middle, center
name: nns

# What is a .red[neural network] made of ?

---
## A Neuron

A neuron is a .green[linear system] with two attributes
> the weight matrix $\mathbf{W}$  
> the linear bias $b$

It takes .green[multiple inputs] (from $\mathbf{x}$) and returns .green[a single output]
> $f(\mathbf{x}) = \mathbf{W} . \mathbf{x} + b $
.center[
  <img src="img/neuron.svg" width="600px" />
]

---
## Linear layers

A linear layer is an .green[array of neurons].

A layer has .green[multiple inputs] (same $\mathbf{x}$ for each neuron) and returns .green[multiple outputs].

.center[
  <!-- <img src="img/linear_layer.jpeg" width="450px" /> -->
  <img src="img/mlp_bkg.svg" width="450px" vspace="40px"/>
]

---
## Hidden layers

All layers internal to the network (not input or output layer) are considered .green[hidden layers].

.center[<img src="img/mlp.jpg" width="600px" vspace="50px" />]

.footnote[[cs231n.github.io](http://cs231n.github.io/)]

---
## Multi-layer perceptron (MLP)


.left-column[
```python
from keras.models import Sequential
from keras.layers import Dense

# initialize model
model = Sequential()

# add layers
model.add(Dense(4, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))
```
]

.right-column[
<img src="img/mlp.jpg" width="350px" vspace="30px" hspace="30px" />
]

---
count: false
## Multi-layer perceptron (MLP)
.left-column[
```python
from keras.models import Sequential
from keras.layers import Dense

# initialize model
model = Sequential()

# add layers
model.add(Dense(4, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))
```
]
.right-column[
<img src="img/mlp.jpg" width="350px" vspace="30px" hspace="30px" />
]

.hidden[aa]
.reset-column[]
.center[
.huge[QUESTION:]</br></br>
.big[How many .red[free parameters] has this model ?]
]

---
count: false
## Multi-layer perceptron (MLP)
.left-column[
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(4, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))

# print model structure
model.summary()
```
]
.right-column[
<img src="img/mlp.jpg" width="350px" vspace="30px" hspace="30px" />
]
.reset-column[
```
__________________________________________________
Layer (type)          Output Shape        Param #
==================================================
dense_1 (Dense)       (None, 4)           16         <=   W (3, 4)   b (4, 1)
__________________________________________________
```
]

---
count: false
## Multi-layer perceptron (MLP)
.left-column[
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(4, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))

# print model structure
model.summary()
```
]
.right-column[
<img src="img/mlp.jpg" width="350px" vspace="30px" hspace="30px" />
]
.reset-column[
```
__________________________________________________
Layer (type)          Output Shape        Param #
==================================================
dense_1 (Dense)       (None, 4)           16
__________________________________________________
dense_2 (Dense)       (None, 4)           20         <=   W (4, 4)   b (4, 1)
__________________________________________________
```
]

---
count: false
## Multi-layer perceptron (MLP)
.left-column[
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(4, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))

# print model structure
model.summary()
```
]
.right-column[
<img src="img/mlp.jpg" width="350px" vspace="30px" hspace="30px" />
]
.reset-column[
```
__________________________________________________
Layer (type)          Output Shape        Param #
==================================================
dense_1 (Dense)       (None, 4)           16
__________________________________________________
dense_2 (Dense)       (None, 4)           20
__________________________________________________
dense_3 (Dense)       (None, 1)           5          <=   W (4, 1)   b (1, 1)
==================================================
Total params: 41
Trainable params: 41
Non-trainable params: 0
```
]

---
exclude: True

## Multi-layer perceptron (MLP)

```python
from keras.models import Sequential
from keras.layers import Dense

# initialize model
model = Sequential()

# add layers
model.add(Dense(4, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))
```

--
exclude: True
```python
# print model structure
model.summary()
```

--
exclude: True
```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 20
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 41
Trainable params: 41
Non-trainable params: 0
```

---

## Adding non linearities

A network with several linear layers remains a .green[linear system].

--

To add non-linearities to the system, .red[activation functions] are introduced. 

.center[<img src="img/artificial_neuron.svg" width="600px" />]

---
count: false
## Adding non linearities

A network with several linear layers remains a .green[linear system].

To add non-linearities to the system, .red[activation functions] are introduced. 

.center[<img src="img/feedforwardnn.gif" width="400px" />]

.footnote[credit: Alexander Chekunkov]

---

## Activation functions 

.hidden[a]  
.left-column[.blue[activation function] ]
.green[its derivative]


.center[<img src="img/activation_functions.svg" width="750px" vspace="30px" />]


An extended list of activation functions can be found on [wikipédia](https://en.wikipedia.org/wiki/Activation_function).

---

## Activation layer

There are two different syntaxes whether the activation is seen as a .green[property] of the neuron layer

```python
model = Sequential()
model.add(Dense(4, input_dim=3, activation='sigmoid'))
```

--

or as an .green[additional layer] to the stack

```python
from keras.layers import Activation

model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Activation('tanh'))
```

--
The activation layer .red[does not add] any .red[depth] to the network.


---
## Backpropagation

A .green[30-years old] algorithm (Rumelhart et al., 1986).

The network is a composition of .green[differentiable] modules.

The .red[chain rule] can be applied.


.center[<img src="img/chain_rule.svg" width="500px" />]

---
## Backpropagation

A .green[30-years old] algorithm (Rumelhart et al., 1986)

.hidden[a]

.center[<img src="img/backpropagation.gif" width="800px" />]

.footnote[credit: Alexander Chekunkov]

---
## Backpropagate gradients

### Compute activation gradients
- $\nabla\_{\mathbf{z}^o(\mathbf{x})} l = \mathbf{f(x)} - \mathbf{e}(y)$

--

### Compute layer params gradients 
- $\nabla\_{\mathbf{W}^o} l = \nabla\_{\mathbf{z}^o(\mathbf{x})} l \cdot \mathbf{h(x)}^\top$
- $\nabla\_{\mathbf{b}^o} l = \nabla\_{\mathbf{z}^o(\mathbf{x})} l$

--

### Compute prev layer activation gradients
- $\nabla\_{\mathbf{h(x)}} l = \mathbf{W}^{o\top} \nabla\_{\mathbf{z}^o(\mathbf{x})} l$
- $\nabla\_{\mathbf{z}^h(\mathbf{x})} l = \nabla\_{\mathbf{h(x)}} l \odot \mathbf{\sigma^\prime(z^h(x))}$


---
class: middle, center

## QUESTION:

### How would you feed images to a network ?

---
class: middle, center
name: cnn

# .red[Convolutional] Neural Networks

---

## Convolutional Neural Networks

- elegant way of passing .green[tensors] to a network
- perform convolutions with .green[3D kernels] 
- training optimizes kernels, not neuron weights

.center[
  <img src="img/cnn_overview2.png", width="800px", vspace="30px", hspace="0px"/>
]

---
## Convolutional layers

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
# First conv needs input_shape
# Shape order depends on backend
model.add(
    Conv2D(15,       # filter size 
           (3, 3),   # kernel size
           strides=1,       # default
           padding='valid', # default
           input_shape=(32, 32, 3)))
# Next layers don't
model.add(Conv2D(16, (3, 3) strides=2))
model.add(Conv2D(32, (3, 3)))
```
]

.right-column[
<img src="img/convlayer2.jpg" width="300px" vspace="50px", hspace="50px"/>
] 

.reset-columns[
  <br/> <br/> <br/> <br/> <br/> <br/> <br/> <br/> <br/>  <br/> 
- **kernel properties**: .green[size] and number of .green[filters]
- **convolution properties**: .green[strides] and .green[padding]
- output shape depends on .red[**all**] these properties
]


---
## No strides, no padding

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(
    Conv2D(1, (3, 3), 
           strides=1,        # default
           padding='valid',  # default
           input_shape=(7, 7, 1)))
model.summary()
```

```
_________________________________________
Layer (type)            Output Shape     
=========================================
conv2d (Conv2D)         (None, 5, 5, 1)  
=========================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________
```
] 
.right-column[
<img src="img/convolution_gifs/full_padding_no_strides_transposed.gif" width="350px"/>
] 


.footnote[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]
---
## Strides (2,2) + padding

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(1, (3, 3), 
*                strides=2, 
                 padding='same', 
                 input_shape=(5, 5, 1)))
model.summary()
```

```
_________________________________________
Layer (type)            Output Shape     
=========================================
conv2d (Conv2D)         (None, 3, 3, 1)  
=========================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________
```
]
.right-column[ 
<img src="img/convolution_gifs/padding_strides.gif" />
]

.footnote[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]

---
## Activation

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(1, (3, 3), 
*                activation='relu'
                 input_shape=(5, 5, 1)))
```
]

.right-column[ 
<img src="img/relu.jpeg" width="250px"  hspace="60px"/>
]

.reset-columns[
  </br>
  </br>
  </br>
  </br>
  </br>
  </br>
- safe choice*: .medium.red[use ReLU] for the convolutional layers
- select the activation of the last layer according to your problem
.small[e.g. sigmoid for binary classification]
]

.footnote[*not been proven (yet) but adopted empirically]
---
## Pooling layers

- reduces the spatial size of the representation (downsampling)<br/>
=> less parameters & less computation
- common method: **`MaxPooling`** or **`AvgPooling`**
- common strides: (2, 2)

.center[
  <img src="img/maxpool.jpeg" width="600px" vspace="20px"/>
]
.footnote[[cs231n.github.io](http://cs231n.github.io/)]

---
## Pooling layers

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D

model = Sequential()
model.add(Conv2D(1, (3, 3), 
                 strides=1, 
                 padding='same', 
                 input_shape=(8, 8, 1)))
model.add(MaxPool2D(((2, 2))))
model.summary()
```

```
__________________________________________________
Layer (type)          Output Shape        Param #
==================================================
conv2d_1 (Conv2D)     (None, 8, 8, 1)     10
__________________________________________________
max_pooling2d_1 (MaxP (None, 4, 4, 1)     0
==================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
__________________________________________________
```
]
.right-column[ 
  <img src="img/maxpool.jpeg" width="350px" vspace="50px" hspace="30px" />
]

---
class: center, middle

# EXERCICE
.medium[on your own time, .red[write down the model] for the following architecture ]

<img src="img/cnn_overview2.png", width="700px", vspace="30px", hspace="0px"/>

.medium[how many .red[free parameters] does this architecture have ?]

---
## Loss and optimizer

Once your architecture (`model`) is ready, a [loss function](https://keras.io/losses/) and an [optimizer](https://keras.io/optimizers/) .red[must] be specified 
```python
model.compile(optimizer='adam', loss='binary_crossentropy')
```
or with better access to optimization parameters
```python
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

model.compile(optimizer=Adam(lr=0.01, decay=0.1), 
              loss=binary_crossentropy)
```

Choose both according to the target output.

---
## Training

It's time to .green[train] your model on the data (`X_train`, `y_train`). 

```python
model.fit(X_train, y_train,
          batch_size=32,        
          epochs=50,  
          validation_split=0.3) # % of data being used for val_loss evaluation

```

- **`batch_size`**: .green[\# of images] used before updating the model<br/>
  32 is a very good compromise between precision and speed*
- **`epochs`**: .green[\# of times] the model is trained with the full dataset

After each epoch, the model will compute the loss on the validation set to produce the **`val_loss`**. 

.red[The closer the values of **`loss`** and **`val_loss`**, the better the training]. 

.footnote[*see [Masters et al. (2018)](https://arxiv.org/abs/1804.07612)]

---
## Callbacks

[Callbacks](https://keras.io/callbacks/) are methods that act on the model during training, e.g.

```python
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# Save the weights of the model based on lowest val_loss value
chkpt = ModelCheckpoint('weights.h5', save_best_only=True)
# Stop the model before 50 epochs if stalling for 5 epochs
early = EarlyStopping(patience=5)

model.fit(X_train, y_train,
          epochs=50,
          callbacks=[chkpt, early])
```
--
- ModelCheckpoint saves the weights, which can be reloaded
  ```python
  model.load_weights('weights.h5')  # instead of model.fit()
  ```
- EarlyStopping saves the planet.

---
class: center, middle
name: practice

# In practice

---
## The right architecture
<!-- class: middle -->

There is currently .red[no magic recipe] to find a network architecture 
that will solve your particular problem.

.center[
  # `¯\_(ツ)_/¯`
]

But here are some advice for non-specialists to guide you in the right direction and/or 
get you out of trouble.

---
## Start with existing (working) models

- look for a relevant architecture for your problem  
  (arXiv, blogs, websites)

.center[
<img src="img/ssd.png" width="600px" />
]

---
count: false
## Start with existing (working) models

- look for a relevant architecture for your problem  
  (arXiv, blogs, websites)
- find an implementation on [GitHub][gh]  
  (often the case if algorithm is efficient)

.center[
<img src="img/ssd_keras.png" width="700px" /> 
]

---
count: false
## Start with existing (working) models

- look for a relevant architecture for your problem  
  (arXiv, blogs, websites)
- find an implementation on [GitHub][gh]  
  (often the case if algorithm is efficient)
- play with the examples and adjust to your inputs/outputs

--
- use [pretrained nets][kerasapp] for the  pre-processing of your data

--
- start tuning the model parameters..

[gh]: https://github.com/
[kerasapp]: https://keras.io/applications/

---
## Plot the training loss

```python
import matplotlib.pyplot as plt

history = model.fit(X_train, y_train, validation_split=0.3)  

# Visualizing the training                    
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epochs'); plt.ylabel('loss'); plt.legend()
```

.center[<img src="img/loss.png" width="500px">]

---

## Plot the training loss

And look for the training .green[sweet spot] (before .red[overfitting]).

.center[<img src="img/overfitting.png" width="550px">]

---
## Plot other metrics

```python
import matplotlib.pyplot as plt

model.compile(..., metrics=['acc'])  # computes other metrics, here accuracy

history = model.fit(X_train, y_train, validation_split=0.3)

# Visualizing the training                    
plt.plot(history.history['acc'], label='training')
plt.plot(history.history['val_acc'], label='validation')
plt.xlabel('epochs'); plt.ylabel('accuracy'); plt.legend()
```

.center[<img src="img/accuracy.png" width="450px">]

---
## Tensorboard

.center[<img src="img/tensorboard_edward.png" width="770px">]

.footnote[credit: [Edward Tensorboard tuto](http://edwardlib.org/tutorials/tensorboard)]

---
class: center, middle
name: optim
# Common optimizations

.medium["avoiding overfitting"]

---
## Data is key

Deep neural nets need .red[a lot of data] to achieve good performance.

.center[<img src="img/dl_perf.jpg", width="600px"/>]

---
count: false
## Data is key

Deep neural nets need .red[a lot of data] to achieve good performance.

Use .red[data augmentation].

.center[<img src="img/data_augment.png", width="600px"/>]

---
count: false
## Data is key

Deep neural nets need .red[a lot of data] to achieve good performance.

Use .red[data augmentation].

Choose a training set .red[representative] of your data.


--
If you cannot get enough labeled data, use simulations or turn to [transfer learning](https://arxiv.org/abs/1411.1792) techniques.

---
## Dropout

A % of random neurons are .green[switched off] during training  
it mimics different architectures being trained at each step 

.center[<img src="img/dropout.png" width="500 px" />]
.footnote[[Srivastava et al. (2014)](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)]

---
## Dropout

```python
...
from keras.layers import Dropout

dropout_rate = 0.25

model = Sequential()
model.add(Conv2D(2, (3, 3), input_shape=(9, 9, 1)))
*model.add(Dropout(dropout_rate))
model.add(Conv2D(4, (3, 3)))
*model.add(Dropout(dropout_rate))
...
```

- regularization technique extremely effective
- .green[prevents overfitting]

**Note:** dropout is .red[not used during evaluation], which accounts for a small gap between **`loss`** and **`val_loss`** during training.


.footnote[[Srivastava et al. (2014)](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)]

---
## Batch normalization

```python
...
from keras.layers import BatchNormalization
from keras.layers import Activation

model = Sequential()
model.add(Conv2D(..., activation=None))
*model.add(BatchNormalization())
model.add(Activation('relu'))
```

- technique that .green[adds robustness against bad initialization]
- forces activations layers to take on a unit gaussian distribution at the beginning of the training
- must be used .red[before] non-linearities

.footnote[[Ioffe & Szegedy (2015)](http://arxiv.org/abs/1502.03167)]

---
## other tricks

Here are some leads (random order) to explore if your model do not converge:
- data normalization
- weight initialization
- learning rate decay
- gradient clipping
- regularization

---
class: center
.hidden[a]
# Next ?

### ML developments are happening at a high pace <br/>.red[stay tuned] !

.hidden[a]
# References

.middle[### A curated list of inspirations for this presentation can be found [here][refs].]

[refs]: https://github.com/aboucaud/slides/blob/master/2018/hands-on-deep-learning/references.md

---
class: center, middle
<!-- background-image: url(img/thankyou.gif) -->
.center[
<img src="img/thankyou.gif" style="width: 700px;" />
]

---
exclude: true
class: center, middle

<img src="img/friendship_algorithm.PNG" />

.medium[but keep in mind that .red[not everything] is differentiable..]
