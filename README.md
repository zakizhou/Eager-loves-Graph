Eager loves Graph
-------------------------------------------------------------------------

notes on writing code that is compatible with both TensorFlow eager mode and graph mode

# Introduction
We love tensorflow because of huge amount of models in its model zoo, numerous tutorials online, deployment support, great community from inside and outside Google and etc.

We hate tensorflow because we can't see the value of tensors but only see them with unknown shapes in graph mode
which frustrates us because of its static graph design, we can't debug tensorflow's program.

Recently, [TensorFlow Eager is released](https://github.com/tensorflow/tensorflow/releases/tag/v1.7.0), bringing the 
imperative execution of operation which solves the problem of graph mode, we call it **eager mode**, so here comes 
the question

**Can we write compatible code to enjoy not only the debugability of eager mode but also performance optimizations
 and production deployment of graph mode?**

**The answer is YES, and all you need to do is to add or delete a `#`, isn't that amazing?**

In this repo we mainly focus on writing code that is compatible with both eager and graph mode,
official tutorial mentions the compatibility in only one section [work with graphs](https://www.tensorflow.org/programmers_guide/eager#work_with_graphs) but I think it provides
few details so I decide to write this personal note on compatibility with eager and graph.

The notes mainly have two parts:
* Components

covers the compatibility of different components such as Variables, Model, Optimizer and etc with eager and graph mode
* Best Practice

introduce how to group codebase and what is the working procedure so that smallest change is required to switch between eager and graph

# Requirements

## Knowledge
* familiar with tensorflow graph execution, know what is `tf.data`, `tf.train.Optimizer`, `tf.Variable`, know how to
build a model in graph mode, know what is tensorflow eager and have read the [official tutorial](https://www.tensorflow.org/programmers_guide/eager) about eager.
* understand reference count and the garbage collection mechanism in Python.

## Hardware
It's better to have a gpu, but if you don't have, don't worry, you can still run nearly all the code in this note

## Software
version: tensorflow >= **1.9.0rc0**

Since tensorflow eager is evolving rapidly, so it is better to use the latest version 1.9.0rc0(as of 2018.6.22)
which contains newest features, all the code in this repo can be run in 1.9.0rc0 but only part can run on other 
versions.

You can easily create a virtual env with conda to run this code.

platform: ubuntu

code is tested in ubuntu but other platforms should work fine.



# Components

## Input Pipeline
In tensorflow version 1.3 or 1.4, a new kind of input pipeline is added to main apis to replace
the old queue-based one, that is `tf.data`, there all two core classes: `tf.data.Dataset` and `tf.data.Iterator`.\
Later when tensorflow eager is released, `tf.data.Dataset` is made to be compatible with eager, but not `tf.data.Iterator` because some kind
of `tf.data.Iterator` requires `tf.Session` to reinitialize its state, only one kind of `tf.data.Iterator` can be used in eager mode, that is the [one-shot](https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator) one

I won't cover too much about how to use `tf.data`, see [documents](https://www.tensorflow.org/programmers_guide/datasets) for details, rather I will focus
on what is incompatible with eager: `tf.data.Iterator`.

In graph mode four kinds of iterators can be used, and `tf.data.Iterator.get_next()` returns symbolic tensors that represent each elements in the iterator\
In eager mode, only one-shot iterator can be used, in the section *Best Practice*, I will mention how to deal with iterator.

```Python
import tensorflow as tf
tf.enable_eager_execution()
random_inputs = tf.random_normal((4, 2))
dataset = tf.data.Dataset.from_tensor_slices(random_inputs)
batched = dataset.batch(2)
```

Comment the line `tf.enable_eager_execution()` to see the compatibility of `tf.data.Dataset` in graph mode

Summary on Input Pipeline

* use `tf.data.TFRecordDataset`, `tf.data.Dataset.map`, `tf.data.Dataset.batch` and etc to build a dataset, this will
make you code compatible with both eager and graph mode


---------------------------------------------------
## Variable
Variables define the version of a model, most tensorflow users are familiar with `tf.Variable` and `tf.get_variable`
but sadly `tf.Variable` can't be used in eager mode, instead, another kind of variable `ResourceVariable` can be used 
in both eager and graph mode, see this [link](https://stackoverflow.com/questions/40817665/whats-the-difference-between-variable-and-resourcevariable-in-tensorflow) on what is a `ResourceVariable`

* use `ResourceVariable`(defined in `tensorflow/python/ops/resource_variable_ops.py`, `ResourceVariable` is an alias of
`tfe.Variable`) instead of `tf.Variable`(defined in `tensorflow/python/ops/variables.py`)

Let's check that `ResourceVariable` is actually `tfe.Variable`
```Python
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import tensorflow.contrib.eager as tfe
print(ResourceVariable is tfe.Variable)  #  True
```

And also check that `ResourceVariable` can be used in both eager mode and graph mode, but `tf.Variable` can only live
in graph mode

in eager mode:
```Python
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
tf.enable_eager_execution()
# fine to create ResourceVariable in eager mode
rv = ResourceVariable([5.20])

# RuntimeError: tf.Variable not supported when eager execution is enabled
v = tf.Variable([13.14])
```

in graph mode:
```Python
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
# tf.enable_eager_execution()
# fine to create ResourceVariable in graph mode
rv = ResourceVariable([5.20])

# fine to create tf.Variable in graph mode
v = tf.Variable([13.14])
```

* Apart from calling the constructor of `ResourceVariable`, the other way to create `ResourceVariable` is to call 
`tf.get_variable(..., use_resource=True)`, the latter one is better
because it enables you to do [variable sharing](https://www.tensorflow.org/programmers_guide/variables#sharing_variables)

---------------------------------------------------
## Layer: Explicit Storage of Variables
Before we keep on, let's see how tensorflow manages the life cycles of its tensors and variables and **why we need
explicit storage of variables**.

In graph mode, when the last reference of a tensor disappears, the tensor still exists in graph.
```Python
import tensorflow as tf
c = tf.constant([5.20], name="c1")
# last reference of tensor c1:0 disappear because we assign the name "c" to another tensor
c = tf.constant([13.14], name="c2")

graph = tf.get_default_graph()
# but c1:0 still exist in graph, let's get it
c1 = graph.get_tensor_by_name("c1:0")

# print all nodes in graph, two tensors in this graph
print(graph.as_graph_def())
```

In eager mode, tensors and variables are **Python objects**, when the last reference of them disappears, they are gone.
```Python
import tensorflow as tf
tf.enable_eager_execution()
c = tf.constant([5.20], name="c1")
# last reference of previous tensor disappears because we assign the name "c" to another tensor, at this time, it is
# garbage collected by Python interpreter and never come back
c = tf.constant([13.14], name="c2")
```

Now let's see a basic routine in graph mode:
```Python
import tensorflow as tf

inputs = do_whatever_to_get_inputs()
def inference(inputs):
    # create variables, note that these variables are implicitly added to a collection called 
    # tf.GraphKeys.GLOBAL_VARIABLES
    variable_to_optimize = tf.Variable([5.20])
    target = do_whatever_with_variables_and_inputs(inputs, variable_to_optimize)
    return target

# after this function call, the variables created in this function still exist in graph (because now we are in graph 
# mode) and can be accessed with tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
target = inference(inputs)

# calculate the gradients of targets with respect to all variables and apply the gradients of variables to do 
# optimization, this is the case in optimizer.minimize(target)
gradients = tf.gradients(target, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
```

So what happens in eager mode? After the `inference(inputs)` is called, the variables created in this function are gone
because they are Python's local variables inside a function now and the last references to them are gone, so we can't update them any more.

How can we keep them? **We can make a explicit storage of variables by assigning them as the attributes of a class!**

Let's see an example in eager mode
```Python
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
tf.enable_eager_execution()


class BadFullyConnectedLayer(object):
    def __init__(self):
        # we explicitly assign variables as attributes of class so the reference to them will not disappear
        self.weight = ResourceVariable([[1., 2.], [3., 4.]])
        self.bias = ResourceVariable([5., 6.])

    def __call__(self, inputs):
        return tf.nn.xw_plus_b(inputs, self.weight, self.bias)


fc = BadFullyConnectedLayer()
with tf.GradientTape() as tape:
    o = tf.reduce_mean(fc(tf.random_normal((2, 2))))
# tf.GradientTape.gradient is used like tf.gradients in graph mode, we will mention this in the next two section Optimizer
gradients = tape.gradient(o, [fc.weight, fc.bias])
```
if you comment the line `tf.enable_eager_execution()` to return to graph mode, the code still runs smoothly, that's really
what we want, isn't it?

But the class `BadFullyConnectedLayer` has pool designs such as lacking methods to collect all variables, name scope
management, input tensor type check and etc.

**So here comes one of the two important classes for model implementation: `tf.keras.layers.Layer`, it is the superclass of all high level classes such
as `tf.keras.layers.Dense` , `tf.keras.layers.Conv2D`, `tf.nn.rnn_cell.LSTMCell` and `tf.keras.layers.Embedding` that have explicit storage of
variables**, it has several utils functions to enable its subclasses to use.

The document of this class says

>
>  We recommend that descendants of `Layer` implement the following methods:
>  * `__init__()`: Save configuration in member variables
>  * `build()`: Called once from `__call__`, when we know the shapes of inputs
>    and `dtype`. Should have the calls to `add_weight()`, and then
>    call the super's `build()` (which sets `self.built = True`, which is
>    nice in case the user wants to call `build()` manually before the
>    first `__call__`).
>  * `call()`: Called in `__call__` after making sure `build()` has been called
>    once. Should actually perform the logic of applying the layer to the
>    input tensors (which should be passed in as the first argument).
>    
> -- [document](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/base_layer.py#L80-L89) of `tf.keras.layer.Layer`


let's read the source code of `tf.keras.layers.Dense` and `tf.keras.layers.Embedding` as examples to see how to use this class

`tf.keras.layer.Dense`: [full code here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/core.py#L816-L973)
```Python
class Dense(Layer):
  # lots of unimportant code are omitted here with ...
  def __init__(self, units, ...):
    # call superclass `tf.keras.layers.Layer`'s constructor
    super(Dense, self).__init__(...)
    self.units = int(units)
    ...
    
  def build(self, input_shape):
    # call self.add_variable to attach a `ResourceVariable` to `tf.keras.layers.Layer`
    self.kernel = self.add_variable('kernel',
                                    shape=[input_shape[-1].value, self.units], ...)
    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[self.units,], ...)
    else:
      self.bias = None
    # must set `self.built` to `True` to signal that `self.build` has been called
    self.built = True

  def call(self, inputs):
    # `Dense` acts as fully connected layer, so do a matrix multiply and add bias here
    outputs = gen_math_ops.mat_mul(inputs, self.kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    ...
    return outputs
```

`tf.keras.layers.Embedding`: [full code here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/embeddings.py#L32-L182)
```Python
class Embedding(Layer):
  def __init__(self,
               input_dim,
               output_dim,
               ...):
    super(Embedding, self).__init__(dtype=dtype, **kwargs)

    self.input_dim = input_dim
    self.output_dim = output_dim

  def build(self, input_shape):
    self.embeddings = self.add_weight(
        shape=(self.input_dim, self.output_dim),
        ...)
    self.built = True
    
  def call(self, inputs):
    out = embedding_ops.embedding_lookup(self.embeddings, inputs)
    return out
```

Summary on `tf.keras.layers.Layer`:
* Why do we need `tf.keras.layers.Layer`:
\
  Because in eager mode, variables created during any function call are **Python's local variables which will 
  be garbage collected**, so we can't do any update of variables since they disappear, so in order to make code compatible
  with eager and graph, we need a container class to explicitly store these variables, that's `tf.keras.layers.Layer`.

* When should we  use `tf.keras.layers.Layer`: 
\
**When we want to create variables by ourself, you must write a class that inherits from it, note that create layers like
`tf.keras.layers.Dense` is not creating variables**.
\
If we don't need to create variables, we can still inherit from `tf.keras.layers.Layer`, but a there's another choice,
see next subsection.

* How do we use `tf.keras.layers.Layer`:
\
Like `tf.keras.layers.Dense`, inherits from `tf.keras.layers.Layer`, and override three methods:
(1) `__init__` \
(2) `build`: call `self.add_variable`(or `self.add_weight`) to attach `ResourceVariable` to the class in `build`, this 
 enables variables collection with `tf.keras.layers.Layer.variables`\
(3) `call`: implement you computation logic with variables created in `build` and inputs 
 

##  Model: Tree Structure of Model(s) and Layer(s)
When build deep neural network models, we seldom explicitly create variables by ourselves, but rather use already defined layers
such as `tf.keras.layers.Dense` , `tf.keras.layers.Conv2D` and `tf.keras.layers.Embedding`, as a result, we want a container of `tf.keras.layers.Layer`,
so after `tf.keras.layers.Layer`, now comes to the other important class to build models, that is `tf.keras.Model`.

A `Model` is composed of other defined `Model`(s) and `Layer`(s) which will finally forms as a tree structure.
the [document of this class](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py#L67-L84) says
>   2 - By subclassing the `Model` class: in that case, you should define your
  layers in `__init__` and you should implement the model's forward pass
  in `call`.
>  ```python
>  import tensorflow as tf
>  class MyModel(tf.keras.Model):
>    def __init__(self):
>      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
>      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
>    def call(self, inputs):
>      x = self.dense1(inputs)
>      return self.dense2(x)
>```

According to the doc, let's implement a five layers of deep neural network as an example, **note that this code is an example to explain the tree structure of
`tf.keras.Model` not for actual use, you can easily build a five layers DNN with `tf.keras.Sequential`**

```Python
import tensorflow as tf
# execute the code in graph mode and then try to uncomment the 
# next line to execute the code in eager mode to see compatibility
# tf.enable_eager_execution()

class OneLayerNet(tf.keras.Model):
    def __init__(self):
        super(OneLayerNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")

    def call(self, inputs):
        return self.dense1(inputs)


class TwoLayersNet(tf.keras.Model):
    def __init__(self):
        super(TwoLayersNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


class FiveLayersNet(tf.keras.Model):
    def __init__(self):
        super(FiveLayersNet, self).__init__()
        self.two_layer_net1 = TwoLayersNet()
        self.two_layer_net2 = TwoLayersNet()
        self.one_layer_net = OneLayerNet()

    def call(self, inputs):
        x = self.two_layer_net1(inputs)
        x = self.two_layer_net2(x)
        x = self.one_layer_net(x)
        return x


five_layers_net = FiveLayersNet()
with tf.GradientTape() as tape:
    x = tf.reduce_mean(five_layers_net(tf.random_normal((32, 128))))

gradients = tape.gradient(x, five_layers_net.variables)

```
The tree structure of this model is
```Python
five_layers_net           --->      tf.keras.Model
|__ two_layer_net1        --->      |__ tf.keras.Model
    |__ dense1            --->          |__ tf.keras.Layer
    |__ dense2            --->          |__ tf.keras.Layer
|__ two_layer_net2        --->      |__ tf.keras.Model
    |__ dense1            --->          |__ tf.keras.Layer
    |__ dense2            --->          |__ tf.keras.Layer
|__ one_layer_net         --->      |__ tf.keras.Model
    |__ dense1            --->          |__ tf.keras.Layer

```
Another example that explains the tree structure of `tf.keras.Model` is the official implementation of [ResNet50](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/resnet50/resnet50.py)

Summary on `tf.keras.Model`:
* Why and when do we need `tf.keras.Model`:
\
When we want to build model only composed of already defined layers or models(including your custom one, see previous sub-section `Layer`),

* How do we use `tf.keras.Model`:
\
Write a class (your model) that inherits from `tf.keras.Model`, then override two methods:
(1) `__init__`: attach the layers(of type `tf.keras.layers.Layer`) and models(of type `tf.keras.Model`) 
you want to use to the class in `__init__`, this enables you to collect all variables of model with `Model.variables`.
(2) `call`: implement your compuation logic with the attached layers, models and your inputs.

* Final model should compose of other sub-models and layers with these sub-models contains other sub-sub-models and other layers...
which forms a tree structure

* Do not call `self.add_weight`(`self.add_variable`) or try other methods to  attach a variable(both `tf.Variable` and `ResourceVariable`) to `tf.keras.Model`
as `self.add_weight` is forbidden by `tf.keras.Model` and `tf.keras.Model.variables` will not collect variables that are
manually attached to it.

**`tf.keras.layers.Layer` and `tf.keras.Model` are the two most important classes in eager mode that are also compatible with
graph mode, readers should go through more examples to see how to use them**

---------------------------------------------------
## Optimizer:
Before we go to see how to make an Optimizer compatible with both eager and graph, let's analyze the mechanism of
`tf.train.Optimizer.minimize`, it is actually a combination of two steps, `Optimizer.compute_gradients` for calculating gradients
of variables and 
`Optimizer.apply_gradients` for updating variables with gradients, see this [comment](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py#L356-L359) for detail.

While `Optimizer.apply_gradients` is naturally compatible with both eager and graph mode(because it is just a update of variables or more preciously, a `tf.assign` operation), 
in graph mode, `Optimizer.compute_gradients` calls `tf.gradients` to compute gradients which relies on the structure of graph which does not
exists in eager mode, as a result, `Optimizer.compute_gradients` is not compatible with eager mode, so now come to the 
`tf.GradientTape` which can be used in both  eager and graph mode.

Check this [link](https://discuss.pytorch.org/t/what-is-a-tape-based-autograd-system/1400) on what is a tape-based system
for autograd

Let's see an example of minimizing $y = x^2$
```Python
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
tf.enable_eager_execution()

# model definition starts from here and can be shared with both eager and graph mode
x = ResourceVariable([5.20])
optimizer = tf.train.AdamOptimizer(0.1)


def train(x, optimizer):
    with tf.GradientTape() as tape:
        y = tf.square(x)

    gradients = tape.gradient(y, [x])
    train_op = optimizer.apply_gradients(zip(gradients, [x]))
    return train_op


# train 100 steps, need to check whether in eager mode or graph mode
for i in range(100):
    if tf.executing_eagerly():
        train(x, optimizer)
    else:
        if i == 0:
            # in graph mode, only need to build graph once
            train_op = train(x, optimizer)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(train_op)
        sess.run(train_op)

# finally print the value of optimized x
if tf.executing_eagerly():
    print(x)
else:
    x_value = sess.run(x)
    print(x_value)
    sess.close()
```
Try to comment `tf.enable_eager_execution()` to do the optimization in graph mode, the code before the for loop is shared
with both eager and graph mode.

Summary on `Optimizer`:
* use `tf.GradientTape` to compute gradients which is compatible with both eager and graph mode instead of `Optimizer.compute_gradients`

* still call `Optimizer.apply_gradients` to do variables updating, in eager mode, this update op immediately gets done while
in graph mode this update op will be the train op to be passed to `tf.Session.run`

TODO recently I read the source code of `Optimizer.compute_gradients` and find that when the `loss` is callable, this method
internally calls `tf.GradientTape.gradient` to compute gradients, find the situation where it is better than `tf.GradientTape` in future

---------------------------------------------------
## Saver and Checkpoint
Most tensorflow users should be familiar with `tf.train.Saver` when it comes to  model save and restore. In graph mode, 
if you pass nothing to the constructor of a `tf.train.Saver`, then tensorflow will implicitly collect all 
variables in graph(including model variables and optimizer variables) and tell this saver to save all variables, see the [source code of Saver](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/saver.py#L1311-L1313)
for details, this can't work in eager mode because these is no graph, no collections, tensorflow can't implicitly collect variables.
In order to let `tf.train.Saver` know which variables to save, we need to explicitly pass variables to the
constructor of `tf.train.Saver`
```Python
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
tf.enable_eager_execution()
r = ResourceVariable([5.20])

# this is ok since we pass variable(s) to the constructor of Saver
saver_with_explicit_variables = tf.train.Saver(r)
# RuntimeError: When eager execution is enabled, `var_list` must specify a list or dict of variables to save
saver_without_explicit_variables = tf.train.Saver()
```

Try to comment `tf.enable_eager_execution()` to see the two savers in graph mode, actually they will act
identically as `saver_without_explicit_variables` will find the `ResourceVariable` `r` in graph.

Note that the saved checkpoint file is also compatible with eager and graph, so you can experiment, debug and experiment several
epoches in eager mode and save the checkpoint, and go to graph mode, restore variables from saved checkpoint and keep on training

Let's see an example, first we save checkpoint in eager mode
```Python
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

tf.enable_eager_execution()
# initialize a variable with initial value 5.20, then save it to "checkpoint/save.ckpt"
x = ResourceVariable([5.20])
saver = tf.train.Saver(var_list=[x])
# make sure the directory checkpoint exists, or it will raise an error
# first parameter sess should be None since now we are in eager mode
saver.save(sess=None, save_path="checkpoint/save.ckpt")
```
Then restore it in graph mode
```Python
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
x = ResourceVariable([13.14])
saver = tf.train.Saver(var_list=[x])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path="checkpoint/save.ckpt")
    # after restore, the value became the value we saved from eager mode instead of initial value [13.14]
    print(sess.run(x)) # [5.2]
```

In recent release of tensorflow, another class called `tfe.Checkpoint` appears in APIs and after testing `tfe.Checkpoint`, I found that
it has same functionality with `tf.train.Saver`, readers can try this class by self, documents are [here](https://www.tensorflow.org/api_docs/python/tf/contrib/eager/Checkpoint)

But still need to mention that, the document says that `tf.train.Saver`'s checkpoint is name-based 
while `tfe.Checkpoint` is object-based, if someone has mixed use of them like save checkpoint with
`tfe.Checkpoint` and restore with `tf.train.Saver`, there will be a WRANING like this
```Python
WARNING:tensorflow:Restoring an object-based checkpoint using a name-based saver. This may be somewhat fragile, and will re-build the Saver. Instead, consider loading object-based checkpoints using tf.train.Checkpoint().
```

So just be familiar with one of the two and insist on it.

Summary on Saver and Checkpoint

* No matter in `tf.train.Saver` or `tfe.Checkpoint`, one should always **explicitly pass variables to the constructor of them** 
if you want your code to be compatible with eager and graph

* The checkpoint saved by `tf.train.Saver` in eager mode can also be used in graph mode, so one can debug the first several
epoches in eager mode, save it, go to graph mode, restore it, and keep on training.
So is `tfe.Checkpoint`.

* Better not to mixedly use `tf.train.Saver` and `tfe.Checkpoint`, since they are of different working mechanism

* There are two important kinds of variables in the training of tensorflow model, they are model variables and optimizer variables,
when saving, optimizer variables may be forgotten since they are unnecessary in inference, please do not forget them if
you want to keep on training

---------------------------------------------------
## Device
In graph mode, device placement cen be done with `with tf.device(/cpu:0):` or `with tf.device(/gpu:1):` block, 
operations under this block will automatically run on the specified device, when eager is released, move a tensor to
gpu can be simply done with `some_tensor.gpu()`, but this does not work in graph mode.

* use `with tf.device("device_name:device_id")` block to specify device placement

**TODO add more details**

---------------------------------------------------
## Summaries and TensorBoard
use `tf.contrib.summary`

**TODO add more details**

---------------------------------------------------
## Control Flow
It must be admit that the naive Python control flows like `for`,`if` and `while` **can't be serialized in graph** so although
they can be used in eager mode, they are not compatible with graph, let's see an example using Python control flows
```Python
import tensorflow as tf
tf.enable_eager_execution()


class SimpleUnrollRNNInEager(tf.keras.Model):
    """
    An unroll simple rnn that can only run in eager mode because of fixed time_steps
    """
    def __init__(self, hidden_units):
        super(SimpleUnrollRNNInEager, self).__init__()
        self.hidden_units = hidden_units
        self.kernel = tf.keras.layers.Dense(self.hidden_units, activation="relu")
        self.recurrent_kernel = tf.keras.layers.Dense(self.hidden_units, activation="relu")

    def call(self, inputs):
        """

        :param inputs: Tensor[batch_size, time_steps, feature_dims]
        :return: Tensor[batch_size, hidden_units]
        """
        # note that in eager mode, the shape of tensor inputs is known, so can we can get the exact value of time_steps
        # but in graph mode, often this dim is unknown, so this class can only used in eager mode
        batch_size, time_steps, dim = inputs.shape.as_list()
        state = tf.zeros((batch_size, self.hidden_units))
        for i in range(time_steps):
            state = self.kernel(inputs[:, i, :]) + self.recurrent_kernel(state)
        return state


simple_rnn = SimpleUnrollRNNInEager(5)
inputs = tf.random_normal((2, 3, 4))
# inputs = tf.placeholder(tf.float32, shape=[2, None, 4])
state = simple_rnn(inputs)
```
if you comment the `tf.enable_eager_execution()`, `inputs = tf.random_normal((2, 3, 4))` and also uncomment
`# inputs = tf.placeholder(tf.float32, shape=[2, None, 4])` to see whether it works in graph mode, you will
see errors:
```Python
TypeError: 'NoneType' object cannot be interpreted as an integer
```

So if you want to use control flow to build your model, the only way to write compatible code with eager and graph is to use the ugly tensorflow control
flow ops `tf.cond` and `tf.while_loop` because they are naive tensorflow operations. I won't talk this in details, here
is the [document](https://www.tensorflow.org/api_guides/python/control_flow_ops) of control flow ops in tensorflow

Summary on control flow
* Try to avoid to use control flow in your model, there are several convenient operations that can help you like `tf.where`, `tf.boolean_mask`
`tf.maximum`, `tf.nn.dynamic_rnn` and etc, implement your model with these operations.

* If you really need control flows to build your model, you have no choices, to make code compatible with eager and graph,
you have to use `tf.cond` and `tf.while_loop` as replacement of Python `if` and `for/while`


# Best Practice
Till now, I have covered the compatibility of different components in tensorflow, now it's time to introduce how to
group codebase, we aim at only one purpose here:

**Most code should be able to be used in both eager and graph mode, we may still need to write small amount of code for eager and graph mode respectively**

To be clear, I directly show the structures of codebase and then explain them in detail

## Structure of Codebase

### Model
```Python
# model.py
import tensorflow as tf
class SubModel1(tf.keras.Model):
    def __init__(self):
        super(SubModel1, self).__init__()
    
    def call(self, inputs):
        ...
        return xxx

class SubModel2(tf.keras.Model):
    def __init__(self):
        super(SubModel2, self).__init__()
    
    def call(self, inputs):
        ...
        return xxx
        
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.sub_model_1 = SubModel1(...)
        self.sub_model_2 = SubModel2(...)
    def call(self, inputs):
        x = self.sub_model_1(...)
        x = self.sub_model_2(...)
        return xxx
```
define all your models in a single file such as `model.py` which contains sub-models and the biggest model, or create package that contains many files that
seperately define your sub-models and finally ensemble them into the biggest model

### Data
```Python
# data.py
import tensorflow as tf
def build_inputs(...):
    dataset = tf.data.XXXRecordDataset(...)
    preprocessed = dataset.map(preprocess_fn)
    ...
    return your_dataset_to_model
```
build your input pipeline also in a single file or package

### Train
```Python
# train.py
import tensorflow as tf

def train(model, optimizer, inputs, labels):
    """
    Args:
    :param model: tf.keras.Model
    :param optimizer: tf.train.Optimizer
    :param inputs: tf.Tensor
    :param labels: tf.Tensor
    :return: Tuple or Dict
    """
    with tf.GradientTape() as tape:
        predicts = model(inputs)
        loss = loss_fn(predicts, labels)
    gradients = tape.gradient(loss, model.variables)
    train_op = optimizer.apply_gradients(zip(gradients, model.variables))
    return loss, train_op # and whatever you want to return
```
This is a function that is compatible with both eager and graph mode.
* In eager mode, this function directly do a batch stochastic gradient descent optimization with `model` 
* In graph mode, this function build the graph and generate the tensor of loss and the op for train 

### Entrance
```Python
# main.py
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from model import Model
from data import build_inputs

# switch between eager mode and graph mode
tf.enable_eager_execution()

model = Model(...)
dataset = build_inputs(...)
optimizer = tf.train.XXXOptimizer()
for i in range(max_epoches):
    if tf.executing_eagerly():
        for batch_id, (inputs, labels) in enumerate(tfe.Iterator(dataset)):
            loss_value, _ = train(model, optimizer, inputs, labels)
    else:
        if i == 0:
            iterator = dataset.make_initializable_iterator()
            inputs, labels = iterator.get_next()
            loss, train_op = train(model, optimizer, inputs, labels)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        while True:
            try:
                loss_value, _ = sess.run([loss, train_op])
            except tf.errors.OutOfRangeError:
                print("epoch done")
                sess.run(iterator.initializer)
                break

    print("epoch summary here")
```
This piece of code works as the entrance of training
* In eager mode, for each epoch, get a batch of inputs and labels from the iterator of dataset, and call `train` to do
optimization
* In graph mode, for the first epoch(`if i == 0`), we build the computation graph, get the tensor of loss and the op for train and then
 comes to a `while` loop inside which we try to iterate over all batches, if a `OutOfRangeError` is signaled which means one epoch is done, we will 
 reinitialize the iterator and go to next epoch

## Working Procedure

### Eager(Debug) mode
Now it's time to solve the debugability of tensorflow program, see the line `tf.enable_eager_execution()` in previous
subsection **Entrance**, when enable eager mode, we enter the debug mode.

Set break points to wherever you like in `model.py`, `data.py`, `train.py` and `main.py`(I even set break points to the source code
of tensorflow while reading it), start debug mode(use any debug tools you like,
personally, I love Pycharm), then you can check every inputs and labels of model, see immediate execution of operations and the intermediate tensors to
see if there is NANs or INFs, inspect whether you have exploding gradients or whatever you need to check.

### Graph(Train) mode
You only need to comment one line `tf.enable_eager_execution()` to switch from debug mode to real train mode.


# Summary
Use components that are compatible with both  eager and graph to build model such as `tf.data.Dataset`, `tf.keras.layers.Layer`
, `tf.keras.Model`, `tf.GradientTape`, `Optimizer.apply_gradients` and etc.

If you obey this rule, you can enjoy both the debugability of eager mode and optimization of graph mode by just adding or deleting
a `#` to `tf.enable_eager_execution()`


# Disclaim
This notes are written by myself according to my own experience and there must be some mistakes in it, 
I apologize for misleading the readers and contributions to make this notes better are always welcome.
