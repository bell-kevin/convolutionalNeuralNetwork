<a name="readme-top"></a>

# 

Convolutional Neural Network (CNN) from scratch in pure NumPy for MNIST digit classification. This is more complicated than a simple MLP because it implements:

  - A convolution layer
    
  - A max pooling layer
     
  - A final fully-connected layer

It uses ReLU activations in hidden layers and softmax + cross-entropy for the output.

Note: Implementing CNNs entirely from scratch in NumPy (especially the backward pass for convolution) is quite involved—and very slow for large datasets. This example is for learning purposes rather than performance. It uses only a subset of MNIST and a few epochs to keep runtime somewhat manageable.


How It Works

    Data Loading:
        We load MNIST via tensorflow.keras.datasets.
        For speed, we use a smaller subset of the training and test sets.
        Images are normalized to [0,1][0,1], and labels are one-hot encoded.

    CNN Architecture:
        Conv2D (8 filters, 3×33×3 kernel), ReLU
        MaxPool2D (2×2)
        Flatten
        Linear (fully connected) from 8×13×13 → 128, ReLU
        Linear (fully connected) from 128 → 10 (logits)
        Softmax for classification (done inside the loss function).

    Convolution + Backprop:
        We implement a manual convolution in Conv2D.forward.
        Backprop is done by correlating the upstream gradient with the saved input patches to update the filters and biases.
        This is the most computationally heavy part.

    Max Pooling + Backprop:
        Forward pass picks the maximum 2×2 region.
        Backward pass “routes” the gradient back to the max positions.

    Loss:
        We use softmax cross-entropy.
        The backward pass returns (\text{probs} - \text{labels}) / \text{batch_size}.

    Training:
        We do a mini-batch gradient descent loop.
        Each step: forward → loss → backward → update parameters.

    Evaluation:
        We do a simple forward pass on the test set and check the prediction accuracy.

    Warning: This code is purely educational. It will run much slower than frameworks like TensorFlow or PyTorch (which use optimized C/C++ kernels, GPU support, etc.). For real-world applications, using a deep learning framework is highly recommended.

--------------------------------------------------------------------------------------------------------------------------
== We're Using GitHub Under Protest ==

This project is currently hosted on GitHub.  This is not ideal; GitHub is a
proprietary, trade-secret system that is not Free and Open Souce Software
(FOSS).  We are deeply concerned about using a proprietary system like GitHub
to develop our FOSS project. I have a [website](https://bellKevin.me) where the
project contributors are actively discussing how we can move away from GitHub
in the long term.  We urge you to read about the [Give up GitHub](https://GiveUpGitHub.org) campaign 
from [the Software Freedom Conservancy](https://sfconservancy.org) to understand some of the reasons why GitHub is not 
a good place to host FOSS projects.

If you are a contributor who personally has already quit using GitHub, please
email me at **bellKevin@pm.me** for how to send us contributions without
using GitHub directly.

Any use of this project's code by GitHub Copilot, past or present, is done
without our permission.  We do not consent to GitHub's use of this project's
code in Copilot.

![Logo of the GiveUpGitHub campaign](https://sfconservancy.org/img/GiveUpGitHub.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
