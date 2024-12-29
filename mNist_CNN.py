import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------------------
# 1. Load and prepare MNIST
# ----------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Subset to speed things up (e.g., take only 10k train, 2k test)
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test  = x_test[:2000]
y_test  = y_test[:2000]

# Normalize to [0,1]
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)

# Shapes:
# x_train: (10000, 28, 28)
# y_train: (10000, 10)
# x_test:  (2000,  28, 28)
# y_test:  (2000,  10)

# ----------------------------------------------------------------
# 2. Hyperparameters
# ----------------------------------------------------------------
epochs = 3
batch_size = 64
learning_rate = 0.001

# Convolution layer config
num_filters = 8   # number of conv filters
kernel_size = 3   # 3x3 kernel

# After conv, we'll do a 2x2 max pool,
# then flatten, then a fully-connected layer of size 128 -> 10.

# ----------------------------------------------------------------
# 3. Define helper functions
# ----------------------------------------------------------------
def one_hot_to_label(y_onehot):
    return np.argmax(y_onehot, axis=1)

def accuracy(predictions, labels):
    return np.mean(predictions == labels)

# ----------------------------------------------------------------
# 4. CNN Layers (Forward + Backward)
#    4.1 Convolution Layer
# ----------------------------------------------------------------
class Conv2D:
    """
    Simple 2D convolution (no padding, stride=1).
    W shape: (num_filters, 1, kernel_size, kernel_size)
    b shape: (num_filters, 1)
    """
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        # We assume a single channel (grayscale) input for MNIST.
        # We'll initialize the filters (num_filters, 1, k, k).
        limit = 1.0 / (kernel_size * kernel_size)
        self.W = np.random.uniform(-limit, limit, 
                                   (num_filters, 1, kernel_size, kernel_size)).astype(np.float32)
        self.b = np.zeros((num_filters, 1), dtype=np.float32)
        
    def forward(self, X):
        """
        X shape: (batch_size, 1, H, W)
        Returns: feature map shape: (batch_size, num_filters, out_h, out_w)
        out_h = H - kernel_size + 1
        out_w = W - kernel_size + 1
        """
        self.X = X  # save for backward
        batch_size, _, H, W = X.shape
        k = self.kernel_size
        out_h = H - k + 1
        out_w = W - k + 1
        
        # Allocate output
        out = np.zeros((batch_size, self.num_filters, out_h, out_w), dtype=np.float32)
        
        # Convolution
        for i in range(out_h):
            for j in range(out_w):
                # Patch of shape (batch_size, 1, kernel_size, kernel_size)
                patch = X[:, :, i:i+k, j:j+k]
                # (batch_size, num_filters) after summation
                # We multiply patch with filters and sum across the spatial dimensions
                # patch: (batch_size, 1, k, k)
                # W:     (num_filters, 1, k, k)
                # we can broadcast multiply:
                # for each filter f, multiply patch by W[f], sum => single number
                # then add bias
                # We'll do it more manually for clarity.
                for f in range(self.num_filters):
                    # element-wise multiply patch and W[f], then sum
                    out[:, f, i, j] = np.sum(patch * self.W[f, :, :, :], axis=(1,2,3))
                    
        # Add bias
        # out shape: (batch_size, num_filters, out_h, out_w)
        # b shape: (num_filters, 1)
        out += self.b.reshape(1, self.num_filters, 1, 1)
        return out
    
    def backward(self, d_out, learning_rate):
        """
        d_out shape: (batch_size, num_filters, out_h, out_w)
        Need to compute gradients w.r.t. self.W, self.b, and X.
        We'll update self.W, self.b in place.
        Returns dX
        """
        X = self.X
        batch_size, _, H, W = X.shape
        k = self.kernel_size
        _, _, out_h, out_w = d_out.shape
        
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(X)
        
        # Gradient w.r.t. b is just the sum of d_out over the batch/spatial dims
        # shape of d_out: (batch_size, num_filters, out_h, out_w)
        db = np.sum(d_out, axis=(0, 2, 3)).reshape(self.num_filters, 1)
        
        # Compute gradient w.r.t. W, X
        for i in range(out_h):
            for j in range(out_w):
                patch = X[:, :, i:i+k, j:j+k]  # shape: (batch_size, 1, k, k)
                for f in range(self.num_filters):
                    # d_out[:, f, i, j] shape: (batch_size,)
                    # We want to multiply patch by d_out value and sum across batch
                    # dW[f] += sum_over_batch( patch * d_out_value )
                    dW[f] += np.sum(
                        patch * d_out[:, f:f+1, i:i+1, j:j+1],
                        axis=0
                    )
                    
                    # Now for dX, we add contribution from filter f
                    dX[:, :, i:i+k, j:j+k] += self.W[f] * d_out[:, f:f+1, i:i+1, j:j+1]
        
        # Update weights
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dX


# ----------------------------------------------------------------
#    4.2 MaxPool Layer (2x2)
# ----------------------------------------------------------------
class MaxPool2D:
    """
    2x2 max pooling with stride=2 (no overlap).
    """
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, X):
        """
        X shape: (batch_size, channels, H, W)
        We'll reduce H, W by factor of 2.
        Output shape: (batch_size, channels, H//2, W//2)
        """
        self.X = X
        batch_size, channels, H, W = X.shape
        p = self.pool_size
        out_h = H // p
        out_w = W // p
        
        out = np.zeros((batch_size, channels, out_h, out_w), dtype=X.dtype)
        
        for i in range(out_h):
            for j in range(out_w):
                # region = X[..., i*p:(i+1)*p, j*p:(j+1)*p]
                region = X[:, :, i*p:(i+1)*p, j*p:(j+1)*p]
                out[:, :, i, j] = np.max(region, axis=(2,3))
        
        return out

    def backward(self, d_out):
        """
        d_out shape: (batch_size, channels, out_h, out_w)
        We need to upsample the gradient to the original X shape,
        distributing the gradient to the max locations.
        """
        X = self.X
        batch_size, channels, H, W = X.shape
        p = self.pool_size
        out_h = H // p
        out_w = W // p
        
        dX = np.zeros_like(X)
        
        for i in range(out_h):
            for j in range(out_w):
                region = X[:, :, i*p:(i+1)*p, j*p:(j+1)*p]
                max_vals = np.max(region, axis=(2, 3), keepdims=True)
                
                # Create a mask of where the maximums are
                mask = (region == max_vals)
                
                dX[:, :, i*p:(i+1)*p, j*p:(j+1)*p] += \
                    mask * d_out[:, :, i:i+1, j:j+1]
        
        return dX


# ----------------------------------------------------------------
#    4.3 Flatten Layer
# ----------------------------------------------------------------
class Flatten:
    """
    Flattens (N, C, H, W) into (N, C*H*W).
    """
    def forward(self, X):
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

# ----------------------------------------------------------------
#    4.4 Fully-Connected (Linear) Layer
# ----------------------------------------------------------------
class Linear:
    """
    Fully connected layer: out = X * W + b
    W shape: (in_features, out_features)
    b shape: (out_features,)
    """
    def __init__(self, in_features, out_features):
        limit = 1.0 / np.sqrt(in_features)
        self.W = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros((out_features,), dtype=np.float32)
    
    def forward(self, X):
        self.X = X  # save for backward
        return X.dot(self.W) + self.b
    
    def backward(self, d_out, learning_rate):
        # d_out shape: (batch_size, out_features)
        # X shape: (batch_size, in_features)
        dW = self.X.T.dot(d_out)  # (in_features, out_features)
        db = np.sum(d_out, axis=0)  # (out_features,)
        dX = d_out.dot(self.W.T)
        
        # update params
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dX

# ----------------------------------------------------------------
#    4.5 ReLU Activation
# ----------------------------------------------------------------
class ReLU:
    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask
    
    def backward(self, d_out):
        return d_out * self.mask

# ----------------------------------------------------------------
#    4.6 Softmax + Cross-Entropy
# ----------------------------------------------------------------
def softmax_cross_entropy_loss(logits, y_true):
    """
    logits: (batch_size, 10)
    y_true: (batch_size, 10) (one-hot)
    Returns loss (scalar) and d_logits
    """
    # Numerically stable softmax
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    # Cross-entropy
    eps = 1e-15
    batch_size = logits.shape[0]
    loss = -np.sum(y_true * np.log(probs + eps)) / batch_size
    
    # Gradient w.r.t. logits
    d_logits = (probs - y_true) / batch_size
    return loss, d_logits

# ----------------------------------------------------------------
# 5. Build the CNN model (forward/backward pipeline)
#    We'll create a small 'Sequential' style model.
# ----------------------------------------------------------------
class SimpleCNN:
    def __init__(self):
        self.conv = Conv2D(num_filters, kernel_size)
        self.relu1 = ReLU()
        self.pool = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        # After pool, image is (batch, 8 filters, 13, 13) -> 8*13*13=1352
        self.fc1 = Linear(8*13*13, 128)
        self.relu2 = ReLU()
        self.fc2 = Linear(128, 10)
    
    def forward(self, X):
        """
        X shape: (batch_size, 28, 28)
        But our Conv2D expects shape: (batch_size, 1, 28, 28)
        """
        out = X.reshape(X.shape[0], 1, 28, 28)
        out = self.conv.forward(out)    # (batch, 8, 26, 26)
        out = self.relu1.forward(out)   # ReLU
        out = self.pool.forward(out)    # (batch, 8, 13, 13)
        out = self.flatten.forward(out) # (batch, 8*13*13)
        out = self.fc1.forward(out)     # (batch, 128)
        out = self.relu2.forward(out)   # ReLU
        out = self.fc2.forward(out)     # (batch, 10) - logits
        return out
    
    def backward(self, d_out):
        # reverse order
        d_out = self.fc2.backward(d_out, learning_rate)
        d_out = self.relu2.backward(d_out)
        d_out = self.fc1.backward(d_out, learning_rate)
        d_out = self.flatten.backward(d_out)
        d_out = self.pool.backward(d_out)
        d_out = self.relu1.backward(d_out)
        d_out = self.conv.backward(d_out, learning_rate)
        return d_out

# ----------------------------------------------------------------
# 6. Training Loop
# ----------------------------------------------------------------
model = SimpleCNN()

num_samples = x_train.shape[0]
num_batches = num_samples // batch_size

for epoch in range(epochs):
    # Shuffle indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    for i in range(num_batches):
        start = i * batch_size
        end   = start + batch_size
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]
        
        # Forward
        logits = model.forward(x_batch)  # shape: (batch_size, 10)
        
        # Loss + grad
        loss, d_logits = softmax_cross_entropy_loss(logits, y_batch)
        
        # Backward
        model.backward(d_logits)
        
        # (Optional) Print progress
        if (i+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{num_batches}, Loss: {loss:.4f}")

# ----------------------------------------------------------------
# 7. Evaluation
# ----------------------------------------------------------------
def predict(model, X):
    logits = model.forward(X)
    return np.argmax(logits, axis=1)

y_test_labels = one_hot_to_label(y_test)
test_preds = predict(model, x_test)
test_acc = accuracy(test_preds, y_test_labels)
print(f"\nTest Accuracy on subset: {test_acc * 100:.2f}%")
