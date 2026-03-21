import cupy as cp
from utils.im2col import im2col_indices, col2im_indices

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.cache = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        fan_in = in_channels * kernel_size * kernel_size
        self.weights = cp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * cp.sqrt(2.0 / fan_in)
        self.biases = cp.zeros((out_channels, 1))

    def forward(self, X):
        n_samples, _, h_in, w_in = X.shape
        
        h_out = (h_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_out = (w_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        X_col = im2col_indices(X, self.kernel_size, self.kernel_size, self.padding, self.stride)
        
        W_row = self.weights.reshape(self.out_channels, -1)
        
        Z_col = W_row @ X_col + self.biases
        
        Z = Z_col.reshape(self.out_channels, h_out, w_out, n_samples)
        Z = Z.transpose(3, 0, 1, 2)
        
        self.cache['X'] = X
        self.cache['X_col'] = X_col
        self.cache['W_row'] = W_row
        
        return Z

    def backward(self, dZ):
        X = self.cache['X']
        X_col = self.cache['X_col']
        W_row = self.cache['W_row']
        
        dZ_col = dZ.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        dW = dZ_col @ X_col.T
        db = cp.sum(dZ_col, axis=1, keepdims=True)
        
        dX_col = W_row.T @ dZ_col
        
        dX = col2im_indices(dX_col, X.shape, self.kernel_size, self.kernel_size, self.padding, self.stride)
        
        self.dW = dW.reshape(self.weights.shape)
        self.db = db
        
        return dX

class MaxPool2D:
    def __init__(self, pool_size, stride=None):
        self.cache = {}
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        
    def forward(self, X):
        N, C, H, W = X.shape
        
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        X_reshaped = X.reshape(N * C, 1, H, W)
        
        X_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride)
        
        max_idx = cp.argmax(X_col, axis=0)
        out = X_col[max_idx, cp.arange(max_idx.size)]
        
        out = out.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)
        
        self.cache['X_shape'] = X.shape
        self.cache['X_col_shape'] = X_col.shape
        self.cache['max_idx'] = max_idx
        
        return out
    
    def backward(self, dZ):
        X_shape = self.cache['X_shape']
        X_col_shape = self.cache['X_col_shape']
        max_idx = self.cache['max_idx']
        N, C, H, W = X_shape
        
        dZ_flat = dZ.transpose(2, 3, 0, 1).ravel()
        
        dX_col = cp.zeros(X_col_shape)
        
        dX_col[max_idx, cp.arange(max_idx.size)] = dZ_flat
        
        dX_reshaped = col2im_indices(dX_col, (N * C, 1, H, W), self.pool_size, self.pool_size, padding=0, stride=self.stride)
        
        dX = dX_reshaped.reshape(X_shape)
        
        return dX

class Flatten:
    def __init__(self):
        self.cache = {}
        
    def forward(self, X):
        self.cache['X_shape'] = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
        
    def backward(self, dZ):
        return dZ.reshape(self.cache['X_shape'])
    
class Dropout:
    def __init__(self, p=0.5):
        self.cache = {}
        self.p = p
        self.training = True

    def forward(self, X):
        if not self.training:
            return X
            
        self.mask = (cp.random.rand(*X.shape) > self.p) / (1.0 - self.p)
        return X * self.mask

    def backward(self, dZ):
        return dZ * self.mask

class Linear:
    def __init__(self, input_dimension, output_dimension):
        self.cache = {}
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        
        self.weights = cp.random.randn(input_dimension, output_dimension) * cp.sqrt(2.0 / input_dimension)
        self.biases = cp.zeros((1, output_dimension))
    
    def forward(self, X):
        Z = X @ self.weights + self.biases
        
        self.cache['X'] = X
        
        return Z
     
    def backward(self, dZ):
        X = self.cache['X']
        
        dW = X.T @ dZ
        db = cp.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.weights.T
        
        self.dW = dW
        self.db = db
        
        return dX
    
class ReLU:
    def __init__(self):
        self.cache = {}
    
    def forward(self, X):
        self.cache['X'] = X
        return cp.maximum(0, X)
    
    def backward(self, dA):
        X = self.cache['X']
        dX = dA * (X > 0) 
        return dX