import cupy as cp

class CategoricalCrossEntropyLoss:
    def __init__(self):
        self.cache = {}
        
    def forward(self, Z, Y):
        """
        Z: Raw logits from the final Linear layer (batch_size, num_classes)
        Y: True labels, one-hot encoded (batch_size, num_classes)
        """
        Z_shifted = Z - cp.max(Z, axis=1, keepdims=True)
        exp_Z = cp.exp(Z_shifted)
        probabilities = exp_Z / cp.sum(exp_Z, axis=1, keepdims=True)
        
        self.cache['P'] = probabilities
        self.cache['Y'] = Y
        
        batch_size = Z.shape[0]
        
        P_clipped = cp.clip(probabilities, 1e-8, 1.0 - 1e-8)
        
        loss = -cp.sum(Y * cp.log(P_clipped)) / batch_size
        
        return loss
    
    def backward(self):
        P = self.cache['P']
        Y = self.cache['Y']
        batch_size = P.shape[0]
        
        dZ = (P - Y) / batch_size
        
        return dZ