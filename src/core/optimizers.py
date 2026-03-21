import cupy as cp

class SGDMomentum:
    def __init__(self, layers, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.trainable_layers = [layer for layer in layers if hasattr(layer, 'weights')]
        
        self.velocities = []
        for layer in self.trainable_layers:
            self.velocities.append({
                'W' : cp.zeros_like(layer.weights),
                'b' : cp.zeros_like(layer.biases)
            })
    
    def step(self):
        for idx, layer in enumerate(self.trainable_layers):
            self.velocities[idx]['W'] = (self.momentum * self.velocities[idx]['W']) - (self.learning_rate * layer.dW)
            self.velocities[idx]['b'] = (self.momentum * self.velocities[idx]['b']) - (self.learning_rate * layer.db)
            
            layer.weights += self.velocities[idx]['W']
            layer.biases += self.velocities[idx]['b']