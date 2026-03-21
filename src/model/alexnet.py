from core.layers import Conv2D, MaxPool2D, Flatten, Linear, ReLU, Dropout
import pickle

import cupy as cp
import numpy as np
import cv2

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.criterion = None
        self.optimizer = None

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def compile(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion
    
    def save_weights(self, filepath="alexnet_weights.pkl"):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append({
                    'weights': cp.asnumpy(layer.weights),
                    'biases': cp.asnumpy(layer.biases)
                })
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model weights successfully saved to {filepath}")
        
    def load_weights(self, filepath="alexnet_weights.pkl"):
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = cp.asarray(params[param_idx]['weights'])
                layer.biases = cp.asarray(params[param_idx]['biases'])
                param_idx += 1
        print(f"Model weights successfully loaded from {filepath}")

    def fit(self, train_dataset, epochs, val_dataset=None, patience=None, min_delta=0.0):
        if self.optimizer is None or self.criterion is None:
            raise ValueError("Model must be compiled before fitting.")

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        temp_weights_path = "temp_best_weights.pkl"

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for step, (batch_data, batch_labels) in enumerate(train_dataset):
                X_batch_np = batch_data.numpy().transpose(0, 3, 1, 2) 
                Y_batch_np = batch_labels.numpy()
                
                X_batch = cp.asarray(X_batch_np)
                Y_batch = cp.asarray(Y_batch_np)

                predictions = self.forward(X_batch)
                
                batch_preds = cp.argmax(predictions, axis=1)
                batch_targets = cp.argmax(Y_batch, axis=1)
                correct_train += int(cp.sum(batch_preds == batch_targets))
                total_train += X_batch.shape[0]

                loss = self.criterion.forward(predictions, Y_batch)
                epoch_loss += float(loss)
                
                dZ = self.criterion.backward()
                self.backward(dZ)
                self.optimizer.step()
                
                del X_batch, Y_batch, predictions, dZ
                cp.get_default_memory_pool().free_all_blocks()

            avg_train_loss = epoch_loss / (step + 1)
            train_acc = correct_train / total_train
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)

            val_print_msg = ""
            if val_dataset is not None:
                val_loss_total = 0.0
                correct_val = 0
                total_val = 0
                
                for val_step, (val_data, val_labels) in enumerate(val_dataset):
                    X_val = cp.asarray(val_data.numpy().transpose(0, 3, 1, 2))
                    Y_val = cp.asarray(val_labels.numpy())
                    
                    val_predictions = self.forward(X_val)
                    val_loss_total += float(self.criterion.forward(val_predictions, Y_val))
                    
                    val_preds = cp.argmax(val_predictions, axis=1)
                    val_targets = cp.argmax(Y_val, axis=1)
                    correct_val += int(cp.sum(val_preds == val_targets))
                    total_val += X_val.shape[0]
                    
                    del X_val, Y_val, val_predictions
                    cp.get_default_memory_pool().free_all_blocks()

                avg_val_loss = val_loss_total / (val_step + 1)
                val_acc = correct_val / total_val
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)
                
                val_print_msg = f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"

                if patience is not None:
                    if avg_val_loss < (best_val_loss - min_delta):
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        self.save_weights(temp_weights_path)
                        val_print_msg += " (Improved!)"
                    else:
                        patience_counter += 1
                        val_print_msg += f" (Patience: {patience_counter}/{patience})"

            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}{val_print_msg}")

            if patience is not None and patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
                print("Restoring best model weights...")
                self.load_weights(temp_weights_path)
                break

        return history

    def predict(self, X):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
                
        Z = self.forward(X)
        
        Z_shifted = Z - cp.max(Z, axis=1, keepdims=True)
        exp_Z = cp.exp(Z_shifted)
        probabilities = exp_Z / cp.sum(exp_Z, axis=1, keepdims=True)
        
        predicted_classes = cp.argmax(probabilities, axis=1)
        
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
                
        return predicted_classes, probabilities
    

def build_alexnet():
    model = Sequential([
        Conv2D(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
        ReLU(),
        MaxPool2D(pool_size=3, stride=2),
        
        Conv2D(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
        ReLU(),
        MaxPool2D(pool_size=3, stride=2),
        
        Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
        ReLU(),
        
        Conv2D(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
        ReLU(),
        
        Conv2D(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
        ReLU(),
        
        MaxPool2D(pool_size=3, stride=2),
        
        Flatten(),
        
        Linear(input_dimension=9216, output_dimension=4096),
        ReLU(),
        Dropout(p=0.5),

        Linear(input_dimension=4096, output_dimension=4096),
        ReLU(),
        Dropout(p=0.5),
        
        Linear(input_dimension=4096, output_dimension=10)
    ])
    
    return model