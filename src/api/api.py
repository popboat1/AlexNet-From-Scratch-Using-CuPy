from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import cv2
import base64
import math
import os
import asyncio

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SafeDense(tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

class SafeConv2D(tf.keras.layers.Conv2D):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'alexnet_cifar10_keras.h5')

model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={
        'Dense': SafeDense, 
        'Conv2D': SafeConv2D
    }
)

conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=[layer.output for layer in conv_layers])

CIFAR10_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def generate_feature_grid(feature_map, max_features=64):
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]
        
    height, width, channels = feature_map.shape
    num_features = min(channels, max_features)
    grid_size = math.ceil(math.sqrt(num_features))
    
    grid_image = np.zeros((grid_size * height, grid_size * width), dtype=np.float32)
    
    for i in range(num_features):
        row = i // grid_size
        col = i % grid_size
        
        channel_img = feature_map[:, :, i]
        channel_img -= channel_img.min()
        if channel_img.max() > 0:
            channel_img /= channel_img.max()
        channel_img *= 255.0
        
        y_start, y_end = row * height, (row + 1) * height
        x_start, x_end = col * width, (col + 1) * width
        grid_image[y_start:y_end, x_start:x_end] = channel_img

    grid_image = np.uint8(grid_image)
    
    colored_grid = cv2.applyColorMap(grid_image, cv2.COLORMAP_VIRIDIS)
    
    b_channel, g_channel, r_channel = cv2.split(colored_grid)
    alpha_channel = grid_image # Use the raw grayscale intensity as the alpha map
    
    transparent_grid = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    _, buffer = cv2.imencode('.png', transparent_grid)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img_resized = cv2.resize(img, (227, 227))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    activations = feature_extractor.predict(img_batch)
    
    predictions = model.predict(img_batch)
    class_idx = np.argmax(predictions[0])
    
    layer_data = []
    for i, activation in enumerate(activations):
        b64_image = generate_feature_grid(activation)
        
        layer_data.append({
            "layer_index": i + 1,
            "shape": activation.shape[1:], 
            "texture_b64": f"data:image/jpeg;base64,{b64_image}"
        })
        
    return {
        "prediction": CIFAR10_CLASSES[class_idx],
        "layers": layer_data
    }

@app.websocket("/ws/predict-video")
async def predict_video_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket Connected for Video Stream")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue

            img_resized = cv2.resize(img, (227, 227))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            activations = feature_extractor.predict(img_batch, verbose=0)
            predictions = model.predict(img_batch, verbose=0)
            class_idx = np.argmax(predictions[0])
            
            layer_data = []
            for i, activation in enumerate(activations):
                b64_image = generate_feature_grid(activation)
                layer_data.append({
                    "layer_index": i + 1,
                    "shape": activation.shape[1:], 
                    "texture_b64": f"data:image/png;base64,{b64_image}"
                })
                
            await websocket.send_json({
                "prediction": CIFAR10_CLASSES[class_idx],
                "layers": layer_data
            })
            
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print("WebSocket Disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        
app.mount("/", StaticFiles(directory="static", html=True), name="static")