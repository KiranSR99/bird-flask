import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import pickle
import datetime

# =======================
# ===== CNN LAYERS ======
# =======================
class Layer:
    def forward(self, inputs):
        raise NotImplementedError
    def backward(self, grad_output, lr, t=1):
        raise NotImplementedError

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        scale = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        # Adam params
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        k = self.kernel_size
        pad = self.padding
        s = self.stride
        self.x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
        out_h = (H + 2*pad - k) // s + 1
        out_w = (W + 2*pad - k) // s + 1
        out = np.zeros((N, self.out_channels, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                region = self.x_padded[:, :, i*s:i*s+k, j*s:j*s+k]
                out[:, :, i, j] = np.tensordot(region, self.W, axes=([1,2,3],[1,2,3])) + self.b
        return out

    def backward(self, grad_output, lr, t=1, beta1=0.9, beta2=0.999, eps=1e-8):
        N, C_out, H_out, W_out = grad_output.shape
        k = self.kernel_size
        pad = self.padding
        s = self.stride

        grad_x_padded = np.zeros_like(self.x_padded)
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)

        for i in range(H_out):
            for j in range(W_out):
                region = self.x_padded[:, :, i*s:i*s+k, j*s:j*s+k]
                for c in range(C_out):
                    grad_W[c] += np.sum(region * grad_output[:, c:c+1, i:i+1, j:j+1], axis=0)
                for n in range(N):
                    grad_x_padded[n, :, i*s:i*s+k, j*s:j*s+k] += np.sum(
                        self.W * grad_output[n, :, i, j][:, None, None, None], axis=0
                    )
        grad_b = np.sum(grad_output, axis=(0, 2, 3))
        grad_x = grad_x_padded[:, :, pad:pad+self.x.shape[2], pad:pad+self.x.shape[3]]

        # Adam update
        self.mW = beta1*self.mW + (1-beta1)*grad_W
        self.vW = beta2*self.vW + (1-beta2)*(grad_W**2)
        self.mb = beta1*self.mb + (1-beta1)*grad_b
        self.vb = beta2*self.vb + (1-beta2)*(grad_b**2)

        mW_hat = self.mW / (1-beta1**t)
        vW_hat = self.vW / (1-beta2**t)
        mb_hat = self.mb / (1-beta1**t)
        vb_hat = self.vb / (1-beta2**t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return grad_x

class MaxPool2D(Layer):
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        s = self.stride
        k = self.size
        out_h = (H - k)//s + 1
        out_w = (W - k)//s + 1
        out = np.zeros((N, C, out_h, out_w))
        self.max_indices = np.zeros_like(x, dtype=bool)
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, :, i*s:i*s+k, j*s:j*s+k]
                max_vals = np.max(region, axis=(2, 3), keepdims=True)
                mask = (region == max_vals)
                self.max_indices[:, :, i*s:i*s+k, j*s:j*s+k] += mask
                out[:, :, i, j] = max_vals.squeeze()
        return out

    def backward(self, grad_output, lr, t=1, **kwargs):
        grad_x = np.zeros_like(self.x)
        s = self.stride
        k = self.size
        out_h, out_w = grad_output.shape[2:]
        for i in range(out_h):
            for j in range(out_w):
                grad_x[:, :, i*s:i*s+k, j*s:j*s+k] += self.max_indices[:, :, i*s:i*s+k, j*s:j*s+k] * grad_output[:, :, i:i+1, j:j+1]
        return grad_x

class Flatten(Layer):
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, grad_output, lr, t=1, **kwargs):
        return grad_output.reshape(self.shape)

class Dense(Layer):
    def __init__(self, in_features, out_features, l2=0.0):
        scale = np.sqrt(2. / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.l2 = l2
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output, lr, t=1, beta1=0.9, beta2=0.999, eps=1e-8):
        grad_W = self.x.T @ grad_output + self.l2 * self.W
        grad_b = np.sum(grad_output, axis=0)
        grad_x = grad_output @ self.W.T

        self.mW = beta1*self.mW + (1-beta1)*grad_W
        self.vW = beta2*self.vW + (1-beta2)*(grad_W**2)
        self.mb = beta1*self.mb + (1-beta1)*grad_b
        self.vb = beta2*self.vb + (1-beta2)*(grad_b**2)

        mW_hat = self.mW / (1-beta1**t)
        vW_hat = self.vW / (1-beta2**t)
        mb_hat = self.mb / (1-beta1**t)
        vb_hat = self.vb / (1-beta2**t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return grad_x

class ReLU(Layer):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def backward(self, grad_output, lr, t=1, **kwargs):
        return grad_output * self.mask

def softmax_crossentropy(logits, y_true):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    N = y_true.shape[0]
    loss = -np.sum(y_true * np.log(probs + 1e-9)) / N
    return loss, probs

def softmax_crossentropy_backward(probs, y_true):
    return (probs - y_true) / y_true.shape[0]

class SimpleCNN:
    def __init__(self):
        self.layers = [
            Conv2D(3, 8, 3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(32*32*8, 10)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr, t):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr, t=t)

# =======================
# ===== DATA LOADER ======
# =======================
IMG_SIZE = 64
NUM_CLASSES = 10
DATA_DIR = "bird/images"  # set path to your CUB-200 subset
CLASS_LIST = None

def load_cub200_subset(data_dir, num_classes=10, img_size=64, class_list=None):
    all_classes = sorted(os.listdir(data_dir))
    if class_list is None:
        classes = all_classes[:num_classes]
    else:
        classes = class_list
        num_classes = len(classes)

    X, y = [], []
    for idx, cname in enumerate(classes):
        img_files = glob(os.path.join(data_dir, cname, "*.jpg"))
        for f in img_files:
            img = cv2.imread(f)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # (C,H,W)
            X.append(img)
            y.append(idx)
    X = np.array(X, dtype=np.float32)
    y = np.eye(num_classes)[y]  # one-hot
    return X, y, classes

# =======================
# ===== ACCURACY ========
# =======================
def accuracy(model, X, y):
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y, axis=1)
    return np.mean(preds == labels)

# =======================
# ===== LOGGING =========
# =======================
log_file = f"the_scratch_training_log.txt"

def log_epoch(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.datetime.now()} - {message}\n")

# =======================
# ===== TRAINING LOOP ====
# =======================
if __name__ == "__main__":
    X, y, classes = load_cub200_subset(DATA_DIR, NUM_CLASSES, IMG_SIZE, CLASS_LIST)
    log_epoch(f"Loaded {X.shape[0]} images from {len(classes)} classes")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    log_epoch(f"Train: {X_train.shape}, Val: {X_val.shape}")

    model = SimpleCNN()
    epochs = 10
    batch_size = 32
    lr = 0.001
    t = 0

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        total_loss = 0

        for i in range(0, len(X_train), batch_size):
            t += 1
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            logits = model.forward(X_batch)
            loss, probs = softmax_crossentropy(logits, y_batch)
            grad = softmax_crossentropy_backward(probs, y_batch)
            model.backward(grad, lr, t)
            total_loss += loss

        train_acc = accuracy(model, X_train[:500], y_train[:500])
        val_acc = accuracy(model, X_val, y_val)
        log_epoch(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, "
                  f"Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")

    # Save model
    with open("simple_cnn_birds.pkl", "wb") as f:
        pickle.dump(model, f)
    log_epoch("Model saved as simple_cnn_birds.pkl")
