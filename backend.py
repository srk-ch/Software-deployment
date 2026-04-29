"""
ACL Tear Detection API — Flask Backend
Handles .npy MRI uploads, runs inference + Grad-CAM++, returns results as JSON
"""

import os
import io
import base64
import sys
from pathlib import Path
import traceback

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from scipy.ndimage import gaussian_filter

BASE_DIR = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
CORS(app)  # Allow all origins (restrict in production)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "best_model.pth"))
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.4377"))

# ============================================================
# MODEL DEFINITION (must match training exactly)
# ============================================================

class SliceEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = models.resnet18(weights=None)
        original_weight = backbone.conv1.weight.data
        new_weight = original_weight.mean(dim=1, keepdim=True)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = new_weight
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        return x.flatten(start_dim=1)


class AttentionPool(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights


class MRNetModel(nn.Module):
    def __init__(self, pretrained=False, dropout=0.3):
        super().__init__()
        self.encoder_axial    = SliceEncoder(pretrained=pretrained)
        self.encoder_coronal  = SliceEncoder(pretrained=pretrained)
        self.encoder_sagittal = SliceEncoder(pretrained=pretrained)
        self.pool_axial    = AttentionPool(512)
        self.pool_coronal  = AttentionPool(512)
        self.pool_sagittal = AttentionPool(512)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def _encode_plane(self, volume, encoder, pool):
        B, S, C, H, W = volume.shape
        flat = volume.view(B * S, C, H, W)
        feats = encoder(flat).view(B, S, -1)
        pooled, weights = pool(feats)
        return pooled, weights

    def forward(self, axial, coronal, sagittal):
        f_ax,  w_ax  = self._encode_plane(axial,    self.encoder_axial,    self.pool_axial)
        f_cor, w_cor = self._encode_plane(coronal,  self.encoder_coronal,  self.pool_coronal)
        f_sag, w_sag = self._encode_plane(sagittal, self.encoder_sagittal, self.pool_sagittal)
        fused = torch.cat([f_ax, f_cor, f_sag], dim=1)
        logit = self.classifier(fused)
        return logit, (w_ax, w_cor, w_sag)


# ============================================================
# GRAD-CAM++
# ============================================================

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, logit):
        self.model.zero_grad()
        logit.backward(retain_graph=True)
        grads = self.gradients
        acts  = self.activations
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        denom = 2 * grads_sq + acts * grads_cu
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads_sq / denom
        weights = (alpha * F.relu(grads)).mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.mean(dim=0).float().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(self, cam, original_slice, alpha=0.5):
        h, w = original_slice.shape
        cam_resized = cv2.resize(np.asarray(cam, dtype=np.float32), (w, h))
        slc_norm = original_slice.copy().astype(np.float32)
        slc_norm = (slc_norm - slc_norm.min()) / (slc_norm.max() - slc_norm.min() + 1e-8)
        slc_uint8 = (slc_norm * 255).astype(np.uint8)
        slc_rgb = cv2.cvtColor(slc_uint8, cv2.COLOR_GRAY2RGB)
        heatmap = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = (alpha * heatmap + (1 - alpha) * slc_rgb).astype(np.uint8)
        return blended, cam_resized


def get_target_layer(encoder):
    return list(encoder.features[-2].children())[-1].conv2


# ============================================================
# PREPROCESSING (matches training exactly)
# ============================================================

def apply_clahe(slice_2d):
    s = slice_2d.astype(np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    s = (s * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    s = clahe.apply(s)
    return s.astype(np.float32) / 255.0


def apply_gaussian_denoise(slice_2d, sigma=0.5):
    return gaussian_filter(slice_2d.astype(np.float32), sigma=sigma)


def preprocess_volume(volume, target_size=224):
    processed = []
    for s in range(volume.shape[0]):
        slc = volume[s].astype(np.float32)
        slc = cv2.resize(slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        slc = apply_gaussian_denoise(slc, sigma=0.5)
        slc = apply_clahe(slc)
        processed.append(slc)
    processed = np.stack(processed, axis=0)
    mean = processed.mean()
    std  = processed.std() + 1e-8
    processed = (processed - mean) / std
    return processed.astype(np.float32)


def sample_slices(volume, num_slices=20):
    S = volume.shape[0]
    if S >= num_slices:
        idx = np.linspace(0, S - 1, num_slices, dtype=int)
    else:
        idx = np.pad(np.arange(S), (0, num_slices - S), mode='edge')
    return volume[idx]


def volume_to_tensor(volume, num_slices=20):
    vol = sample_slices(preprocess_volume(volume), num_slices)
    t = torch.tensor(vol, dtype=torch.float32).unsqueeze(1)  # (S, 1, 224, 224)
    return t.unsqueeze(0)  # (1, S, 1, 224, 224)


def arr_to_b64(arr):
    _, buf = cv2.imencode('.png', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')


def gray_to_b64(arr):
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    uint8 = (norm * 255).astype(np.uint8)
    _, buf = cv2.imencode('.png', uint8)
    return base64.b64encode(buf).decode('utf-8')


# ============================================================
# LOAD MODEL AT STARTUP
# ============================================================

model = None
gradcam_axial = gradcam_coronal = gradcam_sagittal = None
model_metadata = {}


def extract_state_dict(checkpoint):
    """Support common checkpoint layouts produced by plain PyTorch and Lightning."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary; cannot extract model weights.")

    for key in ("model_state", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state

    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint

    raise ValueError("Could not find a compatible state dict in checkpoint.")

def load_model():
    global model, gradcam_axial, gradcam_coronal, gradcam_sagittal, model_metadata
    if not Path(MODEL_PATH).exists():
        print(f"[WARN] Model file not found at {MODEL_PATH}. Run inference will fail until placed.")
        return
    print(f"[INFO] Loading model from {MODEL_PATH} on {DEVICE}")
    m = MRNetModel(pretrained=False).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state = extract_state_dict(checkpoint)
    # Handle lightning prefix
    state = {k.replace("model.", ""): v for k, v in state.items()}
    incompatible = m.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"[WARN] Missing keys while loading checkpoint: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys while loading checkpoint: {len(incompatible.unexpected_keys)}")
    m.eval()
    model = m
    model_metadata = {
        "epoch": checkpoint.get("epoch"),
        "best": checkpoint.get("best"),
        "val": checkpoint.get("val"),
        "auc": checkpoint.get("auc"),
        "threshold": PREDICTION_THRESHOLD,
    } if isinstance(checkpoint, dict) else {"threshold": PREDICTION_THRESHOLD}
    gradcam_axial    = GradCAMPlusPlus(model, get_target_layer(model.encoder_axial))
    gradcam_coronal  = GradCAMPlusPlus(model, get_target_layer(model.encoder_coronal))
    gradcam_sagittal = GradCAMPlusPlus(model, get_target_layer(model.encoder_sagittal))
    print("[INFO] Model loaded successfully ✅")

load_model()


# ============================================================
# ROUTES
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "threshold": PREDICTION_THRESHOLD,
        "model_metadata": model_metadata,
    })


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "acl_scan.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects multipart/form-data with:
      axial    : .npy file
      coronal  : .npy file
      sagittal : .npy file

    Returns JSON with prediction, confidence, attention weights,
    and base64-encoded Grad-CAM++ overlays for each plane.
    """
    if model is None:
        return jsonify({"error": f"Model not loaded. Place best_model.pth at {MODEL_PATH}"}), 503

    try:
        # Load uploaded volumes
        def read_npy(key):
            f = request.files.get(key)
            if f is None:
                raise ValueError(f"Missing file: {key}")
            buf = np.frombuffer(f.read(), dtype=np.uint8)
            # .npy files — load via BytesIO
            arr = np.load(io.BytesIO(f.stream.read() if hasattr(f, 'stream') else buf.tobytes()))
            return arr

        files = {}
        for plane in ["axial", "coronal", "sagittal"]:
            f = request.files.get(plane)
            if f is None:
                return jsonify({"error": f"Missing file: '{plane}'"}), 400
            raw = f.read()
            arr = np.load(io.BytesIO(raw))
            files[plane] = arr

        ax_vol  = volume_to_tensor(files["axial"]).to(DEVICE)
        cor_vol = volume_to_tensor(files["coronal"]).to(DEVICE)
        sag_vol = volume_to_tensor(files["sagittal"]).to(DEVICE)

        ax_vol.requires_grad_(True)
        cor_vol.requires_grad_(True)
        sag_vol.requires_grad_(True)

        with torch.enable_grad():
            logit, (w_ax, w_cor, w_sag) = model(ax_vol, cor_vol, sag_vol)
            prob = torch.sigmoid(logit).item()

            # Grad-CAM++ for each plane
            cam_ax  = gradcam_axial.generate(logit)
            cam_cor = gradcam_coronal.generate(logit)
            cam_sag = gradcam_sagittal.generate(logit)

        prediction = int(prob >= PREDICTION_THRESHOLD)
        confidence = max(
            0.0,
            min(
                1.0,
                abs(prob - PREDICTION_THRESHOLD) / max(PREDICTION_THRESHOLD, 1.0 - PREDICTION_THRESHOLD)
            )
        )

        # --- Build visualizations ---
        def make_gradcam_images(raw_volume, cam, gradcam_obj, n_show=5):
            """Return n_show slice overlays around the center of the volume."""
            proc = preprocess_volume(raw_volume)
            S = proc.shape[0]
            mid = S // 2
            indices = np.linspace(max(0, mid - n_show // 2),
                                  min(S - 1, mid + n_show // 2),
                                  n_show, dtype=int)
            images = []
            for i in indices:
                slc = proc[i]
                overlay_img, _ = gradcam_obj.overlay(cam, slc, alpha=0.5)
                images.append({
                    "slice_idx": int(i),
                    "raw_b64":    gray_to_b64(slc),
                    "overlay_b64": arr_to_b64(overlay_img)
                })
            return images

        gradcam_images = {
            "axial":    make_gradcam_images(files["axial"],    cam_ax,  gradcam_axial),
            "coronal":  make_gradcam_images(files["coronal"],  cam_cor, gradcam_coronal),
            "sagittal": make_gradcam_images(files["sagittal"], cam_sag, gradcam_sagittal),
        }

        # Attention weights (squeeze to list)
        attn = {
            "axial":    w_ax.squeeze().cpu().tolist(),
            "coronal":  w_cor.squeeze().cpu().tolist(),
            "sagittal": w_sag.squeeze().cpu().tolist(),
        }

        return jsonify({
            "prediction": prediction,         # 0 = Normal, 1 = ACL Tear
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "threshold": round(PREDICTION_THRESHOLD, 4),
            "label": "ACL Tear Detected" if prediction == 1 else "Normal — No Tear",
            "attention_weights": attn,
            "gradcam_images": gradcam_images,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/demo", methods=["GET"])
def demo():
    """Return a demo result with synthetic data — for UI testing without a real model."""
    import random
    pred = random.randint(0, 1)
    prob = round(random.uniform(0.65, 0.98) if pred == 1 else random.uniform(0.05, 0.35), 4)

    def fake_b64():
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.png', img)
        return base64.b64encode(buf).decode('utf-8')

    gradcam_images = {}
    for plane in ["axial", "coronal", "sagittal"]:
        gradcam_images[plane] = [
            {"slice_idx": i, "raw_b64": fake_b64(), "overlay_b64": fake_b64()}
            for i in range(5)
        ]

    return jsonify({
        "prediction": pred,
        "probability": prob,
        "confidence": round(abs(prob - 0.5) * 2, 4),
        "threshold": round(PREDICTION_THRESHOLD, 4),
        "label": "ACL Tear Detected" if pred == 1 else "Normal — No Tear",
        "attention_weights": {
            p: [round(1/20, 4)] * 20 for p in ["axial", "coronal", "sagittal"]
        },
        "gradcam_images": gradcam_images,
        "demo": True
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("APP_PORT", "7860")))
    app.run(host="0.0.0.0", port=port, debug=False)
