# app.py -- single-image demo for the AI3003 leaf classifier (Group 7)
"""Gradio demo: upload one tomato leaf photo, get the predicted class.

Run with:  python app.py   (opens http://127.0.0.1:7860)
"""
from pathlib import Path

import gradio as gr
import torch
from torchvision import transforms

from train_baseline import CLASS_NAMES, build_model

ROOT = Path(__file__).parent
CKPT = ROOT / "runs" / "E3_resnet18_pre" / "best.pt"
EXAMPLE_DIR = ROOT / "examples"

# pretty names for the interface; the model itself uses the underscored labels
DISPLAY = {"healthy": "Healthy", "early_blight": "Early blight", "late_blight": "Late blight"}

# --------------------------------------------------------------------------- #
# 1. Model
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not CKPT.exists():
    raise SystemExit(
        f"Checkpoint not found: {CKPT}\n"
        "Train it first (about two minutes on a laptop GPU):\n"
        "  python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color "
        "--model resnet18 --pretrained --epochs 20 --batch-size 32 --seed 42 "
        "--out-dir runs/E3_resnet18_pre"
    )

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model, note = build_model(ckpt["model"], num_classes=len(CLASS_NAMES))
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE).eval()
print(f"loaded: {note} | epoch {ckpt.get('epoch')} | val macro-F1 {ckpt.get('val_macro_f1'):.4f}")
print(f"device: {DEVICE}")
print(f"classes: {CLASS_NAMES}")

# Must match training exactly: 256 short side -> centre crop 224 -> ImageNet stats
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --------------------------------------------------------------------------- #
# 2. Prediction
# --------------------------------------------------------------------------- #
def predict(img):
    """PIL image -> ({class: probability}, markdown verdict)."""
    if img is None:
        return {}, "### No image\nUpload a leaf photo, or click one of the samples below."

    x = preprocess(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0].cpu()

    scores = {DISPLAY[c]: float(p) for c, p in zip(CLASS_NAMES, probs)}
    top = max(scores, key=scores.get)
    verdict = (
        f"### {top}\n"
        f"confidence **{scores[top] * 100:.1f}%**  \n"
        f"<span style='color:#6b7280'>margin over 2nd choice: "
        f"**{(scores[top] - sorted(scores.values())[-2]) * 100:.1f} pts**</span>"
    )
    return scores, verdict


# --------------------------------------------------------------------------- #
# 3. Interface
# --------------------------------------------------------------------------- #
THEME = gr.themes.Soft(primary_hue="green", secondary_hue="green", neutral_hue="slate")

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto;}
#header {text-align: center; margin-bottom: 0.2rem;}
#header h1 {margin-bottom: 0.1rem; font-size: 2rem;}
#header p {color: #4b5563; margin-top: 0;}
#footer {font-size: 0.85rem; color: #4b5563;}
"""

SAMPLES = sorted(str(p) for p in EXAMPLE_DIR.glob("*.jpg"))

with gr.Blocks(theme=THEME, css=CSS, title="Tomato Leaf Disease Classifier - AI3003 Group 7") as demo:
    gr.Markdown(
        "# Tomato Leaf Disease Classifier\n"
        "**AI3003 Group 7** · ResNet18 trained on PlantVillage · tested on PlantDoc",
        elem_id="header",
    )

    with gr.Row():
        with gr.Column(scale=3, min_width=320):
            image = gr.Image(
                type="pil",
                label="Leaf photo",
                sources=["upload", "clipboard"],
                height=360,
            )
            with gr.Row():
                clear_btn = gr.ClearButton(image, value="Clear")
                run_btn = gr.Button("Classify", variant="primary", scale=3)

        with gr.Column(scale=2, min_width=280):
            label = gr.Label(num_top_classes=3, label="Prediction")
            verdict = gr.Markdown(
                "### Waiting\nUpload a photo and press **Classify**, "
                "or click a sample below.",
                elem_id="verdict",
            )

    gr.Examples(
        examples=[[p] for p in SAMPLES],
        inputs=image,
        outputs=[label, verdict],
        fn=predict,
        label="Sample photos from PlantDoc (used only for testing, never for training)",
        examples_per_page=4,
    )

    gr.Markdown(
        "**Model** ResNet18, 11.2M parameters · PlantVillage test accuracy **99.0%** · "
        "PlantDoc accuracy **61.8%**  \n"
        "**Why the gap** Field photos carry backgrounds, lighting and framing the model never saw "
        "in training. The healthy class is the weakest: recall drops from 98.7% to **12.7%** on "
        "PlantDoc. Confidence is therefore not correctness - treat a confident answer on a cluttered "
        "photo with care.  \n"
        "**Data** PlantVillage (Mohanty et al., 2016) · PlantDoc (Singh et al., 2020), external test "
        "set only.",
        elem_id="footer",
    )

    run_btn.click(predict, inputs=image, outputs=[label, verdict])

if __name__ == "__main__":
    demo.launch(show_api=False)
