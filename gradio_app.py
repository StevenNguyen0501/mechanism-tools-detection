import time
import numpy as np
from ultralytics import YOLO
import PIL.Image as Image
import gradio as gr

class_names = {
    0: 'caliper',
    1: 'crowbar',
    2: 'folding ruler',
    3: 'hammer',
    4: 'handy work light',
    5: 'paper cutter',
    6: 'pliers',
    7: 'pocket knife',
    8: 'screwdriver',
    9: 'wrench'
}

# Load the model
model = YOLO('best.pt')

def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    timestamp = int(time.time())

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    img.save(f'input/input_{timestamp}.jpg')
    im.save(f'output/output_{timestamp}.jpg')
    return im

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    live=True,
    title="VNEXT - Mechanical Tools Detection",
    description="Upload images for inference.",
    examples=[
        ["input-1.jpg", 0.25, 0.45],
        ["input-2.jpg", 0.25, 0.45],
    ],
)

if __name__ == '__main__':
    iface.launch()