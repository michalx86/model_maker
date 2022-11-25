# This example requires opencv-python instead of opencv-python-headless
#
# Full package requirements are:
#
# Package        Version
# -------------- -----------
# numpy          1.23.4
# opencv-python  4.6.0.66
# Pillow         9.3.0
# pip            22.0.4
# pycoral        2.0.0
# setuptools     58.1.0
# tflite-runtime 2.5.0.post1


import numpy as np
import cv2 
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import tflite_runtime.interpreter as tflite 
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file

LABELS_FILENAME = 'labels.txt'
TFLITE_FILENAME = 'efficientdet0-lite-gestures.tflite'
#LABELS_FILENAME = 'gesture_model/labelmap.txt'
#TFLITE_FILENAME = 'gesture_model/detect.tflite'

def draw_objects(draw, objs, scale_factor, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    color = tuple(int(c) for c in COLORS[obj.id])
    draw.rectangle([(bbox.xmin * scale_factor, bbox.ymin * scale_factor),
                    (bbox.xmax * scale_factor, bbox.ymax * scale_factor)],
                   outline=color, width=3)
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=15)
    draw.text((bbox.xmin * scale_factor + 4, bbox.ymin * scale_factor + 4),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill=color, font=font)

# Load the TF Lite model
labels = read_label_file(LABELS_FILENAME)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype=np.uint8)

interpreter = tflite.Interpreter(TFLITE_FILENAME)
interpreter.allocate_tensors()


cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened(): 
    ret, frame_bgr = cap.read()

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image_rgb = Image.fromarray(np.array(frame_rgb))
    #image_rgb.show()

    image = Image.fromarray(np.array(frame_bgr))
    #image.show()

    # Resize the image for input
    _, scale = common.set_resized_input(
        interpreter, image_rgb.size, lambda size: image_rgb.resize(size, Image.Resampling.LANCZOS))

    # Run inference
    interpreter.invoke()
    objs = detect.get_objects(interpreter, score_threshold=0.4, image_scale=scale)

    # Resize again to a reasonable size for display
    display_width = 500
    scale_factor = display_width / image.width
    height_ratio = image.height / image.width
    image = image.resize((display_width, int(display_width * height_ratio)))
    draw_objects(ImageDraw.Draw(image), objs, scale_factor, labels)
    cv2.imshow('object detection',  np.asarray(image))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

