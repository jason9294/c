#coding=UTF-8
import argparse
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from PIL import Image
from PIL import ImageDraw
import cv2
import numpy
import time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Run object detection with Edge TPU.")
    parser.add_argument('--model', required=True, help='Path to the .tflite model file.')
    parser.add_argument('--label', required=True, help='Path to the label file.')
    return parser.parse_args()

def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')

def main(args:argparse.Namespace):
    labels = read_label_file(args.label)
    engine = make_interpreter(args.model)
    engine.allocate_tensors()

    cap = cv2.VideoCapture(1)
    print('Camera initialized.')

    while cap.isOpened():
        ret, frame = cap.read()  # Read camera feed
        start_time = time.time()  # Start timing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert OpenCV to PIL
        
        draw = ImageDraw.Draw(image)  # For drawing bounding boxes
        
        _, scale = common.set_resized_input(engine, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        engine.invoke()
        objs = detect.get_objects(engine, 0.4, scale)
        
        for obj in objs:
            print(labels.get(obj.id, obj.id))
            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)
        draw_objects(draw, objs, labels)

        if not objs:
            print('No objects detected.')
        image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)  # Convert PIL back to OpenCV
        cv2.imshow('img', image)  
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
        # Calculate FPS
        print('FPS=', 1 / (time.time() - start_time))
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    print(args, '='*200, sep='\n')
    main(args)