import cv2
import numpy as np
from gluoncv import model_zoo, data, utils
import gluoncv as gcv
import mxnet as mx

#net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
#net = model_zoo.get_model('yolo3_mobilenet1.0_voc', pretrained=True)
net = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True)

# Capturing video from webcam:
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Handles the mirroring of the current frame
    frame = cv2.flip(frame,1)

    # Image pre-processing
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=256, max_size=700)
    
    # Run frame through network
    class_IDs, scores, bounding_boxes = net(rgb_nd)

    # Display the result
    img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
    gcv.utils.viz.cv_plot_image(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()