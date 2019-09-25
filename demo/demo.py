from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
config_file = "/home/b532/zyh/python_code/facebook_mask/" \
              "maskrcnn-benchmark/configs/retinanet/retinanet_R-50-FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = cv2.imread('/home/b532/zyh/python_code/small_plane_new/JPEGImages/15.jpg')
predictions = coco_demo.run_on_opencv_image(image)
cv2.imshow('out', predictions)
cv2.waitKey()