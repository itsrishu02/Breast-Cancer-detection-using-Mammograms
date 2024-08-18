import torch
import os
import torchvision
import random
import supervision as sv
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from pytorch_lightning import Trainer
from pycocotools.coco import COCO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.ops as ops

from transformers import DetrImageProcessor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

dataset = 'D:\\Sem 2\\CV\\Assignment_4\\mammo_1k\\coco_1k'

VAL_DIRECTORY = os.path.join(dataset, "val2017")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor, 
        train: bool
    ):
        if train:
            annotation_file_path = "D:\\Sem 2\\CV\\Assignment_4\\mammo_1k\\coco_1k\\annotations\\instances_train2017.json"
        else:
            annotation_file_path = "D:\\Sem 2\\CV\\Assignment_4\\mammo_1k\\coco_1k\\annotations\\instances_val2017.json"

        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    

TEST_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)

categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}


class Detr(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",num_labels=len(id2label),ignore_mismatched_sizes=True)

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Detr()
model.load_state_dict(torch.load('CV_transformer.pt'))
model.to(DEVICE)

# utils
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()

for i in range(20,50):
    # select random image
    image_ids = TEST_DATASET.coco.getImgIds()
    image_id= i
    print('Image #{}'.format(image_id))

    # load image and annotations
    image = TEST_DATASET.coco.loadImgs(image_id)[0]
    annotations = TEST_DATASET.coco.imgToAnns[image_id]
    image_path = os.path.join(TEST_DATASET.root, image['file_name'])
    image = cv2.imread(image_path)

    if len(annotations) > 0:

        # Annotate ground truth
        detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
        labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
        frame_ground_truth = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)


        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
            outputs = model(**inputs)

            # post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
            results = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=0.4, 
                target_sizes=target_sizes
            )[0]

            detections = sv.Detections.from_transformers(transformers_results=results)
            print("Number of detections before NMS:", len(detections))
            
            if detections:
                # Check for None values in detections
                if any(det is None for det in detections):
                    print("Warning: There are None values in detections. Skipping NMS.")
                else:
                    labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
                    frame_detections = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

                    detections = [det for det in detections if det is not None]

                    # Now convert to tensor
                    boxes = torch.tensor([det[0] for det in detections]).to(DEVICE)
                    print(boxes)
                    scores = torch.tensor([det[1] for det in detections]).to(DEVICE)  
                    labels = torch.tensor([det[2] for det in detections]).to(DEVICE)  
                    keep = ops.nms(boxes, scores, iou_threshold=0.25)  
                    filtered_detections = [detections[i] for i in keep]  
                    print("Number of detections after NMS:", len(filtered_detections))  

                    # Annotate detections after NMS
                    labels_nms = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in filtered_detections]
                    frame_detections_nms = box_annotator.annotate(scene=image.copy(), detections=filtered_detections, labels=labels_nms)
            else:
                print("No valid detections found. Skipping NMS.")  # Add this line for debugging

        # Combine both images side by side and display
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cv2.cvtColor(frame_ground_truth, cv2.COLOR_BGR2RGB))
        axs[0].axis('off')
        axs[0].set_title('Ground Truth')

        axs[1].imshow(cv2.cvtColor(frame_detections_nms, cv2.COLOR_BGR2RGB))
        axs[1].axis('off')
        axs[1].set_title('Detections')

        plt.show()
    else:
        print("Annotation for this image in available")




