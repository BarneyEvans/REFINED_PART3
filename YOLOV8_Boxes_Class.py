class BoundingBox:
    def __init__(self, box, confidence, cls_name):
        self.x1, self.y1, self.x2, self.y2 = box  # Corner coordinates
        self.confidence = confidence
        self.cls_name = cls_name
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1

    def get_details(self):
        return {
            "center": (self.center_x, self.center_y),
            "corners": (self.x1, self.y1, self.x2, self.y2),
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class": self.cls_name
        }

class DetectionInfo:
    def __init__(self, image_name, results, orig_shape):
        self.image_name = image_name
        self.orig_shape = orig_shape
        self.bounding_boxes = []
        self.class_names = results.names  # Assume results have 'names' attribute with class names
        self.load_boxes(results)

    def load_boxes(self, results):
        if results.boxes:
            for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                cls_name = self.class_names[int(cls_id)]  # Convert cls_id to integer and fetch class name
                bbox = BoundingBox(box.cpu().numpy(), conf.item(), cls_name)
                self.bounding_boxes.append(bbox)

    def get_summary(self):
        return {
            "image_name": self.image_name,
            "number_of_boxes": len(self.bounding_boxes),
            "boxes_info": [box.get_details() for box in self.bounding_boxes],
            "original_image_shape": self.orig_shape
        }
