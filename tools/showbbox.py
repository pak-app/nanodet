import cv2
import json
import numpy as np

def create_boxes(img_size, norm_ann):

    height, width, _ = img_size

    x_center, y_center, w, h = norm_ann
    x = int((x_center - w / 2) * width)
    y = int((y_center - h / 2) * height)
    w = int(w * width)
    h = int(h * height)

    return (x, y), (x + w, y + h)


if __name__ == "__main__":

    img = cv2.imread(r'C:\Users\p.alishah\Desktop\Projects\python\models\nanodet\datasets\data\RaccoonYD-38\test\raccoon-6_jpg.rf.e1ef482779f9ef651ec62ed3a9c1e2d7.jpg')
    
    # Load data from a local JSON file
    with open(r'C:\Users\p.alishah\Desktop\Projects\python\models\nanodet\workspace\nanodet-plus-m_416_Raccoon\20240801091605\results.json', 'r') as f:
        ann_pred = json.load(f)
    
    with open(r"C:\Users\p.alishah\Desktop\Projects\python\models\nanodet\datasets\data\RaccoonYD-38\test\raccoon-6_jpg.rf.e1ef482779f9ef651ec62ed3a9c1e2d7.txt", 'r') as file:
        ann_real = file.read().split()
    
    
    ann_real = list(map(float, ann_real[1:]))
    
    # Scale annotations
    p1, p2 = create_boxes(img.shape, ann_real)

    # Load the image
    print("Predicted annotatioins:", ann_pred[0]['bbox'])
    # Define bounding box coordinates
    x, y, w, h = ann_pred[0]['bbox']
    
    # Draw the bounding box
    cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
    cv2.rectangle(img, p1, p2, (255, 0, 0), 2)

    # Display the image
    cv2.imshow('Image with Bounding Box', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
