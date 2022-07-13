import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.names", "r") as f:
    classes = f.read().splitlines()

#cap = cv2.VideoCapture('video4.mp4')
#cap = 'test_images/<your_test_image>.jpg'
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    #_, img = cap.read()
    img1 = cv2.imread("test_images/3.jpg")
    img = img1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

    # img2 = cv2.imread("test_images/2.jpg")
    # img3 = cv2.imread("test_images/3.jpg")
    # print(np.shape(img2))
    # img3 = cv2.resize(img3, (300,300))
        
    # horizontally concatenates images
    # of same height 
    # img = cv2.hconcat([img1, img2])
    # img = cv2.hconcat([img, img3])
    
    # show the output image
    # cv2.imshow('man_image.jpeg', img)
    
    height, width, _ = img.shape

    # samp = cv2.dnn.blobFromImage(img3, (416, 416), (0,0,0), swapRB=True, crop=False)
    samp = cv2.resize(img, (416, 416))
    blob = cv2.dnn.blobFromImage(samp, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    # cv2.imshow('man_image.jpeg', samp)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # print(scores)
            if confidence > 0.2:
                print(confidence)
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img1, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img1, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)

    cv2.imshow('Image', img1)
    key = cv2.waitKey(1)
    if key==27:
        break

#cap.release()
cv2.destroyAllWindows()