import cv2
import numpy as np
import time

net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
classes = []
with open("classes.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


# print(classes)
layer_names = net.getLayerNames()
# print(layer_names)
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

cap=cv2.VideoCapture('test_video/1.mp4')
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0

_,frame= cap.read()
h, w, _ = frame.shape

size = (h, w)

result = cv2.VideoWriter('output/video.mp4', 
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        10, size)

while True:
    _,frame= cap.read() # 
    frame_id+=1

    height,width,channels = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

    #detecting objects
    # blob = cv2.dnn.blobFromImage(frame, 0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

        
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])


    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # print(scores)
            if confidence > 0.01:
                print(confidence)
                #onject detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
            

    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)

    cv2.imshow("Image", cv2.resize(frame, (800, 600)))
    

    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter('output/video.mp4', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)

    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame

    result.write(frame)
    if key == 27: #esc key stops the process
        break;
    
cap.release()    
result.release()
cv2.destroyAllWindows()