import cv2

# Our Video
video = cv2.VideoCapture("test_video.mp4")
video2 = cv2.VideoCapture("pedestrians.mp4")

# Our pre-trained car and pedestrian classifier
car_tracker_file = "car_detector.xml"
pedestrian_tracker_file = "pedestrian_detector.xml"

# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


# run forever until car stops
while True:

    # read the current frame (1 frame from the video per one loop)
    (read_successful, frame) = video2.read() # everytime call this read 1 frame and save in 'frame' 

    # Safe Coding,
    if read_successful: # when end of the video this will be False then while loop breaks
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrian
    cars = car_tracker.detectMultiScale(grayscaled_frame) #detecMultiScale -> will detect any size of cars
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectagles on cars realtime
    for (x, y, w, h) in cars:
        # cv2.rectangle(frame, (x+2,y+2), (x+w, y+h), (255,0,0), 2)  
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)  

    # Draw rectagles on pedestrians realtime
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2) 


    # Display the image with the cars spotted
    cv2.imshow('Python AI Car Detector', frame)

    # Don't autoclose (Wait here in the code and listen for a key press) / Pause the process
    key = cv2.waitKey(1) # 1 = 1ms/ 1000 = 1second 

    # Stop if Q key is pressed
    if key == 81 or key == 113: # 'key' is automatically set from waitKey() 
        break

print("Code Completed")
