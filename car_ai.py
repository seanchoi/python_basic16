import cv2

# Our Image
img_file = "carimg02.jpg"

# Our pre-trained car classifier
classifier_file="car_detector.xml"

# create opencv image
img = cv2.imread(img_file)

# convert to grayscale (needed for harr cascade) -> black n white speeds up the algorithm a lot faster than color
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # cvtColor - ConvertColor, RGB(BGR) to GRAY

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white) #detecMultiScale -> will detect any size of cars

# Draw Rectangles around the cars


"""
with img_file="carimg01.jpg"

car1 = cars[4]
(x,y,w,h) = car1 # will be [ x=469, y=98, w=40, h=40 ]
cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2) # will mark on color 'img = cv2.imread(img_file)' in color code BGR (Blue, Green, Red) -> (0,0,255) with thickness '2'

print(cars)

will print this 'cars' array(lists)

 top     bottom
 left    right  Width  Height
 (x)     (y)    (x+w)  (y+h)

[
[ 469     98    40     40   ] index[0]
[ 111     114   51     51   ] index[1]
[ 284     83    55     55   ]
[ 181     78    110    110  ]
[ 446     145   154    154  ]
[   7     73    77     77   ]
[ 354     138   81     81   ]
]
"""

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2) 


# Display the image with the cars spotted
cv2.imshow('Python AI Car Detector', img)

# Don't autoclose (Wait here in the code and listen for a key press) / Pause the process
cv2.waitKey()

print("Code Completed")
