import cv2
from soil_image_plant_detection import detect_plants

filename = 'soil_image.jpg'

def process(x):
    # Get parameter values
    blur = cv2.getTrackbarPos('Blur', 'image')
    if blur % 2 == 0: blur += 1
    morph = cv2.getTrackbarPos('Morph', 'image')
    iterations = cv2.getTrackbarPos('Iterations', 'image')

    # Process image with parameters
    img = detect_plants(filename, 
          blur=blur, morph=morph, iterations=iterations, 
          debug=True, save=False)

    #Show processed image
    cv2.imshow('image', img)

cv2.namedWindow('image')
cv2.createTrackbar('Blur', 'image', 0, 100, process)
cv2.createTrackbar('Morph', 'image', 1, 100, process)
cv2.createTrackbar('Iterations', 'image', 1, 100, process)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break

cv2.destroyAllWindows()
