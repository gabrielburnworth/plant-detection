import cv2
from soil_image_plant_detection import detect_plants

filename = 'soil_image.jpg'
window = 'Plant Detection'

def process(x):
    # Get parameter values
    blur = cv2.getTrackbarPos('Blur', window)
    if blur % 2 == 0: blur += 1
    morph = cv2.getTrackbarPos('Morph', window)
    iterations = cv2.getTrackbarPos('Iterations', window)

    # Process image with parameters
    img = detect_plants(filename, 
          blur=blur, morph=morph, iterations=iterations, 
          debug=True, save=False)

    #Show processed image
    cv2.imshow(window, img)

cv2.namedWindow(window)
cv2.createTrackbar('Blur', window, 0, 100, process)
cv2.createTrackbar('Morph', window, 1, 100, process)
cv2.createTrackbar('Iterations', window, 1, 100, process)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break

cv2.destroyAllWindows()
