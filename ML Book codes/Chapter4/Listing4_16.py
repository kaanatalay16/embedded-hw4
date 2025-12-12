import cv2

#read the color image
img = cv2.imread('mandrill.tif')

#display the color image
cv2.imshow('Color Image',img)
cv2.waitKey(0)

cv2.destroyAllWindows()

#write the image to a file
#cv2.imwrite('mandrill_store.tif',img)