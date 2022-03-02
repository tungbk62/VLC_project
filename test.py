import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

cap = cv2.VideoCapture('D:/do_an_3/input_output/1div3200_1K_40m.mp4')

cap.set(cv2.CAP_PROP_POS_FRAMES,10)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
  # Capture frame-by-frame
ret, frame = cap.read()
if ret == True:
    # Display the resulting frame
    cv2.imwrite("D:/do_an_3/input_output/input_image.png", frame)
originalImage = frame
#originalImage = cv2.imread("input_image_0123_500hz.png")

grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
cv2.imshow("cdac", grayImage)

# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY)
# cropImage = blackAndWhiteImage[0:932, 702:722]
# sizeOfImage = cropImage.shape
height = grayImage.shape[0]
width = grayImage.shape[1]
print(height)
print(width)



#tim diem co gia tri lon hon nguong
def findMinimumEnclosingCircle(grayImage, threshMaxValue):
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    maxPoints = []
    for i in range(width):
        for j in range(height):
            if(grayImage[j, i] >= threshMaxValue):
                maxPoints.append((j, i))
    print(len(maxPoints))

    (x,y), radius = cv2.minEnclosingCircle(np.float32(maxPoints))

    return (int(y), int(x)), int(radius)


def calculateArrayA(grayImage, increaseCircleValue, x, y, radius):
    height = grayImage.shape[0]
    width = grayImage.shape[1]

    grayScale = []
    for i in range(width):
        sum = 0
        mauSo = 0
        for j in range(height):
            if(int(math.sqrt((x - i)*(x - i) + (y - j)*(y - j))) > int(radius*increaseCircleValue)):
                mauSo = mauSo + 1
                sum = sum + grayImage[j, i]
        sum = sum/mauSo
        grayScale.append(sum)
    return grayScale


def calculateThresholdArray(grayScale):

    leftGrayScale = np.roll(grayScale, -1)
    rightGrayScale = np.roll(grayScale, 1)

    localMaxMinPoints = []

    length = len(grayScale)
    for i in range(length):
        value = grayScale[i]
        if(value > leftGrayScale[i] and value > rightGrayScale[i]):
            localMaxMinPoints.append(value)
        elif(value < leftGrayScale[i] and value < rightGrayScale[i]):
            localMaxMinPoints.append(value)
        else:
            localMaxMinPoints.append(0)


    copy_grayScale = np.sort(grayScale, axis=-1, kind='quicksort', order=None)
    thirdLargest = copy_grayScale[len(copy_grayScale) -3 ]
    TcValue = (thirdLargest/2) - 35

    for i in range(length):
        if(localMaxMinPoints[i] != 0 and TcValue > localMaxMinPoints[i]):
            localMaxMinPoints[i] = TcValue

    threshArray = []

    for i in range(length):
        firstValue = 0
        averageValue = 0
        mark = 0
        if(i == 0 and localMaxMinPoints[i] == 0):
            for j in range(i+1, length):
                if(localMaxMinPoints[j] != 0):
                    for k in range(j+1, length):
                        if(localMaxMinPoints[k] != 0 and abs(localMaxMinPoints[k] - localMaxMinPoints[j]) > 5):
                            averageValue = (localMaxMinPoints[j] + localMaxMinPoints[k])/2
                            firstValue = averageValue
                            break
                    break
        elif(i == 0 and localMaxMinPoints[i] != 0):
            for j in range(i+1, length):
                if(localMaxMinPoints[j] != 0 and abs(localMaxMinPoints[i] - localMaxMinPoints[j]) > 5):
                    averageValue = (localMaxMinPoints[i] + localMaxMinPoints[j])/2
                    firstValue = averageValue
                    break
        if(localMaxMinPoints[i] != 0):
            for j in range(i + 1, length):
                if(localMaxMinPoints[j] != 0):
                    mark = j - i
                    averageValue = (localMaxMinPoints[i] + localMaxMinPoints[j])/2
                    if(abs(localMaxMinPoints[i] - localMaxMinPoints[j]) > 5):
                        for k in range(mark):
                            threshArray.append(averageValue)
                    else:
                        beforeValue = 0
                        try:
                            beforeValue = threshArray[len(threshArray) -1]
                        except:
                            beforeValue = firstValue
                        for k in range(mark):
                            threshArray.append(beforeValue)
                    break

    if(len(threshArray) < length):
        count = length - len(threshArray)
        lastValue = threshArray[len(threshArray) - 1]
        for i in range(count):
            threshArray.append(lastValue)
    
    return threshArray
    

def calculateNewGrayScale(grayScale, threshArray):
    length = len(grayScale)
    newGrayScale = []
    for i in range(length):
        if(grayScale[i] >= threshArray[i]):
            newGrayScale.append(255)
        else:
            newGrayScale.append(0)

    return newGrayScale

def lengthStripeArray(newGrayScale):
    copyNewGrayScale = newGrayScale[:]
    copyNewGrayScale.append(-1)
    length = len(copyNewGrayScale)
    bitArray = []
    lengthStripeArray = []
    startAt = 0
    for i in range(length):
        if(startAt == length - 1):
            break
        if(i < startAt):
            continue
        lengthStripe = 1
        mark = copyNewGrayScale[i]
        for j in range( i + 1, length):
            if(copyNewGrayScale[j] == mark):
                lengthStripe = lengthStripe + 1
            else:
                startAt = j
                if(mark == 255):
                    bitArray.append(1)
                else:
                    bitArray.append(0)
                lengthStripeArray.append(lengthStripe)
                break
    return bitArray[::-1], lengthStripeArray[::-1]
        
def convertPixelToBit(bitArray, lengthStripeArray):
    bitInformationArray = []
    length = len(bitArray)
    smallest_0 = [99999, 99999]
    smallest_1 = [99999, 99999]
    for i in range(length):
        if(bitArray[i] == 0):
            if(lengthStripeArray[i] <= smallest_0[1]):
                smallest_0[1] = lengthStripeArray[i]
            elif(lengthStripeArray[i] < smallest_0[0]):
                smallest_0[0] = lengthStripeArray[i]
        else:
            if(lengthStripeArray[i] <= smallest_1[1]):
                smallest_1[1] = lengthStripeArray[i]
            elif(lengthStripeArray[i] < smallest_1[0]):
                smallest_1[0] = lengthStripeArray[i]
    
    if(smallest_0[0] == 99999):
        smallest_0[0] = smallest_0[1]
    if(smallest_1[0] == 99999):
        smallest_1[0] = smallest_1[1]

    for i in range(length):
        if(bitArray[i] == 0):
            if(float(lengthStripeArray[i])/smallest_0[0] <= 1.5):
                bitInformationArray.append(0)
            else:
                bitInformationArray.append(0)
                bitInformationArray.append(0)
        else:
            if(float(lengthStripeArray[i])/smallest_1[0] <= 1.5):
                bitInformationArray.append(1)
            else:
                bitInformationArray.append(1)
                bitInformationArray.append(1)
    return bitInformationArray

def findHeader(bitInformationArray, header):
    lengthHeader = len(header)
    lengthBitInformationArray = len(bitInformationArray)
    headerPosition = -1
    for i in range(lengthBitInformationArray):
        if(i + lengthHeader - 1 > lengthBitInformationArray - 1):
            break
        for j in range(lengthHeader):
            if(header[j] != bitInformationArray[i + j]):
                break
            if(header[j] == bitInformationArray[i + j] and j == lengthHeader - 1):
                headerPosition = i
        if(headerPosition != -1):
            break
    return headerPosition

def decodeInformation(bitInformationArray, header, headerPosition):
    lengthHeader = len(header)
    lengthBitInformationArray = len(bitInformationArray)
    return

def removeHeadAndTail(bit, bitLength, number):
    for i in range(number):
        bit.pop(0)
        bit.pop(len(bit) - 1)
        bitLength.pop(0)
        bitLength.pop(len(bitLength) - 1)

increaseValue = 1.2
(x, y), radius = findMinimumEnclosingCircle(grayImage, 220)
center = (x, y)
cv2.circle(grayImage, center, int(radius*increaseValue),(0,255,0),2)

grayScale = calculateArrayA(grayImage, increaseValue, x, y, radius)
threshArray = calculateThresholdArray(grayScale)
newGrayScale = calculateNewGrayScale(grayScale, threshArray)
print(newGrayScale)

bit, bitLength = lengthStripeArray(newGrayScale)
print("bit array")
print(bit)
print("length array")
print(bitLength)

lengthOfBitArray = len(bit)

# bit.pop(0)
# bit.pop(0)
# bit.pop(lengthOfBitArray - 3)
# bit.pop(lengthOfBitArray - 4)
# bitLength.pop(0)
# bitLength.pop(0)
# bitLength.pop(lengthOfBitArray - 3)
# bitLength.pop(lengthOfBitArray - 4)
removeHeadAndTail(bit, bitLength, 2)

print("\nnew bit array")
print(bit)
print(" new length array")
print(bitLength)

resultArray = convertPixelToBit(bit, bitLength)
print("result")
print(resultArray)
print(len(resultArray))

plt.plot(range(width), newGrayScale)
plt.plot(range(width), threshArray)
plt.xlabel("pixel")
plt.ylabel("grayscale")
plt.show()

cv2.imwrite("D:/do_an_3/input_output/output_image.png", grayImage)

cv2.waitKey(0)
cv2.destroyAllWindows()