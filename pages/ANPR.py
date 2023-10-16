import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import imutils
import easyocr

uploaded_file = st.file_uploader("Upload Image")

col1, col2 = st.columns(2, gap="medium")


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    cv2.imwrite('img/color.jpg', cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    img = cv2.imread('img/color.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(gray)

    bfilter = cv2.bilateralFilter(img, 11, 17, 17) #Noise reduction

    
    t1 = st.slider('Treshold 1 Amount', 1, 300, 10)
    t2 = st.slider('Treshold 2 Amount', 1, 300, 10)
    edged = cv2.Canny(bfilter, t1, t2) #Edge detection

    st.image(edged)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    epsilon = st.slider('Epsilon Amount', 1, 150, 10)


    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            location = approx
            break
    
    if location is None:
        st.write("Plat tidak terdeteksi")

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    st.image(new_image)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    st.image(cropped_image)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    text = result[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)

    st.image(res)

    recognized_text = ""
    for result1 in result:
        text = str(result1[1])
        recognized_text += text + ' '

    st.write(recognized_text)
