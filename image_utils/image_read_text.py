import cv2
from pytesseract import pytesseract
from pytesseract import Output

pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def read_picture(frame, x=10, y=25, w=500, h=75) -> str:
    test = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    words = pytesseract.image_to_string(test)
    return words


def read_picture_canny(frame, x=15, y=590, w=260, h=70) -> str:
    test = cv2.Canny(frame[y:y + h, x:x + w], 150, 200)
    image_data = pytesseract.image_to_data(test, output_type=Output.DICT)
    words = []
    for i, word in enumerate(image_data['text']):
        if len(word) > 3:
            words.append(word)

    return " ".join(words)
