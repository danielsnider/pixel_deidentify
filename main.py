#!/bin/python
import pydicom
import pytesseract
import cv2
import fuzzywuzzy
import pandas as pd

from matplotlib import pyplot as plt
from itertools import chain

def flatten(foo):
    for x in foo:
        if hasattr(x, '__iter__'):
            for y in flatten(x):
                yield y
        else:
            yield x

# Load Image
filename = "94862596.dcm"
dicom = pydicom.dcmread(filename)
img = dicom.pixel_array
# im_cropped = img[:80,280:]
im_resized = cv2.resize(img, dsize=(img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_CUBIC)

# Build list of PHI to look for: MNR, FirstName, LastName
PHI = [dicom.PatientID, dicom.PatientName.split('^')] # PatientName ist ypically: FirstName^LastName
PHI = [item for sublist in PHI for item in sublist] # Flatten list
list(flatten(PHI))

# Find PHI matches in image
ocr_match(im_resized,'') 

def ocr_match(image, search_strings, tesseract_config=None):
  detection = pytesseract.image_to_data(image,config=tesseract_config, output_type=pytesseract.Output.DICT)
  detected_strings = [text for text in detection['text'] if text] # remove empty strings
  for search_string in search_strings:
    match_text, match_conf = fuzzywuzzy.process.extractOne(search_string, detected_strings, scorer=fuzzywuzzy.fuzz.ratio)
    match_pos = detection['text'].index(match_text)
    match_left = detection['left'][match_pos]
    match_top = detection['top'][match_pos]
    match_width = detection['width'][match_pos]
    match_height = detection['height'][match_pos]

df2 = pd.DataFrame({ 'A' : 1.,
                        'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo' })

image name
patient name
num_name_parts
MNR
found name
ocr1_name_part1_text
ocr1_name_part1_conf
ocr1_name_part1_left
ocr1_name_part1_top
ocr1_name_part1_width
ocr1_name_part1_height
ocr1_name_partN_*
orc1_

# fuzzywuzzy.fuzz.ratio
# fuzzywuzzy.fuzz.partial_ratio
# fuzzywuzzy.fuzz.token_sort_ratio
# fuzzywuzzy.fuzz.partial_token_sort_ratio
# fuzzywuzzy.fuzz.token_set_ratio
# fuzzywuzzy.fuzz.partial_token_set_ratio
# fuzzywuzzy.fuzz.WRatio


ocr_match(tesseract_config='--oem 0 --psm 3')

text = pytesseract.image_to_string(im_resized,config='--oem 0 --psm 3')
pytesseract.image_to_string(im_resized,config='--oem 1 --psm 3')
pytesseract.image_to_string(im_resized,config='--oem 2 --psm 3')

from fuzzywuzzy import fuzz


text = pytesseract.image_to_string(data)
print text

plt.imshow(data, cmap='gray')
plt.show()

plt.imshow(im_cropped, cmap='gray')
plt.show()

