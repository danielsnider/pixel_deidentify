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
PHI = list(flatten(PHI))

dicom_metadata = {}
[dicom_metadata.__setitem__(key,dicom.get(key)) for key in dicom.dir() if key not in ['PixelData']]

# Find PHI matches in image

row = pd.DataFrame([{ 'FileName' : filename,
                     'FilePath' : filename,
                     'Modality' : '??',
                     'PatientName' : dicom.PatientName,
                     'PatientID' : dicom.PatientID,
                     'PHI' : [PHI],
                     'MetaData' : dicom_metadata,
                   }])



def ocr_match(row, image, search_strings, ocr_num=None):
  # Do OCR
  tesseract_config = '--oem %d --psm 3' % ocr_num
  detection = pytesseract.image_to_data(image,config=tesseract_config, output_type=pytesseract.Output.DICT)
  detected_strings = [text for text in detection['text'] if text] # remove empty strings
  row['ocr%d_detected_strings' % ocr_num] = detected_strings

  # Transform to uppercase because case might not match from image pixels to metadata
  detected_strings = [x.upper() for x in detected_strings]
  search_strings = [x.upper() for x in search_strings]

  # Search for each given string
  for string_num, search_string in enumerate(search_strings):
    match_text, match_conf = fuzzywuzzy.process.extractOne(search_string, detected_strings, scorer=fuzzywuzzy.fuzz.ratio)
    match_pos = detection['text'].index(match_text) # Get match position in detection table
    print match_pkon['text'][match_pos], detection['conf'][match_pos]
    row['ocr%d_part%d_text' % (ocr_num, string_num)] = detection['text'][match_pos]
    row['ocr%d_part%d_ocr_conf' % (ocr_num, string_num)] = detection['conf'][match_pos]
    row['ocr%d_part%d_match_conf' % (ocr_num, string_num)] = match_conf
    row['ocr%d_part%d_left' % (ocr_num, string_num)] = detection['left'][match_pos]
    row['ocr%d_part%d_top' % (ocr_num, string_num)] = detection['top'][match_pos]
    row['ocr%d_part%d_width' % (ocr_num, string_num)] = detection['width'][match_pos]
    row['ocr%d_part%d_height' % (ocr_num, string_num)] = detection['height'][match_pos]

ocr_match(row, im_resized, PHI, ocr_num=0)
ocr_match(row, im_resized, PHI, ocr_num=1)




