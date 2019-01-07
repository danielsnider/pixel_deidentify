#!/bin/python
import os
import time
import pydicom
import pytesseract
import cv2
import fuzzywuzzy
import logging
import pandas as pd

from matplotlib import pyplot as plt
from itertools import chain
from datetime import datetime
from dateutil.relativedelta import relativedelta

logging.basicConfig(format='%(asctime)s.%(msecs)d[%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
log = logging.getLogger('main')
log.info('Starting Pixel De-Identify')

def flatten(foo):
    for x in foo:
        if hasattr(x, '__iter__'):
            for y in flatten(x):
                yield y
        else:
            yield x

def orc_image(row, image, search_strings, times, ocr_num=None):
  # Do OCR
  tesseract_config = '--oem %d --psm 3' % ocr_num
  times['ocr%d_before' % ocr_num] = datetime.now()
  detection = pytesseract.image_to_data(image,config=tesseract_config, output_type=pytesseract.Output.DICT)
  times['ocr%d_after' % ocr_num] = datetime.now()
  detected_strings = [text for text in detection['text'] if text] # remove empty strings

  if not detected_strings:
    log.warn('No text found by OCR')
    return
  row['ocr%d_detected_strings' % ocr_num] = detected_strings

  # Transform to uppercase because case might not match from image pixels to metadata
  detected_strings = [x.upper() for x in detected_strings]
  return detected_strings
  

  # Search for each given string
  times['match%d_before' % ocr_num] = datetime.now()
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
  times['match%d_after' % ocr_num] = datetime.now()

# Main code body
df = pd.DataFrame()
times_per_image = []

for subdir, dirs, files in os.walk('/home/dan/Favourite_Images'):
  for file in files:
    times = {}
    filepath =  os.path.join(subdir, file)
    log.info('Processing: %s' % filepath)

    # Load Image
    times['read_image_before'] = datetime.now()
    dicom = pydicom.dcmread(filepath)
    times['read_image_after'] = datetime.now()

    # Preprocess
    times['preprocess_image_before'] = datetime.now()
    img = dicom.pixel_array
    im_resized = cv2.resize(img, dsize=(img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_CUBIC)
    times['preprocess_image_after'] = datetime.now()

    # Build list of PHI to look for: MNR, FirstName, LastName
    PHI = [dicom.PatientID, dicom.PatientName.split('^')] # PatientName ist ypically: FirstName^LastName
    PHI = list(flatten(PHI))
    PHI = [x.upper() for x in PHI] # ensure upper case
    dicom_metadata = {}
    [dicom_metadata.__setitem__(key,dicom.get(key)) for key in dicom.dir() if key not in ['PixelData']]

    # Datastructure for this image
    row = pd.DataFrame([{'FilePath' : filepath,
                         'Modality' : subdir,
                         'PatientName' : dicom.PatientName,
                         'PatientID' : dicom.PatientID,
                         'PHI' : [PHI],
                         'MetaData' : dicom_metadata,
                       }])

    # Find PHI matches in image
    orc_image(row, im_resized, PHI, times, ocr_num=0)
    # orc_image(row, im_resized, PHI, times, ocr_num=1)
    # orc_image(row, im_resized, PHI, times, ocr_num=2)

    df = df.append(row)
    times_per_image.append(times)

    log.info('Finished Processing: %s' % filepath)

    from IPython import embed
    embed() # drop into an IPython session



def time_diff(t_a, t_b):
  from dateutil.relativedelta import relativedelta
  t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
  return t_diff, '{h}h {m}m {s}s {ms}ms'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds, ms=t_diff.microseconds/1000)


## Calculate Timing Statistics
times_df = pd.DataFrame()
# Compute time differences for each time step for each image
for times in times_per_image:
  times_row = pd.DataFrame()
  time_step_names = [key.replace('_before', '') for key in times.keys() if '_before' in key]
  for name in time_step_names:
    diff_datetime, diff_string = time_diff(times['%s_before' % name], times['%s_after' % name])
    times_row[name] = [diff_datetime.microseconds]
  times_df = times_df.append(times_row)


ax = times_df.boxplot()
# ax.set_yscale('log')
plt.legend(loc='best')
plt.show()

# Missing name in header: 1/10
# Missing MNR in header: 1/10
# Name in image doesn't match header: 1/10
# Correctly found entire name in image: 9/10 (True Positive)
# Correctly found part of name in image: 9/10 (True Positive)
# Correctly found no name in image: 9/10 (True Negative)
# Correctly found MNR in image: 9/10 (True Positive)
# Correctly found no MNR in image: 9/10 (True Negative)

# Correctly De-Identified: 50/100




