import pydicom
from pydicom.errors import InvalidDicomError
import PIL
import numpy as np
import pyspark
import io
import sys
import os
from pathlib import Path

IMG_PX_SIZE = 512

INPUT_PATH = '/opt/application/data/input'
OUTPUT_PATH = '/opt/application/data/output'



def process_image(rdd_image):
    image_bytes = io.BytesIO(rdd_image.content)
    data = b''
    png_image = io.BytesIO()
    try:
        dicom_conversion_result = "SUCCESS"
        ds = pydicom.dcmread(image_bytes, force=True)
        data = ds.pixel_array
        
        #Data normalization 0 - 255
        data = data - np.min(data)
        data = (data * 255).astype(np.uint8)
    except InvalidDicomError as err:
        dicom_conversion_result = err
    except:
        #gets the type of the exception being handled 
        dicom_conversion_result = sys.exc_info()[0]
    try:
        image_conversion_result = "SUCCESS"
        height_px = len(data)
        width_px = len(data[0])
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        # L = 8 pixels = black and white
        PIL.Image.frombytes("L", (height_px, width_px), data) \
                 .resize((IMG_PX_SIZE, IMG_PX_SIZE)) \
                 .save(png_image, 'png')
    except OSError:
        image_conversion_result = "FAIL"
    except IndexError:
        image_conversion_result = "FAIL"
        
    
    image_png_rdd_element = {"path": rdd_image.path,
                             "modificationTime": rdd_image.modificationTime,
                             'image_conversion_result': image_conversion_result,
                             'dicom_conversion_result': dicom_conversion_result,
                             "content": png_image}
    return image_png_rdd_element


def write_processed_image(rdd_image):
    png_image = rdd_image['content']
    filename = OUTPUT_PATH + '/' + (Path(os.path.basename(rdd_image['path'])).stem) + '.png'
    try:
        put_result = "SUCCESS"
        # make file
        newFile = open(filename, "wb")
        # write png file
        newFile.write(png_image.getbuffer().tobytes())
    except:
        #gets the type of the exception being handled 
        put_result = sys.exc_info()[0]
    return {'output_key': filename,
            'image_size': sys.getsizeof(png_image),
            'image_conversion_result': rdd_image['image_conversion_result'],
            'dicom_conversion_result': rdd_image['dicom_conversion_result'],
            'put_result': put_result }


if __name__== "__main__":
    sc = pyspark.sql.SparkSession.builder.getOrCreate()

    #RDD of dicom images, read from GCP Bucket
    dicom_images = sc.read.format('binaryFile').option("recursiveFileLookup", "true").load(INPUT_PATH)
    print("Dicom image count " + str(dicom_images.count()))


    #Convert to RDD of png images (resized)
    processed_images = dicom_images.rdd.map(process_image)

    #Write back the RDD to bucket
    filenames = processed_images.map(write_processed_image).collect()
    for filename in filenames:
        print(filename)
    exit(0)