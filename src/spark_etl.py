import botocore
import boto3
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.errors import InvalidDicomError
import PIL
import numpy as np
import pyspark
import io
import sys
import os

IMG_PX_SIZE = 224

S3_BUCKET = 's3BucketName'
S3_INPUT_KEY_PREFIX = 's3FolderName'
S3_INPUT_PATH = 's3a://' + S3_BUCKET + '/' + S3_INPUT_KEY_PREFIX
S3_OUTPUT_KEY_PREFIX = S3_INPUT_KEY_PREFIX + '_processed_images'

INPUT_PATH = S3_INPUT_PATH
OUTPUT_PATH = S3_OUTPUT_KEY_PREFIX


def process_image(rdd_image):
    image_bytes = io.BytesIO(rdd_image.content)
    data = b''
    png_image = io.BytesIO()
    try:
        dicom_conversion_result = "SUCCESS"
        ds = pydicom.dcmread(image_bytes, force=True)
        data = ds.pixel_array
        data = data - np.min(data)
        data = (data * 255).astype(np.uint8)
    except InvalidDicomError as err:
        dicom_conversion_result = err
    except:
        dicom_conversion_result = sys.exc_info()[0]
    try:
        image_conversion_result = "SUCCESS"
        height_px = len(data)
        width_px = len(data[0])
        PIL.Image.frombytes("L", (height_px, width_px), data).resize((IMG_PX_SIZE, IMG_PX_SIZE)).save(
            png_image, 'png')

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
    scan_type = os.path.split(os.path.split(rdd_image['path'])[0])[1]
    patient_dir = os.path.split(os.path.split(os.path.split(rdd_image['path'])[0])[0])[1]
    filename = OUTPUT_PATH + '/' + patient_dir + '/' + scan_type + '/' + os.path.basename(rdd_image['path']) 
    output_key = filename[:len(filename) - 4] + '.png'
    try:
        s3 = boto3.resource('s3')
        put_result = "SUCCESS"
        image_object = s3.Object(S3_BUCKET, output_key)
        png_image.seek(0, 0)
        image_object.put(Body=png_image)
    except botocore.exceptions.ClientError as err:
        put_result = err
    return {'output_key': output_key,
            'image_size': sys.getsizeof(png_image),
            'image_conversion_result': rdd_image['image_conversion_result'],
            'dicom_conversion_result': rdd_image['dicom_conversion_result'],
            'put_result': put_result }


sc = pyspark.sql.SparkSession.builder.getOrCreate()
dicom_images = sc.read.format('binaryFile').option("recursiveFileLookup", "true").load(INPUT_PATH)

print("Dicom image count " + str(dicom_images.count()))
processed_images = dicom_images.rdd.map(process_image)
filenames = processed_images.map(write_processed_image).collect()
for filename in filenames:
    print(filename)
exit(0)