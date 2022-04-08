import warnings
# Ignoring the warnings to improve readability of the notebook
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from bigdl.dllib.nn.layer import Model
#from bigdl.dllib.nn.criterion import *
#from bigdl.dllib.optim.optimizer import *
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql.functions import col, udf
#from pyspark.sql.types import *
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType, ArrayType

#from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.image import *
from bigdl.dllib.nnframes import *
from bigdl.dllib.net.net_load import Net
#from bigdl.dllib.keras.layers import *

import streamlit as st

import time
import matplotlib.image as mpimg

#Global variables
#label_texts = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
 #              "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


label_texts = ["bacteria", "Normal", "virus"]

label_map = {k: v for v, k in enumerate(label_texts)}

label_length = len(label_texts)

#Function to return array of labels from csv col label
def text_to_label(text):
    arr = [0.0] * label_length
    for l in text.split("|"):
        if l != "No Finding":
            arr[label_map[l]] = 1.0
    return arr


def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;">{url}</p>', unsafe_allow_html=True)

def print_prediction_output(predDF):
    """
    Display results using streamlit
    """
    st.text("\n\n")  

    #Collect RDD to display
    predictions_list = predDF.collect()[0].prediction
    labelList = predDF.collect()[0].label
    header("{:<25} - {:<25} - {:<15}".format('Finding_Labels', 'Prediction', 'Label'))
    header("{:<25} - {:<25} - {:<15}".format('-'*len('Finding_Labels'), '-'*len('Prediction'), '-'*len('Label')))
    for indx in range(0, len(predictions_list)):
        header("{:<25} - {:<25} - {:<15}".format(label_texts[indx], predictions_list[indx], labelList[indx]))
    st.text("\n\n")
    
    

  
def img_inference(image_path):
    """
    Performs inference and displays target image with result.
    """
    with st.spinner('Wait for it...'):
        time.sleep(3)

    #Model & label file paths
    model_path = "/opt/application/data/model" + "/xray_model_classif.bigdl"
    bin_path = "/opt/application/data/model" + "/xray_model_classif.bin"
    label_path = "/opt/application/data" + "/Data_Entry_2017_v2020.csv"
    
    
    kaggle_path = "/opt/application/data_kaggle" 
    
    test_path = kaggle_path + "/Test"
    train_path = kaggle_path + "/Train"
    val_path = kaggle_path + "/Val"
    
    # Get Spark Content
    sparkConf = create_spark_conf().setAppName("ChestXray_Inference")
    sc = init_nncontext(sparkConf)
    spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
    sqlContext = SQLContext(sc)
    
    # Load the model
    trained_model = Net.load(model_path, bin_path)
    
    # Predict 
    # load the image & labels
    getLabel = udf(lambda x: text_to_label(x), ArrayType(DoubleType()))
    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    
    # 
    ##  NNImageReader : output DataFrame contains a single column named image
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=256, resizeW=256, image_codec=1)\
                    .withColumn("Image_Index", getName(col('image')))
    
    labelDF = spark.read.option('timestampFormat', 'yyyy/MM/dd HH:mm:ss ZZ')\
                .load(label_path, format="csv", sep=",", inferSchema="true", header="true")\
                .select("Image_Index", "Finding_Labels")\
                .withColumn("label", getLabel(col('Finding_Labels')))

    #Join to create 1 DF
    inferDF = imageDF.join(labelDF, on="Image_Index", how="inner")    
    
    # Predict output of when inputdf is passed through model
    transformer = ChainedPreprocessing([
                    RowToImageFeature(),
                    ImageCenterCrop(224, 224),
                    ImageChannelNormalize(123.68, 116.779, 103.939),
                    ImageMatToTensor(),
                    ImageFeatureToTensor()])
    
    #Create classifier model & predict
    classifier_model = NNModel(trained_model, transformer).setFeaturesCol("image")\
                        .setBatchSize(1)
    predictionDF = classifier_model.transform(inferDF)
    
    #Display image
    display_img = mpimg.imread(image_path)
    st.image(display_img, width=300)
    #Display results
    print_prediction_output(predictionDF)