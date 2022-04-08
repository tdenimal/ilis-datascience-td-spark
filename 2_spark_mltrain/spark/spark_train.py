import warnings
# Ignoring the warnings to improve readability of the notebook
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import random
import time
import numpy as np
from math import ceil
#from bigdl.dllib.optim.optimizer import SGD, SequentialSchedule, Warmup, Poly,\Plateau, EveryEpoch, 
#TrainSummary,\ValidationSummary, SeveralIteration, Step
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.storagelevel import StorageLevel
from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.image.imagePreprocessing import *
from bigdl.dllib.feature.common import ChainedPreprocessing
from bigdl.dllib.optim.optimizer import EveryEpoch
from bigdl.dllib.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, Dropout,Sequential,Conv2D,Activation,MaxPooling2D
from bigdl.dllib.keras.metrics import AUC
from bigdl.dllib.keras.optimizers import Adam
from bigdl.dllib.keras.models import Model
from bigdl.dllib.net.net_load import Net
from bigdl.dllib.nnframes import NNEstimator, NNImageReader
from bigdl.dllib.keras.objectives import BinaryCrossEntropy
from pyspark.sql.types import StringType, ArrayType
#from bigdl.dllib.keras.layers import *

#Global variables
label_texts = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

label_map = {k: v for v, k in enumerate(label_texts)}

label_length = len(label_texts)

#Function to return array of labels from csv col label
def text_to_label(text):
    arr = [0.0] * label_length
    for l in text.split("|"):
        if l != "No Finding":
            arr[label_map[l]] = 1.0
    return arr

# Function to load a ResNet50 model
def build_model(label_length):
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(3, 224, 224)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(label_length, activation="sigmoid"))
    return model


#The following function calculates the ROC for disease k. We use ML Pipeline BinaryClassificationEvaluator for this.
def get_auc_for_kth_class(k, df, label_col="label", prediction_col="prediction"):
    get_Kth = udf(lambda a: a[k], DoubleType())
    extracted_df = df.withColumn("kth_label", get_Kth(col(label_col))) \
        .withColumn("kth_prediction", get_Kth(col(prediction_col))) \
        .select('kth_label', 'kth_prediction')
    roc_score = BinaryClassificationEvaluator(rawPredictionCol='kth_prediction',
                                              labelCol='kth_label', metricName="areaUnderROC").evaluate(extracted_df)
    return roc_score


def evaluate_and_plot(testDF):
    #Apply preprocessing to test data
    predictionDF = nnModel.transform(testDF).persist(storageLevel=StorageLevel.DISK_ONLY)

    total_auc = 0.0
    roc_auc_label =dict()
    for i in label_texts:
        roc_score = get_auc_for_kth_class(label_map[i], predictionDF)
        total_auc += roc_score
        print('{:>12} {:>25} {:>5} {:<20}'.format('ROC score for ', i, ' is: ', roc_score))
        roc_auc_label[i]=(roc_score)
    print("Average AUC: ", total_auc / float(label_length))
    



if __name__== "__main__":
    #Variables
    random.seed(1234)
    batch_size = 12 #1024 
    num_epoch = 15

    #    model_path - Path to save the model
    #    image_path - Path to all images
    #    label_path - Path to the label file (Data_Entry_2017.csv) available from NIH
    image_path = "/opt/application/data/output"
    label_path = "/opt/application/data"
    model_path = "/opt/application/data/model" 

    # Get Spark Context
    sparkConf = create_spark_conf().setAppName("Chest X-ray Training")
    sc = init_nncontext(sparkConf)
    spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

    # Make sure the batchsize is a multiple of (Number of executors * Number of cores)
    numexecutors = len(sc._jsc.sc().statusTracker().getExecutorInfos()) - 1
    numcores = int(sc.getConf().get('spark.executor.cores','1'))

    print("Number of Executors = " +str(numexecutors))
    print("Number of Cores = " + str(numcores))
    print("Batch Size = " + str(batch_size))


    #User defined function to get label & filename
    getLabel = udf(lambda x: text_to_label(x), ArrayType(DoubleType()))
    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    
    #Create Dataframe containing image
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=256, resizeW=256, image_codec=1) \
        .withColumn("Image_Index", getName(col('image')))

    #Create Dataframe containg labels from csv
    labelDF = spark.read.load(label_path + "/Data_Entry_2017_v2020.csv", format="csv", sep=",", inferSchema="true", header="true") \
        .select("Image_Index", "Finding_Labels") \
        .withColumn("label", getLabel(col('Finding_Labels'))) 
    
    #Join the 2 DF
    totalDF = imageDF.join(labelDF, on="Image_Index", how="inner")

    

    #(trainingDF, validationDF) = totalDF.randomSplit([0.8, 0.2])
    trainingDF=totalDF
    validationDF=totalDF
    print("Number of training images: ", trainingDF.count())
    print("Number of validation images: ", validationDF.count())
    
    
    # Load the pretrained model
    xray_model = build_model(label_length)
    
    
    
    # Image preprocessing
    transformer = ChainedPreprocessing(
            [RowToImageFeature(), ImageCenterCrop(224, 224), ImageRandomPreprocessing(ImageHFlip(), 0.5),
             ImageRandomPreprocessing(ImageBrightness(0.0, 32.0), 0.5),
             ImageChannelNormalize(123.68, 116.779, 103.939), ImageMatToTensor(), ImageFeatureToTensor()])
    
    
    #Define classifier
    classifier = NNEstimator(xray_model, BinaryCrossEntropy(), transformer) \
            .setBatchSize(batch_size) \
            .setMaxEpoch(num_epoch) \
            .setFeaturesCol("image") \
            .setCachingSample(False) \
            .setValidation(EveryEpoch(), validationDF, [AUC()], batch_size)\
            .setOptimMethod(Adam())
    
    #Train model
    nnModel = classifier.fit(trainingDF)
    
    
    # Evaluate model on validation data
    print("Evaluating the model on validation data:")
    evaluate_and_plot(validationDF)
    
    
    # Save model for inference
    save_path = model_path + '/xray_model_classif'
    nnModel.model.saveModel(save_path + ".bigdl", save_path + ".bin", True)
    print('Model saved at: ', save_path)