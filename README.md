# ilis-datascience-td-spark
TD Spark pour Master Ingénierie de la santé - Parcours datascience


## STEP 1 - Convert Dicom files to png files using SPARK

### 1.1 - Build docker container

There is 2 docker containers

Dev : includes all the necessary to run notebooks, generate graphs etc
Run : only necessary dependencies to run the task

Build dev container:
```bash
docker-compose build sparkdata_dev
```
Build run container:
```bash
docker-compose build sparkdata_run
```

### 1.2 - Explanations 

This task convert DICOM files to PNG images using PYDICOM and spark.

Run dev env to get a jupyter lab using:
```bash
docker-compose run -p 8888:8888  sparkdata_dev
```


### 1.3 - Run Spark job

```bash
docker-compose run sparkdata_run
```


## STEP 2 - Train DL model using BigDL and SPARK

### 2.1 - Build docker container

Dev : includes all the necessary to run notebooks, generate graphs etc
Run : only necessary dependencies to run the task

Build dev container:
```bash
docker-compose build sparkml_dev
```
Build run container:
```bash
docker-compose build sparkml_run
```

### 2.2 - Explanations 

This task train a dummy CNN model using BigDL library on Spark.

Next : Train a model using GCP Dataproc on 100k images (available in a bucket) -
Use a pre-trained model on imagenet - Resnet50 avilable for bigDL.

Put the notebook content into script, and change the Dockerfile for sparkml_run service to run as a job ( Like in STEP 1)

Run dev env to get a jupyter lab using:
```bash
docker-compose run -p 8888:8888  sparkml_dev
```

### 2.3 - Run  train on spark

```bash
docker-compose run  sparkml_run
```


## STEP 3 - API to expose model

### 3.1 - Build docker container

Dev : includes all the necessary to run notebooks, generate graphs etc
Run : only necessary dependencies to run the task

Build dev container:
```bash
docker-compose build sparkapi_dev
```


Build run container:
```bash
docker-compose build sparkapi_run
```

### 3.2 - Explanations 

Use the previously trained model and streamlit to create an API to serve predictions on test images. 

### 3.3 - Run API and test

```bash
docker-compose run -p 8080:8080 sparkapi_run
```

## ANNEXE
Retrieve the dataset containing Xrays from Kaggle:


### Install kaggle API & configuration
Follow the instructions in [https://github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api)

After kaggle API installation, retrieve the kaggle.json file from your profile in kaggle :

Connect to kaggle and get to account part:

![kaggle_account.png](../assets/kaggle_account.png?raw=true)

Get kaggle.json using `Create new API Token` button :
![get_kaggle_json.png](../assets/get_kaggle_json.png?raw=true)

Then copy it in the location ~/.kaggle/kaggle.json. You can get easy access to the directory using this method in WSL:


```bash
cd ~/.kaggle && explorer.exe .
```

These commands should open a windows explorer of the WSL directory, just copy/paste the downloaded kaggle.json in it.

You can also change the permission on the kaggle.json file :

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Retrieve the dataset and move it to the data/ directory in the repository
Replace <path_to_repo> with the directory containing the ilis-datascience-td-spark respository

```bash
kaggle datasets download -d ahmedhaytham/chest-xray-images-pneumonia-with-new-class -p  <path_to_repo>/ilis-datascience-td-spark/data
```


Should result in following : 
```bash
ls data/
chest-xray-pneumonia.zip
```

Unzip file in data/ directory
```bash
ls data/
chest-xray-pneumonia.zip
```
