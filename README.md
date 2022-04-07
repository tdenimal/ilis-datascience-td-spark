# ilis-datascience-td-spark
TD Spark pour Master Ingénierie de la santé - Parcours datascience


## STEP 1 - Convert Dicom files to png files using SPARK

### 1.1 - Build docker container

### 1.2 - Explanations 

### 1.3 - Run Spark job

## STEP 2 - Train DL model using BigDL and SPARK

### 2.1 - Build docker container

### 2.2 - Explanations 

### 2.3 - Run API and test

## STEP 3 - API to expose model

### 3.1 - Build docker container

### 3.2 - Explanations 

### 3.3 - Run API and test

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
