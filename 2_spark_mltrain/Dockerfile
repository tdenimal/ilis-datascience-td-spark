FROM datamechanics/spark:3.2-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3
ENV SPARK_DRIVER_MEMORY=4g

WORKDIR /opt/application/spark

USER root
#Install package dependencies
RUN apt-get update
RUN apt-get -y install gcc g++ libjpeg62-turbo-dev libffi-dev
RUN pip install --upgrade pip

USER 185
#copy src files including spark jobs, requirements.txt
COPY ./2_spark_mltrain/requirements.txt /opt/application/requirements.txt
COPY ./2_spark_mltrain/spark /opt/application/spark


#Install pex for packaging dependencies for spark
RUN pip install pex
RUN pip install -r /opt/application/requirements.txt


#Create pex archive
RUN pex --inherit-path=prefer --python=python3.8 -r /opt/application/requirements.txt -o /tmp/py38_spark_etl.pex

#Run spark job
#Env variable PEX_ROOT for base directory for pex
#PEX_INHERIT_PATH : pex inherits of current env, to get pyspark dependencies for ex.

#Workaround for missing jar - dontknow why jar variables in spark-submit are not ok ?
USER root
RUN cp /opt/conda/lib/python3.8/site-packages/bigdl/share/dllib/lib/bigdl-dllib-spark_3.1.2-2.0.0-jar-with-dependencies.jar /opt/spark/jars


USER 185
CMD /opt/spark/bin/spark-submit  \
--conf spark.executorEnv.PEX_ROOT=/tmp \
--conf spark.yarn.appMasterEnv.PEX_ROOT=/tmp \
--conf spark.executorEnv.PEX_INHERIT_PATH=prefer \
--conf spark.yarn.appMasterEnv.PEX_INHERIT_PATH=prefer \
--conf spark.pyspark.driver.python=/tmp/py38_spark_etl.pex \
--conf spark.pyspark.python=/tmp/py38_spark_etl.pex \
--jars /opt/conda/lib/python3.8/site-packages/bigdl/share/dllib/lib \
--conf spark.driver.extraClassPath=bigdl-dllib-spark_3.1.2-2.0.0-jar-with-dependencies.jar \
--conf spark.executor.extraClassPath=bigdl-dllib-spark_3.1.2-2.0.0-jar-with-dependencies.jar \
--files /tmp/py38_spark_etl.pex /opt/application/spark/spark_train.py