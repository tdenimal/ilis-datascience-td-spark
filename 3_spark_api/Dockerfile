FROM datamechanics/spark:3.2-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3

WORKDIR /opt/application/web

USER root
#Install package dependencies
RUN apt-get update
RUN apt-get -y install gcc g++ libjpeg62-turbo-dev

USER 185
#copy src files including spark jobs, requirements.txt
COPY ./3_spark_api/requirements.txt /opt/application/requirements.txt

#Install pex for packaging dependencies for spark
RUN pip install pex
RUN pip install -r /opt/application/requirements.txt


#Create pex archive
RUN pex --inherit-path=prefer --python=python3.8 -r /opt/application/requirements.txt -o /tmp/py38_spark_etl.pex

#Run spark job
#Env variable PEX_ROOT for base directory for pex
#PEX_INHERIT_PATH : pex inherits of current env, to get pyspark dependencies for ex.
CMD /bin/bash