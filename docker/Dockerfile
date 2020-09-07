FROM tensorflow/tensorflow:1.12.0-gpu-py3 
#FROM python:3.6
#FROM nvidia/cuda:9.1-runtime-ubuntu16.04


# install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


# clone inside image github repository
RUN git clone https://github.com/KarimMibrahim/user-aware-music-autotagging.git /src_code/repo
ENV PYTHONPATH=$PYTHONPATH:/src_code/repo

# install anaconda
#ENV PATH /opt/conda/bin:$PATH
#COPY install_miniconda.sh .
#RUN bash ./install_miniconda.sh && rm install_miniconda.sh

# Downgrade to cudatoolkit 9.0 for compatibility reasons
#RUN conda install -y -c conda-forge cudatoolkit=9.0

# install requirements
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt

WORKDIR /src_code/repo
ENTRYPOINT ["bash"]

