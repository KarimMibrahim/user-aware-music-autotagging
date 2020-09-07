FROM tensorflow/tensorflow:1.12.0-gpu-py3 

# install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


# clone inside image github repository
RUN git clone https://github.com/KarimMibrahim/user-aware-music-autotagging.git /src_code/repo
ENV PYTHONPATH=$PYTHONPATH:/src_code/repo

# download the dataset
RUN apt-get install wget 
RUN wget -P /src_code/repo/GroundTruth/ https://zenodo.org/record/3961560/files/test_set.csv
RUN wget -P /src_code/repo/GroundTruth/ https://zenodo.org/record/3961560/files/train_set.csv
RUN wget -P /src_code/repo/GroundTruth/ https://zenodo.org/record/3961560/files/user_embeddings.csv
RUN wget -P /src_code/repo/GroundTruth/ https://zenodo.org/record/3961560/files/validation_set.csv

# install requirements
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt

WORKDIR /src_code/repo
ENTRYPOINT ["bash"]

