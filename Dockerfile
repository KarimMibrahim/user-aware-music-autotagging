FROM tensorflow/tensorflow:1.12.0-gpu-py3 

# install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


# clone inside image github repository
RUN git clone https://github.com/KarimMibrahim/user-aware-music-autotagging.git /src_code/repo
ENV PYTHONPATH=$PYTHONPATH:/src_code/repo


# Downgrade to cudatoolkit 9.0 for compatibility reasons
RUN conda install -y -c conda-forge cudatoolkit=9.0

# install requirements
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt


