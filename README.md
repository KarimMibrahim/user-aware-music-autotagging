# Should we consider the users in contextual music auto-tagging models?
This is the code for the ISMIR 2020 paper 'Should we consider the users in contextual music auto-tagging models?'. If you use the code in your research, please cite the paper as:

> Karim M. Ibrahim, Elena V. Epure, Geoffroy Peeters, and Gaël Richard. 2020. Should we consider the users in contextual music auto-tagging models? In Proceedings of *the 21st International Society for Music Information Retrieval Conference* (ISMIR '20). , Montréal, Canada.

## Instructions

We recommend using [Docker](https://www.docker.com/) for reproducing the results of the experiments. 

Clone the repository then follow these instructions.

1. To build the docker image run:
```
docker build -t user-aware-image docker
```
This will automatically clone the repository and download the dataset inside the docker container

2. Run the container with: 
```
nvidia-docker run -ti --rm --memory=20g --name=user_aware_container -p 8888:8888 user-aware-image
```
Note: you might need to adjust the memory or port according to your machine. 

3. Download the audio previews inside the docker using the [Deezer API](https://developers.deezer.com/api). Then compute the melspectrograms (recommended using librosa) in the direcory "/src_code/repo/spectrograms/" with the following parameters: 
```
"n_fft": 1024,
"hop_length": 1024,
"n_mels": 96
```
Note: we plan on releasing a script to automatically complete this step in the future.

4. Run the labels pre-processing script which creates the multi-label sets from the downloaded dataset: 
```
python preprocess_dataset_labels.py
```

5. For training the audio model run: 
```
python audio_based_model.py
```
The model evaluation results exist in the output directory in 'src_code/repo/experiment_results/audio_system_multilabel/[EXPERIMENT_DATE]'


6. For training the user+audio model run: 
```
python user_aware_model.py

```
The model evaluation results exist in the output directory in 'src_code/repo/experiment_results/user_aware_system/[EXPERIMENT_DATE]'

7. Run the pretrained multi-label audio model on the single label groundtruth case
```
python generate_single_label_audio_output.py
```
**Note:** must edit the diretory to match the trained model in your machine 'marked with [TODO]'

8. For displaying the results, first start the jupyter notebook with: 
```
jupyter notebook --allow-root --ip=$(awk 'END{print $1}' /etc/hosts) --no-browser --NotebookApp.token= &
```
Then access the notebook through the designated port and run all code blocks in order of the following notebook
- Extra_evaluation_protocols.ipynb
This will compute and display the results for the audio model (MO-MG MO-SG and SO-SG), as well as the user model results. 

9. For the user-satisfaction evaluation, run the code in the notebook 'Per-user evaluation.ipynb'.

## Items 
This repository contains the following item: 
- '**audio_based_model.py**' the script for training the audio model.
- '**user_aware_model.py**' the script for training the user-aware model.
- '**Preprocess_dataset_labels.py**' the script to generate the multi-label dataset for the audio model.
- '**utilities.py**' script containing utility functions used for both models.
- '**generate_single_label_audio_output.py**' rerun the pretrained audio model on the single-label dataset.
- '**Extra_evaluation_protocols.ipynb**' generates the evaluation results for the MO-SG and SO-SG protocols.
- '**Per-user evaluation.ipynb**' generates the user-satisfaction evaluation results. 
- '**requirements.txt**' contains the required packages to run the code. Only needed in case docker is not used.   


The repository contains one directories: 
- '**Docker**' contains the script for building the docker image and installing the requirements.


## Acknowledgment
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.
