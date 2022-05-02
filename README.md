# Agen-Motion-Forecasting_Deep-Learning-Project

## Folder structure for each model
Each folder contains the execution of a model with following file structure:  
|  
|--agent_motion_config.yaml  
|--agent_motion_prediction.py  
|--setup_local_data.sh  
|--run_train_sbatch.sh  
    
    
agent_motion_config.yaml- The file defines the parameters like model architechture, history_frames, future_frames and image size 

agent_motion_prediction.py- The python script to be executed containing the model scripts and training script it saves the trained model and the training loss to be plotted

setup_local_data.sh- This bash script is used to set local data path and extract the downloaded data to a folder

run_train_sbatch.sh -  This bash script is used to run the python script on the WPI ace cluster

The dataset can be downloaded from: https://woven-planet.github.io/l5kit/dataset.html
 
The bash script contains sample dataset of (55MB) which can be automatically downloaded and extracted by the script itself.

The models are saved to a folder named Out with (.pth) format and the training loss is saved in (.csv) fie within the folder.

<!-- ![Training loss](assets/training_loss for all networks normalized.png?raw=true "Training loss for all networks normalized" ) -->
<img src="https://github.com/tanmay-dhasade/Agen-Motion-Forecasting_Deep-Learning-Project/tree/main/assets/training_loss for all networks normalized.png" alt="Alt text" title="Training loss for model">
