# Agen-Motion-Forecasting_Deep-Learning-Project

Each folder contains the execution of a model with following file structure:  
|  
|---|--agent_motion_config.yaml  
    |--agent_motion_prediction.py  
    |--setup_local_data.sh  
    |--run_train_sbatch.sh  
    
    
    
agent_motion_config.yaml- The file defines the parameters like model architechture, history_frames, future_frames and image size 

agent_motion_prediction.py- The python script to be executed containing the model scripts and training script it saves the trained model and the training loss to be plotted

setup_local_data.sh- This bash script is used to set local data path and extract the downloaded data to a folder

run_train_sbatch.sh -  This bash script is used to run the python script on the WPI ace cluster
