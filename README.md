# eurora_system_power_prediction


This repository holds python code for modeling system level power for the Eurora HPC system, presented in the paper "Predicting system-level power for a hybrid supercomputer" by Alina Sirbu and Ozalp Babaoglu.

The repository depends on the code at https://github.com/alinasirbu/eurora_job_power_prediction , for job power prediction.

The repository contains:

code_big_query.txt - BigQuery commands required to extract features from the data. These commands should be used after having performed the preprocessing in https://github.com/alinasirbu/eurora_job_power_prediction 

build_linear_model.py represents the first step of our analysis. It takes input data containing system and component power measurements (system_data_nodes_down_mics_idle.csv, obtained during preprocessing in https://github.com/alinasirbu/eurora_job_power_prediction) and builds a linear model, trained on one month of data (determined by variable train_month). The model is tested on the data from a different month (test_month). 
 
 component_power.py represents the second step of our analysis. It uses the models obtained in https://github.com/alinasirbu/eurora_job_power_prediction for job power prediction. These are applied to data from October, job power is summed over all users and idle power is added so that we can obtain prediction of power of computing components. 
 
apply_linear_to_predicted.py combines the two steps to obtain the two-layer model. 



