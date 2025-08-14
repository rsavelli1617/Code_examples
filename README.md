# Code_examples
Repo for showcasing some of my work

<<<<<<< Updated upstream
# SPM_multidatasets.py
In SPM_multidatasets.py, I fused multiple datasets (meteorology, river discharge, sediment concentration, wave buoy, water level) to explore the impact of a dam removal project on the concentration of sediment (SPM = Suspended Particulate Matter) at the river mouth. These datasets all originate from different sources and have different formats. After cleaning them and aligning them around a daily average frequency, I checked for correlated variables that might bias future model predictions. Then I scaled the dataset to normalize all 30 variables. From that, I built a Keras Regressor model and explored the hyperparameter space to select the optimal values. Once the model was trained, I ran the prediction, evaluated it on the test sample, and explored the variables' importance.
=======
# NN_SPM_multidatasets.py
In NN_SPM_multidatasets.py, I fused multiple datasets (meteorology, river discharge, sediment concentration, wave buoy, water level) to explore the impact of a dam removal project on the concentration of sediment at the river mouth. These datasets all originate from different sources and have different formats. After cleaning them and aligning them around a daily average frequency, I checked for correlated variables that might bias future model predictions. Then I scaled the dataset to normalize all 30 variables. From that, I built a Keras Regressor model and explored the hyperparameter space to select the optimal values. Once the model was trained, I ran the prediction, evaluated it on the test sample, and explored the variables' importance.
>>>>>>> Stashed changes

Unzip imported_datasets archive to access datasets from external sources.
