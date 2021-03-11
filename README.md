

# Master Thesis, University of Passau
### Topic: Domain-Adaptation
- In this thesis, the development of a generalizable domain adaptation model technique is discussed, which could help to solve various computer vision tasks. The model is trained on popular visual domain datasets for image classification tasks, and it's performance is evaluated compared to other available domain adaptation methods.
-  The "Magnitude based weight pruning" technique is used to perform target feature extractor optimization.

## Description about the code: 
1.  **models.py** module defines the source & target models. **Xception Network & Top layers**
2.  **config.py** module defines various parameters like set paths, experiment datasets combination ids, etc. **Other configurations may be added in future**
3.  **loss.py** defines the addtional loss methods.
4.  **preprocessing.py** module defines data preprocessing pipeline with various dataset combinations including Data augmentation. 
5. **train_test.py** is a helper module which defines training and evaluation methods.
6. **evals_helper.py** is a helper module which defines evaluation methods in detail. 
7. **utlis.py** defines various plotting, helper methods and various logging paths like tensorboard, csv, model checkpoint, etc.
8. **main.py** is the runnable script which defines various command line arguments of the experiment. **In progress mode = "eval", script is running for mode="train_test"**
9. *logs/CombinationID_SourceModel_LambdaLossValue/experiments.log* defines the run logs.
10. Models & model weights are stored at **model_data** folder.
11. **requirements.txt** defines the libraries dependency of the experiments.
12. Bash script  to run multiple expleriments. **run.sh**.
13. **evaluation** folder shows the loss/accuracy plots, also can be viewed in Tensorboards.

You may launch the program with the following command: (have a look at the main.py script for more informations about the attributes)

- Activate tf environment by : **conda activate tf**
-  Monitor **experiments.log** for log paths and script progress.
- Check the tensorboard logs by: tensorboard --lodir "path to  tb logs"
- Check **training_logs.csv** for model training logs. (path: *logs/CombinationID_SourceModel_LambdaLossValue/DateTimeStampValue*)

**python main.py  
--combination="Amazon_to_Webcam"
--architecture="Xception"
--batch_size=16
--resize=299
--learning_rate=0.0001
--mode="train_test"
--lambda_loss=0.5  
--epochs=50
--input_shape=(299,299,3)
--output_classes=31
--loss_function="CORAL"
--augment=True
--prune=True
--prune_val=0.30
--technique=True
--save_weights=True
--save_model=True 
--use_multiGPU=False**
