
# Master Thesis, University of Passau
## Topic: Domain-Adaptation
- In this thesis, the development of a generalizable domain adaptation model called "Deep Domain Adaptation Concatenation Network" (DDACN) is discussed, which would help to solve various computer vision tasks. The model is trained on popular visual domain datasets for image classification tasks, and its performance is evaluated compared to other available domain adaptation methods.
-  The "Magnitude based weight pruning" technique is used to perform target feature extractor optimization.

### Description about the code: 
1.  **source_model.py** module defines the source models. **In Progress: 2nd Source model method will be added**
2. **target_model.py** defines the target model.
3. **combined_model.py** defines the merged model architecture, custom evaluation strategy and also various logging paths like tensorboard, csv, model checkpoint, etc.
4.  **config.py** module defines various parameters like set paths, experiment datasets combination ids, etc. **Other configurations may be added in future**
5.  **loss.py** defines the addtional loss methods.
6.  **preprocessing.py** module defines data preprocessing pipeline with various dataset combinations. **In Progress - for Combination: 2, 3, 4**
7. **train_eval.py** is a helper module which defines training and evaluation methods.
8. **evals_helper.py** is a helper module which defines evaluation methods in detail. **May be combined with utils.py in future**
9. **utlis.py** defines various plotting and helper methods. 
10. **main.py** is the runnable script which defines various command line arguments of the experiment. **In progress mode = "eval", script is running for mode="train_test"**
11. **logs/experiments.log** defines the script logs.
12. Models & model weights are stored at **model_data** folder.
13. **requirements.txt** defines the libraries dependency of the experiments.
14. Bash script **run.sh**. **In progress**

You may launch the program with the following command: (have a look at the main.py script for more informations about the attributes)

- Activate tf environment by : **conda activate tf**
-  Monitor experiments.log for log paths.
- Check the tensorboard logs by: tensorboard --lodir "path to  tb logs"


**python main.py  
--combination=1
--source_model=1
--target_model="target_model"
--sample_seed=500
--batch_size=32 
--resize=32
--learning_rate=0.001
--mode="train_test"
--lambda_loss=0.5  
--epochs=50
--save_weights=True
--save_model=True 
--use_multiGPU=False
-- log_file=os.path.join(cn.LOGS_DIR, "experiments.log")**

### Status:

 - Hyperparamter tuning for **lambda loss 2** paramater for one
   experiment, one seed value.
  - Bash script - for running multiple experiments
  - Source Model 2 method
  - Model optimization phase-2 code
