


# Master Thesis, University of Passau
### Topic: Domain-Adaptation
- In this thesis, the development of a discrepancy based domain adaptation models named MBM (Modified Baseline Network - an extension of Deep CORAL) & CDAN (Custom Domain Adaptive Network) are discussed. The models are trained on popular benchmarked visual recognition domain datasets like Office-31, GTSRB and Synthetic Signs for image classification tasks, and their performances are evaluated compared to other available domain adaptation methods.
-  The "Magnitude based weight pruning" with *Constant Sparsity* approach is used to perform target feature extractor optimization.

## Description about the code: 
1.  **[models.py](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/main/modules/models.py)** module defines the source & target models. **Xception Network & Top layers**
2.  **[config.py](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/main/modules/config.py)** module defines various parameters like set paths, domain adaptation scenarios, backbone model selection, etc. 
3.  **[loss.py](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/main/modules/loss.py)** defines the domain alignment loss functions. **Deep CORAL loss, KL Divergence, etc.**
4.  **[preprocessing.py](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/main/modules/preprocessing.py)** module defines data preprocessing pipeline with various experimental scenarios including Data augmentation methods. 
5. **[train_test.py](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/main/modules/train_test.py)** is a helper module which defines training and evaluation methods, including Evaluatiion metrics like Confusion Matrix, etc.
6. **[utlis.py](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/main/modules/utils.py)** defines various plotting, helper methods and various logging paths like tensorboard, csv, model checkpoint, etc.
7. **[main.py](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/main/main.py)** is the runnable script which defines various command line arguments of the experiment. **In progress mode = "eval", script is running for mode="train_test"**
8. *logs/CombinationID_SourceModel_LambdaLossValue/experiments.log* defines the run logs.
9. Models & model weights are stored at **model_data** folder.
10. **[requirements.txt](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/requirements.txt)** defines the libraries dependency of the experiments. 
11. Bash script  to run multiple expleriments. **[run.sh](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/run.sh)**.
12. **evaluation** folder shows the loss/accuracy plots, also can be viewed in Tensorboards.
13.  **model_data** folder stores the intermediate and final weights of the model.
14. **logs** folder saves the logs for a particular run and create *experiments.log* file.
15. **data** folder contains the datasets.
16. Monitor **experiments.log** for log paths and script progress.
17. Check the **tensorboard logs** by: tensorboard --lodir "path to  tb logs"
18. Check **training_logs.csv** for model training logs. 
19.  **Log paths**: *logs/CombinationID_BackboneModel_DomainLossUsed_LambdaWeight_Original/DateTimeStampValue*) -> MBM
*logs/CombinationID_BackboneModel_DomainLossUsed_LambdaWeight/DateTimeStampValue*) -> CDAN
20.  **For Pruning**: 
*logs/CombinationID_BackboneModel_DomainLossUsed_LambdaWeight_PrunedValue/DateTimeStampValue*) -> CDAN
*logs/CombinationID_BackboneModel_DomainLossUsed_LambdaWeight_Original_PrunedValue/DateTimeStampValue*) -> MBM


### Steps to execute the code: 
 1. Create conda environment (tf): Install all the required dependencies using both **pip** and **conda** as mentioned in the **requirements.txt** file.
 2. Activate conda environment by : **conda activate tf**.
 3. You may launch the program by executing the [main.py](https://github.com/Rajatsharma07/Master-Thesis/blob/main/code/main/main.py) script directly from an IDE or via terminal.
 4. Also, one can run the shell command **sh run.sh** in order to run the series of python experiments.

### Script parameters: 
**python main.py 
--combination="Amazon_to_Webcam"  --architecture="Xception"  --batch_size=16    resize=299  
--learning_rate=0.0001  --mode="train_test"  --lambda_loss=0.5  --epochs=50  
--input_shape=(299,299,3)  --output_classes=31  --loss_function="CORAL"  --augment  --prune
--prune_val=0.30  --technique  --save_weights  --save_model  --use_multiGPU**
