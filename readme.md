
# VoE Dataset Generation and Models
This repository hosts the dataset generation code and code for the models.


## Dataset Generation
The dataset must be generated in blender's python API for version `2.93.1`. The generation uses the bpycv library for generation. The python environment requirements can be found in `dataset_generation/requirements.txt`. The variations are fixed for this dataset and are specified in the CSV files. 

## Models
To train the model with the data, set up the `models/setup.yaml` file and run `python models/main.py` using the environment from `models/requirements.txt`. Note that `resnet3d_direct` refers to the baseline model. You must set the `ABSOLUTE_DATA_PATH`  to the absolute path to the data. An instance of the config can be seen as such:

    ###############################################
    # training and testing related configurations #
    ###############################################
    
    TRAIN: true
    TEST: true
    
    TRAINING:
      # options: "A", "B", "C", "D", "E"
      EVENT_CATEGORY: "A"
      # options: "resnet3d_direct", "random", "OF_PR", "Ablation"
      MODEL_TYPE: "random"
      # path to save experiment data *relative to main.py*
      RELATIVE_SAVE_PATH: "experiments/"
      # path to save dataset objects for quick loading and retrieval *relative to main.py*
      RELATIVE_DATASET_PATH: "datasets/"
      # path to AVoE Dataset (with the 5 folders for each event category)
      ABSOLUTE_DATA_PATH: ''
      # path to state_dict of model to load (best_model.pth) *relative to main.py*
      RELATIVE_LOAD_MODEL_PATH: null
      # type of decision tree. Options: "normal", "direct", "combined"
      DECISION_TREE_TYPE: "combined"
      # to use oracle model for observed outcome
      SEMI_ORACLE: true
      # to use gpu or not to
      USE_GPU: true
      # learning rate for training
      LEARNING_RATE: 0.001
      # number of epochs
      NUM_EPOCHS: 10
      # batch size for training
      BATCH_SIZE: 16
      # to use pretrained wights for transfer learning
      USE_PRETRAINED: true
      # whether to freeze pretrained weights
      FREEZE_PRETRAINED_WEIGHTS: true
      # set the random seed
      RANDOM_SEED: 1
      # optimizer type. Options: ['adam', 'sgd']
      OPTIMIZER_TYPE: "adam"
      # dataset efficiency (cpu) Options: ['time', 'memory'] time loads entire dataset into CPU
      DATASET_EFFICIENCY: 'time'
    
    TESTING:
      # options: "A", "B", "C", "D", "E", "combined"
      EXPERIMENT_ID: null
      # path to save experiment data *relative to main.py*
      RELATIVE_SAVE_PATH: "experiments/"
      # path to save dataset objects for quick loading and retrieval *relative to main.py*
      RELATIVE_DATASET_PATH: "datasets/"
      # path to AVoE Dataset (with the 5 folders for each event category)
      ABSOLUTE_DATA_PATH: ''
      # type of decision tree. Options: "normal", "direct", "combined"
      DECISION_TREE_TYPE: "combined"
      # to use oracle model for observed outcome
      SEMI_ORACLE: true
      # to use gpu or not to
      USE_GPU: true
      # dataset efficiency (cpu) Options: ['time', 'memory'] time loads entire dataset into CPU
      DATASET_EFFICIENCY: 'time'
