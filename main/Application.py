from Configure import Configurer
from ModelTrainerTaskWise import ModelTrainerTaskWise
from ModelEvaluator import ModelEvaluator
from ModelsEnum import ModelsEnum
import sys
import os
import wandb
import traceback

try:
    config = Configurer()
    args = config.parseArguments()
    
    if not os.path.isdir(args.modelPath):
        raise ValueError("Model Path doesn't exist.")
    
    if args.mode == "train":
        print("In training mode")
        
        if args.nTasks < 1:
            raise ValueError("Number of tasks cannot be less than 1.")
        
        if args.latentSize < 1:
            raise ValueError("Latent Size cannot be less than 1.")
        
        if args.modelLayers > 5 and args.modelLayers < 1:
            raise ValueError("Invalid number of Layers.")
        
        if args.modelFilters > 5 and args.modelFilters < 1:
            raise ValueError("Invalid number of Filters.")
        
        if args.epochs < 1 :
            raise ValueError("Number of epochs cannot be less than 1.")
        
        if args.lr <= 0 :
            raise ValueError("Invalid Learning rate")
        
        config = dict(
            mode = args.mode,
            epochs = args.epochs,
            opt = args.optimizer,
            latent_size = args.latentSize,
            layers = args.modelLayers,
            convLayers = args.modelConvLayers,
            filters = args.modelFilters,
            loss = args.modelLoss,
            lr = args.lr,
            activation = args.activation,
            tasks = args.nTasks,
            batchSize = args.batchSize,
            onlineEWC = args.onlineEWC,
            ewc_lambda = args.ewcLambda,
            gamma = args.gamma
        )
        with(wandb.init(project="streamed_ml", config = config)):
            run_name = wandb.run.name
            trainer = ModelTrainerTaskWise(args.modelPath, config["loss"], config["layers"], config["convLayers"], config["filters"], config["latent_size"], config["epochs"], config["lr"], run_name, args.minNormVal, args.maxNormVal ,config["activation"],config["opt"],batch_size = config["batchSize"],onlineEWC = config["onlineEWC"], ewc_lambda = config["ewc_lambda"], gamma = config["gamma"])
            trainer.train_tasks(config["tasks"], args.saveModelInterval)
    else:
        print("In Evaluation mode.")
        if not args.modelName:
            raise ValueError("Model Name not provided.")
        
        config = dict(
            mode = args.mode,
            model = args.modelName,
            )

        with(wandb.init(project="streamed_ml", config = config)):
            evaluator = ModelEvaluator(args.modelPath, args.modelName, [20,40,60,80], ModelsEnum.Autoencoder_Pooling)
            evaluator.evaluate()
            evaluator.image_show()
        
except ValueError as err:
    print(err)
except:
    print(traceback.format_exc())