from ModelTrainer import ModelTrainer
from Configure import Configurer
from ModelEvaluator import ModelEvaluator
from ModelsEnum import ModelsEnum
import wandb
import traceback
import os
from ReplayTrainer import ReplayTrainer

if __name__ == '__main__':
    try:
        config = Configurer()
        args = config.parseArguments()

        """ Validation checks before beginning of training """

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

            if args.epochs < 1:
                raise ValueError("Number of epochs cannot be less than 1.")

            if args.lr <= 0:
                raise ValueError("Invalid Learning rate")

            config = dict(
                mode=args.mode,
                epochs=args.epochs,
                opt=args.optimizer,
                latent_size=args.latentSize,
                layers=args.modelLayers,
                convLayers=args.modelConvLayers,
                filters=args.modelFilters,
                loss=args.modelLoss,
                lr=args.lr,
                activation=args.activation,
                tasks=args.nTasks,
                batchSize=args.batchSize,
                onlineEWC=True if args.ewcLambda > 0.0 else False,
                ewc_lambda=args.ewcLambda,
                saveModelInterval=args.saveModelInterval,
                gamma=args.gamma,
                dataset=args.datasetName,
                offlineTrain=True if args.nTasks == 1 else False,
                Replayer=True if args.replayMemory > 0 else False,
                A_GEM=True if args.refGradMemory > 0 else False,
                EncodedStorage= args.lowMemStorage,
                LayerwiseGradUpdate = args.layerWise,
                Episodic_memory = args.replayMemory,
                Gradient_episodic_memory= args.refGradMemory,
                methodLambda = args.agemEncLambda,
                masLambda=0.0,
            )

            """ 
                'classes' variable defines number of classes/data iterations 
                that will be split into tasks based on (classes / nTasks) 
            """ 

            is_e_field = False
            if 'mnist' in args.datasetName:
                input_channels = 1
                input_sizes = (28, 28)
                #model_type = ModelsEnum.Autoencoder2D
                model_type = ModelsEnum.MLP
                classes = 10
            elif 'cifar' in args.datasetName:
                input_channels = 3
                input_sizes = (32, 32)
                model_type = ModelsEnum.Autoencoder2D
                classes = 10
                if 'cifar-100' in args.datasetName:
                    classes = 20
            elif 'e_field' in args.datasetName:
                is_e_field = True
                input_channels = 3
                input_sizes = (128, 1280, 128)
                model_type = ModelsEnum.Autoencoder3D
                classes = 50
                
            with(wandb.init(project="streamed_ml", config=config)):
                run_name = wandb.run.name
                if not config["Replayer"]:
                    trainer = ModelTrainer(args.modelPath, config["loss"], input_channels, config["layers"], config["convLayers"], config["filters"], config["latent_size"], config["epochs"], config["lr"], run_name, input_sizes, config["tasks"], config["dataset"], classes, config["saveModelInterval"],
                                           model_type, e_field_dimension=None, is_e_field=is_e_field, activation=config["activation"], optimizer=config["opt"], batch_size=config["batchSize"], onlineEWC=config["onlineEWC"], ewc_lambda=config["ewc_lambda"], gamma=config["gamma"], mas_lambda=config["masLambda"])
                    trainer.train()
                else:
                    print("Replay Training....")
                    trainer = ReplayTrainer(args.modelPath, config["loss"], input_channels, config["layers"], config["convLayers"], config["filters"], config["latent_size"], config["epochs"], config["lr"], run_name, input_sizes, config["tasks"], config["dataset"], classes, config["saveModelInterval"], replayer_mem_size= args.replayMemory, aGEM_selection_size= args.refGradMemory, store_encoded = config["EncodedStorage"], layerWise = config["LayerwiseGradUpdate"],model_type = model_type, e_field_dimension=None, is_e_field=is_e_field, activation=config["activation"], optimizer=config["opt"], batch_size=config["batchSize"], onlineEWC=config["onlineEWC"], ewc_lambda=config["ewc_lambda"], gamma=config["gamma"], mas_lambda=config["masLambda"], agem_l_enc_lambda=config["methodLambda"])
                    trainer.train_with_replay()
        else:
            print("In Evaluation mode.")
            if not args.modelName:
                raise ValueError("Model Name not provided.")

            config = dict(
                mode=args.mode,
                model=args.modelName,
            )

            with(wandb.init(project="streamed_ml", config=config)):
                evaluator = ModelEvaluator(args.modelPath, args.modelName, [
                                           20, 40, 60, 80], ModelsEnum.Autoencoder_Pooling)
                evaluator.evaluate()
                evaluator.image_show()

    except ValueError as err:
        print(err)
    except:
        print(traceback.format_exc())
