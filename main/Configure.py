import argparse
import sys

class Configurer:
    def __init__(self):
        print("Initialized Configurer")
    
    def parseArguments(self):
        parser = argparse.ArgumentParser(description='openPMD Pipe')
        parser.add_argument("--modelPath", type=str, help='Model store or read file path.', required =True)
        parser.add_argument("--nTasks", type=int, help='Number of tasks to be trained on', default = 101)
        parser.add_argument("--modelLoss", type=str, help='Loss function.', choices=['L1', 'MSE','custom'], default = 'MSE')
        parser.add_argument("--modelLayers", type=int, help='Number of layers.', default = 5)
        parser.add_argument("--modelConvLayers", type=int, help='Number of Convolutional layers.', default = 5)
        parser.add_argument("--modelFilters", type=int, nargs='+', help='Number of Filters.')
        parser.add_argument("--latentSize", type=int, help='Encoded layer size.', default = 1000)
        parser.add_argument("--epochs", type=int, help='Number of epochs.', default = 50)
        parser.add_argument("--batchSize", type=int, help='Batch Size.', default = 4)
        parser.add_argument("--offlineTrain", type=bool, help='Offline or Online Training', default = False)
        parser.add_argument("--saveModelInterval", type=int, help='Save Model Interval', default = 1)
        parser.add_argument("--lr", type=float, help='Learning Rate.',default=0.001)
        parser.add_argument("--activation", type=str, help='Activation functions.',choices=['relu', 'tanh','leaky_relu'], default = 'leaky_relu')
        parser.add_argument("--datasetName", type=str,
                            help='Dataset to train on.', required=True)
        parser.add_argument("--optimizer", type=str, help='Model Optimizer',choices=['adam', 'sgd'], default = 'adam')
        parser.add_argument("--mode",type=str, help='Running mode of the application', choices = ['eval','train'], default ='train')
        parser.add_argument("--modelName",type=str, help='Name of Model to be Evaluated')
        parser.add_argument("--ewcLambda",type=float, help="How strong to weigh EWC-loss ('regularisation strength')", default = 0.0)
        parser.add_argument("--gamma",type=float, help="Decay-term for old tasks - 'contribution to quadratic term'", default = 0.0)
        parser.add_argument("--horovod",type=bool, help="Using Horovod", default = False)
        parser.add_argument("--replayMemory",type=int, help="Memory size for replay", default = 0)
        parser.add_argument("--refGradMemory",type=int, help="Trainable memory for Reference Gradient (A-GEM)", default = 0)
        parser.add_argument("--agemEncLambda",type=float, help="Proposed method regularizer to scale encoded loss to reconstructed loss", default = 1.0)
        parser.add_argument("--lowMemStorage",type=bool, help="Store encoded Data for replayer", default = False)
        parser.add_argument("--layerWise",type=bool, help="User replayer methods layerwise or not", default = False)
        args = parser.parse_args()
        return args
