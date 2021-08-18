import argparse
import sys

class Configurer:
    def __init__(self):
        print("Initiliazed Configurer")
    
    def parseArguments(self):
        parser = argparse.ArgumentParser(description='openPMD Pipe')
        parser.add_argument("--modelPath", type=str, help='Model store or read file path.', required =True)
        parser.add_argument("--nTasks", type=int, help='Number of tasks to be trained on', default = 101)
        parser.add_argument("--modelLoss", type=str, help='Loss function.', choices=['L1', 'MSE'], default = 'MSE')
        parser.add_argument("--modelLayers", type=int, help='Number of layers.', default = 5)
        parser.add_argument("--modelConvLayers", type=int, help='Number of Convolutional layers.', default = 5)
        parser.add_argument("--modelFilters", type=int, help='Number of Filters.', default = 3)
        parser.add_argument("--latentSize", type=int, help='Encoded layer size.', default = 1000)
        parser.add_argument("--epochs", type=int, help='Number of epochs.', default = 50)
        parser.add_argument("--batchSize", type=int, help='Batch Size.', default = 4)
        parser.add_argument("--saveModelInterval", type=int, help='Save Model Interval', default = 20)
        parser.add_argument("--lr", type=float, help='Learning Rate.',default=0.01)
        parser.add_argument("--activation", type=str, help='Activation functions.',choices=['relu', 'tanh','leaky_relu'], default = 'leaky_relu')
        parser.add_argument("--optimizer", type=str, help='Model Optimizer',choices=['adam', 'sgd'], default = 'adam')
        parser.add_argument("--mode",type=str, help='Running mode of the application', choices = ['eval','train'], default ='train')
        parser.add_argument("--modelName",type=str, help='Name of Model to be Evaluated')
        parser.add_argument("--minNormVal",type=float, help='Minimum Normalize constant.', default = -2.1393063)
        parser.add_argument("--maxNormVal",type=float, help='Maximum Normalize constant.', default = 2.140939)
        parser.add_argument("--onlineEWC",type=bool, help='Should Apply EWC', default = False)
        parser.add_argument("--ewcLambda",type=float, help="How strong to weigh EWC-loss ('regularisation strength')", default = 0.0)
        parser.add_argument("--gamma",type=float, help="Decay-term for old tasks - 'contribution to quadratic term'", default = 0.0)
        args = parser.parse_args()
        return args
