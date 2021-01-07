import argparse
import sys

class Configurer:
    def __init__(self):
        pass
    
    def parseArguments(self):
        parser = argparse.ArgumentParser(description='openPMD Pipe')
        parser.add_argument("--streamFile", type=str, help='Stream file with location appened.')
        parser.add_argument("--numReadNodes", type=int, help='Number of readabel Cluster.')
        args = parser.parse_args()
        return args
    
    def validateArguments(self, config):
        missing = []
        validation_result = True
        
        if config.streamFile is None:
            missing.append("--streamFile")
            validation_result = False
        if config.numReadNodes is None:
            missing.append("--numReadNodes")
            validation_result = False
            
        return missing,validation_result
        