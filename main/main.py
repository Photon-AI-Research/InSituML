from Configure import Configurer
from StreamDataReader import stream_reader
import os
import sys
import logging

try:
    config = Configurer()
    parsedArgs = config.parseArguments()
    missingList, validation_result = config.validateArguments(parsedArgs)
    if(validation_result):
        if(not os.path.isfile(parsedArgs.streamFile)):
            raise FileNotFoundError
        if(parsedArgs.numReadNodes < 1):
            raise ValueError
        
    #start data reader
        
    else:
        print("Missing arguments:",str(missingList))
except FileNotFoundError:
    print("No such stream File Found! " + config.streamFile)
except ValueError:
    print("Invalid value for number of nodes.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise