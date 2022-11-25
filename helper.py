import json
import os
from os.path import exists


def checkDirectory(path):
    isExist = exists(path)
    if not isExist:
        os.makedirs(path)
        print("Directory created at : %s" % path)


def checkFile(path):
    isExist = exists(path)
    if not isExist:
        json_object = json.dumps({}, indent=4)
        with open(path, "w") as outfile:
            outfile.write(json_object)
        print("Configuration file created at : %s" % path)


def check_configuration(config):
    print("=" * 10 + " Initializing Environment and Sanity Check " + "=" * 10)
    checkDirectory(config['model_path'])
    checkDirectory(config['plots_path'])
    checkFile(config['model_path'] + 'configurations.json')
    config["model_path_actor"] = config["model_path"] + config["model_name"] + "_actor" + ".pth"
    config["model_path_critic"] = config["model_path"] + config["model_name"] + "_critic" + ".pth"
    config["preload_model"] = (exists(config["model_path_actor"]) and exists(config["model_path_critic"]))
    return


def loadConfigurationForModel(configuration):
    if not configuration['preload_model']:
        return configuration
    else:
        modelName = configuration["model_name"]
        modelPath = configuration["model_path"]
        device = configuration['device']
        data = loadConfigurationFile(modelPath)
        try:
            configuration = data[modelName]
            configuration['device'] = device
        except KeyError:
            print("Model not found, try training a fresh model or pass a valid model name")
            exit(-1)
        return configuration

def loadConfigurationFile(modelPath):
    f = open(modelPath + "configurations.json", "r")
    data = json.load(f)
    return data