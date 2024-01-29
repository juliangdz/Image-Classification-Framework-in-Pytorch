import json 

def read_config(config_path='config.json'):
    with open(config_path,"r") as cf:
        data = json.load(cf)
    return data

def modify_config(mod_config,config_path='config.json'):
    with open(config_path,"w") as cf:
        json.dump(mod_config,cf,indent=4)
    config = read_config(config_path=config_path)
    return config