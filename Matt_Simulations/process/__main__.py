# this file is only called when the package is called from the command line

import yaml
import sys

if __name__ == "__main__":

    config_file = sys.argv[1];

    config = {}
    fh = open(config_file, "r")

    for k,v in yaml.load(fh).items():
        config[k] = v

    ## Now just call pipeline from here
