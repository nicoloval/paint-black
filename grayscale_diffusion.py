import blocksci

import sys, os, os.path, socket
import numpy as np
import networkx as nx
import zarr
import time
from tqdm import tqdm
import pandas as pd
from csv import DictWriter, DictReader
import pickle as pkl
from datetime import datetime, timedelta
from itertools import compress

from util import SYMBOLS, DIR_BCHAIN, DIR_PARSED, SimpleChrono

def parse_command_line():
    import sys, optparse

    parser = optparse.OptionParser()

    parser.add_option("--curr", action='store', dest="currency", type='str',
                                              default=None, help="name of the currency")
    parser.add_option("--heur", action='store', dest="heuristic", type='str',
                                                  default=None, help="heuristics to apply")
    parser.add_option("--overwrite", action='store_true', dest = "overwrite" )
    parser.add_option("--output", action='store', dest = "output_folder", 
                        default="uniform_black/", type='str', help='directory to save outputs in')
    parser.add_option("--start", action="store", dest="start_date",
                                   default = None, help= "starting date for network creation in YYYY-MM-DD format")
    parser.add_option("--end", action="store", dest="end_date",
                                       default = None, help = "ending date for network creation in YYYY-MM-DD format")


    options, args = parser.parse_args()

    options.currency = SYMBOLS[options.currency]


    options.cluster_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}/"

    options.cluster_data_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/"

    options.output_folder = f"{options.output_folder}/heur_{options.heuristic}_data/"
    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    """
    options.output_active_folder = options.output_folder + f'black_nodes_by_block/'
    if not os.path.exists(options.output_active_folder):
        os.mkdir(options.output_active_folder)
    """

    options.output_csv = f"{options.output_folder}/diffusion_block.csv"


    # atm ground truth is in the output folder
    options.black_data_folder = options.output_folder


    return options, args