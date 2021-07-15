#!/usr/bin/env python3
"""
*  F. CHUNG  ANDL. LU,The  average  distances  in  random  graphs  withgiven expected degrees, PNAS, 99 (2002), pp. 15879–15882.
"""


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
from numba import jit

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
    parser.add_option("--freq", action="store", dest="frequency",
                                   default = "day", help = "time aggregation of networks - choose between day, week, 2weeks, 4weeks")


    options, args = parser.parse_args()

    options.currency = SYMBOLS[options.currency]


    options.cluster_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}/"

    options.cluster_data_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/"

    options.network_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_networks_{options.frequency}/"


    options.output_folder = f"{options.output_folder}/heur_{options.heuristic}_data/"
    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    options.black_active_folder = options.output_folder + f'active_black_nodes_{options.frequency}/'

    options.output_csv = f"{options.output_folder}/modularity_net_{options.frequency}.csv"


    return options, args

def interlinks_ratio(g, n):
    e = nx.edge_boundary(g, n)
    k_out_n = g.out_degree(n)
    k_in_n = g.in_degree(n)
    m = g.size()
    nc = g.nodes - n
    k_out_nc = g.out_degree(nc)
    k_in_nc = g.in_degree(nc)

@jit(nopython=True)
def f(e, kout1,kin1,kout2,kin2):
    s = 0

if __name__ == "__main__":
    options, args = parse_command_line()

    # heur_file   = zarr.open_array(f"{options.cluster_folder}_data/address_cluster_map.zarr", mode = 'r')
    # memstore    = zarr.MemoryStore()
    # zarr.copy_store(heur_file.store, memstore)
    # heur_mem    = zarr.open_array(memstore)
    chrono = SimpleChrono()

    # LOAD STUFF
    """
    chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}.cfg")
    am = AddressMapper(chain)
    am.load_clusters(f"{options.cluster_data_folder}")
    addr_cluster_map = zarr.load(f"{options.cluster_data_folder}/address_cluster_map.zarr")
    """
    # black_cluster: index-cluster, bool value-true if cluster is black
    clust_is_black_ground = zarr.load(f"{options.black_data_folder}/cluster_is_black_ground_truth.zarr")
    # networks list
    network_list = [n for n in os.listdir(options.network_folder)]
    network_list.sort()

    # PRE-PROCESSING

    col_names = ['date', 'ground_truth_modularity', 'old_black_modularity']
    csv_fout = DictWriter(open(options.output_csv, "w") , fieldnames=col_names)

    csv_fout.writeheader()

    # set of black users
    # node type should change to str because in graphs nodes ids are str
    clust_is_black_ground_set = set([str(i) for i in range(len(clust_is_black_ground)) if clust_is_black_ground[i]])

    chrono.print(message="init")

    print(f"[CALC] starts black bitcoin diffusion...")


    # RUN ON ALL NETWORKS
    for network in network_listi[3000:3001]:
        network_date = network[:-12]
        _ = {}
        # load network
        g = nx.read_graphml(options.network_folder + network)

        with open(options.black_active_folder + network_date + ".pkl", 'rb') as pfile:
            old_black_nodes, active_black_nodes = pkl.load(pfile)


        _['date'] = network_date
        _['ground_truth_modularity'] =
        _['old_black_modularity'] =

        csv_fout.writerow(_)


    chrono.print(message="took", tic="last")



   # addr_no_input_tx = zarr.load(f"{options.data_in_folder}/cluster_no_input_tx.zarr")
   # addr_no_output_tx = zarr.load(f"{options.data_in_folder}/cluster_no_output_tx.zarr")


