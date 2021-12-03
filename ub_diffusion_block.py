#!/usr/bin/env python3
"""
input:
    - `{options.black_data_folder}/cluster_is_black_ground_truth.zarr` ground truth clusters from `ub_ground_truth.py`
    - `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` clustering data
    - `{DIR_PARSED}/{options.currency}.cfg` blockchain data
outputs:
    * zarr file: `cluster_is_black_when_block.zarr` index is cluster id, value is int block when the cluster became black
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


class AddressMapper():
    def __init__(self, chain):
        self.chain = chain

        self.__address_types = [blocksci.address_type.nonstandard, blocksci.address_type.pubkey,
                                blocksci.address_type.pubkeyhash, blocksci.address_type.multisig_pubkey,
                                blocksci.address_type.scripthash, blocksci.address_type.multisig,
                                blocksci.address_type.nulldata, blocksci.address_type.witness_pubkeyhash,
                                blocksci.address_type.witness_scripthash, blocksci.address_type.witness_unknown]

        self.__counter_addresses = { _:self.chain.address_count(_) for _ in self.__address_types }

        self.__offsets = {}
        offset = 0
        for _ in self.__address_types:
            self.__offsets[_] = offset
            offset += self.__counter_addresses[_]


        self.total_addresses = offset
        print(f"[INFO] #addresses: {self.total_addresses}")
#        print(self.__counter_addresses)


    def map_clusters(self,cm):
#        address_vector = {_: np.zeros(self.__counter_addresses[_], dtype=np.int64) for _ in self.__address_types }
        cluster_vector = {_: np.zeros(self.__counter_addresses[_], dtype=np.int64) for _ in self.__address_types }

        self.cluster = np.zeros(self.total_addresses, dtype=np.int64)
        offset = 0
        for _at in cluster_vector.keys():
            clusters = cluster_vector[_at]
            print(f"{_at}     -  {len(clusters)}")
#            addrs = address_vector[_at]
            for _i, _add in enumerate(chain.addresses(_at)):
#                addrs[_i] = _add.address_num
                clusters[_i] = cm.cluster_with_address(_add).index
                #max_addr_num = max(max_addr_num, addrs[_i])
#        pickle.dump(cluster_vector, open("cluster_dict.pickle","wb"))

        offset = 0
        for _ in cluster_vector.keys():
            v = cluster_vector[_]
            self.cluster[offset:offset + len(v)] = v
            offset += len(v)



    def dump_clusters(self, output_folder):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        zarr.save(f"{output_folder}/address_cluster_map.zarr", self.cluster)


#    def dump_offsets(self, output_folder):
#        if not os.path.exists(output_folder):
#            os.mkdir(output_folder)
#        pickle.dump(self.__offsets, open(f"{output_folder}/offsets.pickle", "wb") )

#    def load_offsets(self, output_folder):
#        if not os.path.exists(output_folder):
#            os.mkdir(output_folder)
#        self.__offsets = pickle.load( open(f"{output_folder}/offsets.pickle", "rb") )

    def load_clusters(self, input_folder):
        self.cluster = zarr.load(f"{input_folder}/address_cluster_map.zarr")



    def __getitem__(self,addr):
        return self.__offsets[addr.raw_type]+ addr.address_num-1

if __name__ == "__main__":
    options, args = parse_command_line()

    # heur_file   = zarr.open_array(f"{options.cluster_folder}_data/address_cluster_map.zarr", mode = 'r')
    # memstore    = zarr.MemoryStore()
    # zarr.copy_store(heur_file.store, memstore)
    # heur_mem    = zarr.open_array(memstore)
    chrono = SimpleChrono()

    # LOAD STUFF
    chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}.cfg")

    am = AddressMapper(chain)
    am.load_clusters(f"{options.cluster_data_folder}")
    # black_cluster: index-cluster, bool value-true if cluster is black
    clust_is_black_ground = zarr.load(f"{options.black_data_folder}/cluster_is_black_ground_truth.zarr")
    # PRE-PROCESSING

    # define blocks range after given dates
    if options.start_date == None:
        start_date = datetime.fromtimestamp(chain.blocks[0].timestamp).date()
    else:
        start_date = datetime.strptime(options.start_date, "%Y-%m-%d").date()
    if options.end_date == None:
        end_date = datetime.fromtimestamp(chain.blocks[-1].timestamp).date()
    else:
        end_date = datetime.strptime(options.end_date, "%Y-%m-%d").date()

    blocks_range = chain.range(start_date, end_date)


    # set of black users
    clust_is_black_ground_set = set(compress(range(len(clust_is_black_ground)), clust_is_black_ground))
    clust_is_black_when = np.zeros(len(clust_is_black_ground), dtype=int)

    chrono.print(message="init")

    print(f"[CALC] starts black bitcoin diffusion...")

    # RUN ON ALL BLOCKS
    for b in blocks_range: 
        new_black_nodes = set([])

        # on a single trx
        for trx in b.txes:
            loc_new_black_nodes = set([])

            flag_input_black = False

            if trx.is_coinbase: continue

            for inp in trx.inputs:
                cluster = am.cluster[am[inp.address]]
                # if the input is already black
                if cluster in clust_is_black_ground_set:
                    # at least on of the input is black
                    flag_input_black = True

            # if at least one input is black
            if flag_input_black:
                for out in trx.outputs:
                    cluster = am.cluster[am[out.address]]
                    if cluster not in clust_is_black_ground_set:
                        # the cluster was not black before
                        # add the cluster to new black nodes
                        loc_new_black_nodes.add(cluster)
                        # update clust_is_black_when
                        clust_is_black_when[cluster] = b.height

            new_black_nodes.update(loc_new_black_nodes)

        # block level
        # update black nodes and write them
        clust_is_black_ground_set.update(new_black_nodes)


    zarr.save(options.output_folder + f'cluster_is_black_when_block.zarr', clust_is_black_when)

    chrono.print(message="took", tic="last")



   # addr_no_input_tx = zarr.load(f"{options.data_in_folder}/cluster_no_input_tx.zarr")
   # addr_no_output_tx = zarr.load(f"{options.data_in_folder}/cluster_no_output_tx.zarr")


