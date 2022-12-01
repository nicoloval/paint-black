#!/usr/bin/env python3
"""
input:
    - `{options.black_data_folder}/cluster_is_black_ground_truth.zarr` ground truth clusters from `ub_ground_truth.py`
    - `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` clustering data
    - `{DIR_PARSED}/{options.currency}.cfg` blockchain data
outputs:
    * zarr file: `cluster_is_black_when_block.zarr` index is cluster id, value is int block when the cluster became black which can also represent time.
"""
# here in this script we replicate the diffusion and from ground-truth we see how users turn black block by block

import blocksci
from decimal import Decimal

import sys, os, os.path, socket
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import networkx as nx
import zarr
import time
from tqdm import tqdm
import pandas as pd
from csv import DictWriter, DictReader
import pickle as pkl
from datetime import datetime, timedelta
from itertools import compress
from scipy.sparse import csc_matrix
from collections import defaultdict

from util import SYMBOLS, DIR_BCHAIN, DIR_PARSED, SimpleChrono

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

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


class AddressMapper(): # same as before
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
        cluster_vector = {_: np.zeros(self.__counter_addresses[_], dtype=np.int64) for _ in self.__address_types }

        self.cluster = np.zeros(self.total_addresses, dtype=np.int64)
        offset = 0
        for _at in cluster_vector.keys():
            clusters = cluster_vector[_at]
            print(f"{_at}     -  {len(clusters)}")
            for _i, _add in enumerate(chain.addresses(_at)):
                clusters[_i] = cm.cluster_with_address(_add).index
        offset = 0
        for _ in cluster_vector.keys():
            v = cluster_vector[_]
            self.cluster[offset:offset + len(v)] = v
            offset += len(v)



    def dump_clusters(self, output_folder):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        zarr.save(f"{output_folder}/address_cluster_map.zarr", self.cluster)


    def load_clusters(self, input_folder):
        self.cluster = zarr.load(f"{input_folder}/address_cluster_map.zarr")



    def __getitem__(self,addr):
        return self.__offsets[addr.raw_type]+ addr.address_num-1


class Graph:
    
    def __init__(self):
        self.graph = dict()
        
        
    def add_edge(self, node1, node2, cost):
        if node1 not in self.graph:
            self.graph[node1] = []
            
        if node2 not in self.graph:
            self.graph[node2] = []
        
        self.graph[node1].append((node2, int(cost)))
        
        if len(self.graph[node1]) == 0:
            self.graph.pop(node1)
        
    def print_graph(self):
        
        for source, destination in self.graph.items():
            print(f"{source}->{destination}")
    
    def graph_size(self):
        return len(self.graph)

if __name__ == "__main__":
    options, args = parse_command_line()

    # Start Chrono
    chrono = SimpleChrono()

    # Load chain and initialize address mapper
    chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}_old.cfg")
    am = AddressMapper(chain)
    am.load_clusters(f"{options.cluster_data_folder}")

    # black_cluster: index-cluster, bool value-true if cluster is black. We use the same file we got from ub_ground_truth.py file
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
    clust_is_black_ground_set = set(compress(range(len(clust_is_black_ground)), clust_is_black_ground)) # transform clust_is_black_ground into a set where we consider only black clusters.
    # initialize empty array
    # clust_is_black_when = np.zeros(len(clust_is_black_ground), dtype=int) 

    chrono.print(message="init")
    print(f"[CALC] Starting the grayscale diffusion for all the blockchain...")

    clustered_nodes = set([]) # the set of all clustered nodes we have seen so far

    # New_black_nodes = set([])
    current_assets = defaultdict(lambda: 0)
    dark_assets = defaultdict(lambda: 0)
    dark_ratio = defaultdict(lambda: 0.0)
    loop = 0

    # RUN ON ALL BLOCKS
    for block in tqdm(blocks_range):
        # Initialize 
        
        # print(f'current_assets before block has started:{block.height}')
        # i = 0
        # for k, v in current_assets.items():
        #     print(f'{k}:{format_e(Decimal(v))}, ', end='')
        #     if i == 9:
        #         print('\n')
        #         i = 0
        #     i+=1
        
        # print('\n')

        # print(f'dark_assets before block has started:{block.height}')
        # i = 0
        # for k, v in dark_assets.items():
        #     print(f'{k}:{format_e(Decimal(v))}, ', end='')
        #     if i == 9:
        #         print('\n')
        #         i = 0
        #     i+=1
        
        # print('\n')

        # Initialize a graph object
        g = Graph()

        trx_is_dark = False 

        #______________________________TRX level_____________________________________

        for trx in block.txes:
            #______________________________Initialize Variables_____________________________________

            #loc_new_black_nodes = set([]) # temporary variable as a set
            #flag_input_black = False # to flag or checks if in this transaction has black input
            clustered_inputs_dict = defaultdict(list)
            clustered_outputs_dict = defaultdict(list)
            #new dictionaries
            loc_new_clustered_nodes = set([]) # the temporary set of local clustered nodes in this trx
            total_trx_input_value = 0

            # skip mining transactions which do not have inputs but just miner output?
            if trx.is_coinbase:

                for out in trx.outputs:
                    cluster, value = am.cluster[am[out.address]], out.value + trx.fee
                    
                    # if not trx.is_coinbase:
                    #     print(f'printing outputs for trx:{trx}')
                    #     print(f'output cluster:{cluster}, value:{value}')
                        
                    # if the input(address) has already been clustered
                    if cluster in list(current_assets.keys()):
                        #clustered_outputs_dict[cluster].append(value)
                        current_assets[cluster] = current_assets[cluster] + value
                        if current_assets[cluster] > 0:
                            dark_ratio[cluster] = dark_assets[cluster]/current_assets[cluster]
                    else:
                        #clustered_outputs_dict[cluster] = [value]
                        current_assets[cluster] = value
                        if current_assets[cluster] > 0:
                            dark_ratio[cluster] = dark_assets[cluster]/current_assets[cluster]
                    
                    # if the input address has not been seen before globally or locally add it to loc_new_clustered_nodes
                    if cluster not in clustered_nodes.union(loc_new_clustered_nodes):
                        loc_new_clustered_nodes.add(cluster)
                    
                    continue
            
            # loop over trx inputs to build a reduced representation of inputs
            # if not trx.is_coinbase:
            #     print(f'printing inputs for trx:{trx}')
            for inp in trx.inputs: 
                cluster, value = am.cluster[am[inp.address]], inp.value

                if cluster in clust_is_black_ground_set:
                    trx_is_dark = True

                # if not trx.is_coinbase:
                #     print(f'input cluster:{cluster}, value:{value}')

                # if the input(address) has already been clustered
                if cluster in list(clustered_inputs_dict.keys()):
                    clustered_inputs_dict[cluster].append(value)
                    #current_assets[cluster] = current_assets[cluster] + value
                    
                else:
                    clustered_inputs_dict[cluster] = [value]
                    #current_assets[cluster] = value

                # if the input address has not been seen before globally or locally add it to loc_new_clustered_nodes
                if cluster not in clustered_nodes.union(loc_new_clustered_nodes):
                    loc_new_clustered_nodes.add(cluster)
                    
                total_trx_input_value = total_trx_input_value + value
            
            # loop over trx inputs to build a reduced representation of inputs

            
            for out in trx.outputs:
                cluster, value = am.cluster[am[out.address]], out.value

                if cluster in clust_is_black_ground_set:
                    trx_is_dark = True
                   
                # if not trx.is_coinbase:
                #     print(f'Result_total_trx_input_value:{total_trx_input_value} for trx index:{trx.index}')
                #     print(f'printing outputs for trx:{trx}')
                #     print(f'output cluster:{cluster}, value:{value}')
                    

                # if the input(address) has already been clustered
                if cluster in list(clustered_outputs_dict.keys()):
                    clustered_outputs_dict[cluster].append(value)
                else:
                    clustered_outputs_dict[cluster] = [value]
                
                # if the input address has not been seen before globally or locally add it to loc_new_clustered_nodes
                if cluster not in clustered_nodes.union(loc_new_clustered_nodes):
                    loc_new_clustered_nodes.add(cluster)

            clustered_nodes.update(loc_new_clustered_nodes)
            # if not trx.is_coinbase:
            #     print("----------Clustered inputs and outputs----------")
            #     print(f'clustered_inputs_dict:{clustered_inputs_dict}')
            #     print(f'clustered_outputs_dict:{clustered_outputs_dict}')

            #-------------------------------------------------------------------------------#
            
            # This loop automatically ignores coinbase transactions since they have no input ?
            for out_sender, sender_value in clustered_inputs_dict.items():
            
                #Set all the assets as dark assets if the cluster belongs to black ground truth
                if out_sender in clust_is_black_ground_set:
                    dark_assets[out_sender] = current_assets[out_sender]
                    dark_ratio[out_sender] = 1
                    # print(f'{out_sender} is a black node')
                
                if total_trx_input_value == 0:
                    continue

                for out_receiver, receiver_value in clustered_outputs_dict.items():

                    if out_receiver in clust_is_black_ground_set:
                        dark_assets[out_receiver] = current_assets[out_receiver]
                        dark_ratio[out_receiver] = 1
                        # print(f'{out_receiver} is a black node')

                    # Calculate the weight of the edge and add the edge to the graph
                    weight = sum(sender_value)/total_trx_input_value*sum(receiver_value)

                    # if not trx.is_coinbase:
                    #     print("----------Graph building and weight calculation----------")
                    #     print(f'out_sender:{out_sender} , sender_value:{sender_value}')
                    #     print(f'out_receiver:{out_receiver} , receiver_value:{receiver_value}')
                    #     print(f'sum(sender_value):{sum(sender_value)} , sum(receiver_value):{sum(receiver_value)}')
                    #     print(f'total_trx_input_value:{total_trx_input_value}')
                    #     print(f'weight:{weight}')

                    g.add_edge(out_sender, out_receiver, weight)
                    
                    #Calculate the dark ratio of the sender
                    
                    if current_assets[out_sender] > 0:
                        dark_ratio[out_sender] = dark_assets[out_sender] / current_assets[out_sender]

                    #update the current assets of the sender
                    current_assets[out_sender] = current_assets[out_sender] - weight

                    #Update the current assets of the receiver
                    current_assets[out_receiver] = current_assets[out_receiver] + weight

                    #update the dark assets of the sender and receiver.
                    dark_assets[out_sender] = dark_assets[out_sender] - (weight*dark_ratio[out_sender])
                    dark_assets[out_receiver] = dark_assets[out_receiver]+ (weight*dark_ratio[out_sender])

                    #update dark ratio of the receiver
                    if current_assets[out_receiver] > 0:
                        dark_ratio[out_receiver] = dark_assets[out_receiver]/current_assets[out_receiver]
        
        # os.chdir('/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/')
        # print("Current working directory before")
        # print(os.getcwd())
        # print(block)

        #dark_assets_arr = np.array(list(dark_assets.values()))
        current_assets_values = np.array(list(current_assets.values()))
        dark_ratio_values = np.array(list(dark_ratio.values()))
        current_assets_index = np.array(list(current_assets.keys()))
        dark_ratio_index = np.array(list(dark_ratio.keys()))

        zarr.save("/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_zarr/dark_ratio/" + f'dark_ratio_values_block_{str(block.height).zfill(6)}.zarr', dark_ratio_values)
        #zarr.save("/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_zarr/dark_assets/" + f'dark_assets_block_{str(block.height).zfill(6)}.zarr', dark_assets_arr)
        zarr.save("/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_zarr/current_assets/" + f'current_assets_values_block_{str(block.height).zfill(6)}.zarr', current_assets_values)
        zarr.save("/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_zarr/current_assets_index/" + f'current_assets_index_block_{str(block.height).zfill(6)}.zarr', current_assets_index)
        zarr.save("/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_zarr/dark_assets_index/" + f'dark_ratio_index_block_{str(block.height).zfill(6)}.zarr', dark_ratio_index)
        # np.save("/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data/dark_ratio/" + f'dark_ratio_block_{str(block.height).zfill(6)}.npy', np.array(dict(dark_ratio)))
        # np.save("/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data/dark_assets/" + f'dark_assets_block_{str(block.height).zfill(6)}.npy', np.array(dict(dark_assets)))
        # np.save("/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data/current_assets/" + f'current_assets_block_{str(block.height).zfill(6)}.npy', np.array(dict(current_assets)))

        # if not trx.is_coinbase:
        #     loop = 0
        #     print(f'----------Results for block:{block.height}----------')
        #     print(f'current_assets after block has finished:{block.height}')
        #     i = 0
        #     for k, v in current_assets.items():
        #         print(f'{k}:{format_e(v)}, ', end='')
        #         if i == 9:
        #             print('\n')
        #             i = 0
        #         i+=1
                
        #     print('\n')

        #     print(f'dark_assets after block has finished:{block.height}')
        #     i = 0
        #     for k, v in dark_assets.items():
        #         print(f'{k}:{format_e(v)}, ', end='')
        #         if i == 9:
        #             print('\n')
        #             i = 0
        #         i+=1
            
        #     print('\n')

        #     print(f'dark_ratio after block has finished:{block.height}')
        #     i = 0
        #     for k, v in dark_ratio.items():
        #         print(f'{k}:{format_e(v)}, ', end='')
        #         if i == 9:
        #             print('\n')
        #             i = 0
        #         i+=1
            
        #     print('\n')

        #     print(f'g.graph size:{g.graph_size()}')
        #     g.print_graph()
        #     print(f'current_assets size:{len(current_assets)}')
        #     print(f'dark_ratio size:{len(dark_ratio)}')
        #     print(f'size of clustered_nodes:{len(clustered_nodes)}')
        #     #zarr.save(options.output_folder + f'dark_ratio_block_{block.height}.zarr', dark_ratio_arr) 
            
        #     #print(f'clustered_nodes:{clustered_nodes}')
        #     print(block)
        #     print("____________________________________________________________________________________________________________________________________________")
        
        # # loop += 1

        # if block.height == 49996:
        #     break

    chrono.print(message="took", tic="last")


