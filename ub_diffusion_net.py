#!/usr/bin/env python3


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

    options.output_active_folder = options.output_folder + f'active_black_nodes_{options.frequency}/'
    if not os.path.exists(options.output_active_folder):
        os.mkdir(options.output_active_folder)

    options.output_csv = f"{options.output_folder}/diffusion_net_{options.frequency}.csv"


    # atm output and inputs are in the same folder
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

def catch(address, am):
    d = am._AddressMapper__offsets
    try:
        add = am.chain.address_from_string(address)
        num = add.address_num
        typ = add.type
        return num + d[typ]
    except:
        return np.nan

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

    col_names = ['date', 'no_old_black_user', 'no_new_black_user', 'no_active_black_user', 'no_cum_black_user', 'black2new_no_trx', 'black2new_value', 'black2new_no_link', 'black2black_no_trx', 'black2black_value', 'black2black_no_link', 'not_black_link']
    csv_fout = DictWriter(open(options.output_csv, "w") , fieldnames=col_names)

    csv_fout.writeheader()

    # set of black users
    # node type should change to str because in graphs nodes ids are str
    clust_is_black_ground_set = set([str(i) for i in range(len(clust_is_black_ground)) if clust_is_black_ground[i]])
    clust_is_black_set_cum = set([])
    clust_is_black_cum = np.zeros(len(clust_is_black_ground), dtype=bool)
    clust_is_black_when = np.zeros(len(clust_is_black_ground), dtype='U10')

    chrono.print(message="init")

    print(f"[CALC] starts black bitcoin diffusion...")


    # RUN ON ALL NETWORKS
    for network in network_list:
        network_date = network[:-12]
        _ = {}
        # load network
        g = nx.read_graphml(options.network_folder + network)
        # nodes in the graphs are users after am.clusters
        # assume node id is index of cluster_is_black
        new_black_nodes = set([])  # new black nodes found this week
        # black nodes we know already from ground truth and updates
        old_black_nodes = clust_is_black_ground_set.intersection(g.nodes)  # black nodes we know already

        black2new_no_trx = 0
        black2new_value = 0
        black2new_link = 0
        black2black_no_trx = 0
        black2black_value = 0
        black2black_link = 0
        for black_node in old_black_nodes:
            # check out-neighbours of black node
            for e in g[black_node]:
                # if not already black, track it
                if e not in old_black_nodes:
                    new_black_nodes.update([e])
                    black2new_no_trx += g[black_node][e]['n_tx']
                    black2new_value += g[black_node][e]['value']
                    black2new_link += 1
                else:
                    black2black_no_trx += g[black_node][e]['n_tx']
                    black2black_value += g[black_node][e]['value']
                    black2black_link += 1
                if clust_is_black_when[int(e)] == '':
                    clust_is_black_when[int(e)] = network_date

        # update black nodes and write them
        clust_is_black_ground_set.update(new_black_nodes)
        clust_is_black_set_cum.update(old_black_nodes.union(new_black_nodes))
        # clust_is_black_cum[list(clust_is_black_set_cum)] = True
        clust_is_black_active_set = old_black_nodes.union(new_black_nodes)


        _['date'] = network_date
        _['no_old_black_user'] = len(old_black_nodes)
        _['no_new_black_user'] = len(new_black_nodes)
        _['no_active_black_user'] = _["no_old_black_user"] + _["no_new_black_user"]
        _['no_cum_black_user'] = len(clust_is_black_set_cum)
        _['black2new_no_trx'] = black2new_no_trx
        _['black2new_value'] = black2new_value
        _['black2new_no_link'] = black2new_link
        _['black2black_no_trx'] = black2black_no_trx
        _['black2black_value'] = black2black_value
        _['black2black_no_link'] = black2black_link
        _['not_black_link'] = g.size() - (black2new_link + black2black_link)

        csv_fout.writerow(_)

        with open(options.output_active_folder + network_date + '.pkl', "wb") as pfile:
            pkl.dump([old_black_nodes, clust_is_black_active_set], pfile)

    zarr.save(options.output_folder + f'cluster_is_black_final_{options.frequency}.zarr', clust_is_black_when)

    chrono.print(message="took", tic="last")



   # addr_no_input_tx = zarr.load(f"{options.data_in_folder}/cluster_no_input_tx.zarr")
   # addr_no_output_tx = zarr.load(f"{options.data_in_folder}/cluster_no_output_tx.zarr")


