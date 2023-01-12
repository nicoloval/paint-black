#!/usr/bin/env python3

import networkx as nx
import blocksci
import zarr
import time
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict
import bz2
import matplotlib.pyplot as plt
import math
import sys, os, os.path, socket
import logging
import gc
import multiprocessing
from tqdm.contrib.concurrent import process_map  # or thread_map


from util import SYMBOLS, DIR_PARSED, SimpleChrono

def parse_command_line():
    import sys, optparse

    parser = optparse.OptionParser()

    parser.add_option("--curr", action='store', dest="currency", type='str',
                                              default=None, help="name of the currency")
    parser.add_option("--heur", action='store', dest="heuristic", type='str',
                                                  default=None, help="heuristics to apply")
    #parser.add_option("--overwrite", action='store_true', dest = "overwrite" )
#    parser.add_option("--period",  action='store', dest="period",
#                       default = None , help = "minimum block number to process" )
    parser.add_option("--start", action="store", dest="start_date",
                       default = None, help= "starting date for network creation in YYYY-MM-DD format")
    parser.add_option("--end", action="store", dest="end_date",
                       default = None, help = "ending date for network creation in YYYY-MM-DD format")
    parser.add_option("--freq", action="store", dest="frequency",
                       default = "day", help = "time aggregation of networks - choose between day, week, 2weeks, 4weeks")

    options, args = parser.parse_args()

    options.currency = SYMBOLS[options.currency]

#    options.period = [0,-1] if options.period == None else list( map( int, options.period.split(",")))
#    assert len(options.period) == 2

    switcher = {"day":1, "week":7, "2weeks":14, "4weeks":28}


    options.cluster_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}"
    options.blocks_folder = f"{DIR_PARSED}/{options.currency}/heur_all_data"
    options.networks_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_networks_{options.frequency}"
    options.frequency = switcher[options.frequency]
    
    if not os.path.exists(options.networks_folder):
        os.mkdir(options.networks_folder)


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
        print(f"Total #Addresses: {self.total_addresses}" )
   #     print(self.__counter_addresses)
        
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

    def load_clusters(self, input_folder):
        self.cluster = zarr.load(f"{input_folder}/address_cluster_map.zarr")
        print(f"Total #Clusters: {max(self.cluster)}" )



    def __getitem__(self,addr):
        return self.__offsets[addr.raw_type]+ addr.address_num-1
 

    
def get_all_files(path):
    os.chdir(path)
    
    list_of_files = list()
    
    for file in os.listdir():
        list_of_files.append(path+file)
            
    return list_of_files
            
    
def g_add_trx(inp,outp, value):

    if inp == outp: return

    if g.has_edge(inp,outp):
            g[inp][outp]['value'] = g[inp][outp]['value'] + value
            g[inp][outp]['n_tx']  = g[inp][outp]['n_tx']  + 1
    else:
            g.add_edge(inp,outp)
            g[inp][outp]['value'] = value
            g[inp][outp]['n_tx']  = 1
    
def build_current_dictionary(asset):

    current_assets_zarr = zarr.load(asset)
    
    current_assets_dict = defaultdict(lambda: 0, dict(zip(current_assets_zarr["current_assets_index"], current_assets_zarr["current_assets_values"])))
        
    return current_assets_dict
        
        
def build_dark_dictionary(ratio):

    dark_ratio_zarr = zarr.load(ratio)
    
    dark_ratio_dict = defaultdict(lambda: 0, dict(zip(dark_ratio_zarr["dark_ratio_index"], dark_ratio_zarr["dark_ratio_values"])))
    
    return dark_ratio_dict

def load_dictionaries(date, heur, freq):

    if freq == "day":
        freq = "daily"
    elif freq == "week":
        freq = "weekly"

    current_assets_path = f"/home/user/yassine/bitcoin_darknet/gs_group/grayscale_op_ali/final/heur_{heur}_data_v2/{freq}/current_assets/"
        
    dark_ratio_path = f"/home/user/yassine/bitcoin_darknet/gs_group/grayscale_op_ali/final/heur_{heur}_data_v2/{freq}/dark_ratio/"

    current_assets_file = f"{current_assets_path}current_assets_{date}.zarr"
    
    dark_ratio_file = f"{dark_ratio_path}dark_ratio_{date}.zarr"
    
    if os.path.exists(current_assets_file):
        current_assets_dict = build_current_dictionary(current_assets_file)
    
    if os.path.exists(dark_ratio_file):
        dark_ratios_dict = build_dark_dictionary(dark_ratio_file)
    
    return current_assets_dict,dark_ratios_dict
                    
                
                
def daterange(date1, date2, by=1):
    return [  date1 + timedelta(n) for n in range(0, int((date2 - date1).days)+1, by) ]         

def build_network_with_attributes(date):

    savelocation = f"/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/final/heur_{options.heuristic}_networks_v2_new/{switcherback[options.frequency]}"
    unitsavelocation = f"{savelocation}/{date.strftime('%Y-%m-%d')}.graphml.bz2"

    if os.path.exists(unitsavelocation):
        logging.info(f'process with PID:{os.getpid()} for the date{date} has started but the file exists so it will shutdown')
        return "Already exists"
    
    logging.info(f'process with PID:{os.getpid()} for the date{date} has started')

    start_time = datetime.now()
    
    chrono.add_tic("net")
    g = nx.DiGraph()

    networks_path = f"/local/scratch/exported/blockchain_parsed/bitcoin/heur_{options.heuristic}_networks_{switcherback[options.frequency]}"
    unit_graph_file = f"{networks_path}/{date.strftime('%Y-%m-%d')}.graphml.bz2"
    
    g = nx.read_graphml(unit_graph_file)

    current_assets_dict_full, dark_ratios_dict_full = load_dictionaries(date, options.heuristic, switcherback[options.frequency])

    list_of_nodes = list(g.nodes)
    
    # print(f'list of nodes of size {len(list_of_nodes)}: {list_of_nodes}')
    
    # print(f'current_assets_dict_full[int(k)] : {current_assets_dict_full[77262777]}')

    current_assets_dict_filtered = { k: current_assets_dict_full[int(k)] for k in list_of_nodes }#int(k)
    dark_ratios_dict_filtered = { k: dark_ratios_dict_full[int(k)] for k in list_of_nodes }

    del current_assets_dict_full
    del dark_ratios_dict_full

    # adding graph attributes for {date.strftime('%Y-%m-%d')}
    for node in g.nodes():
        
        nx.set_node_attributes(g, {node:current_assets_dict_filtered[node]}, 'current_assets')
        
        nx.set_node_attributes(g, {node:dark_ratios_dict_filtered[node]}, 'dark_ratio')
        
    del current_assets_dict_filtered
    del dark_ratios_dict_filtered
    gc.collect()

    # print(f'size of g.nodes :{len(g.nodes(data=True))}')
    # print(g.nodes(data=True))

    logging.info(f'process with PID:{os.getpid()} for the date{date} has finished with t={datetime.now() - start_time} building the graph and now is starting assortativity calculation')

    start_time = datetime.now()

    CA_assortativity = nx.numeric_assortativity_coefficient(g, "current_assets")
    DR_assortativity = nx.numeric_assortativity_coefficient(g, "dark_ratio")

    # print(type(CA_assortativity))
    # print(type(DR_assortativity))

    if math.isnan(CA_assortativity):
        CA_assortativity = -2.0
        
    if math.isnan(DR_assortativity):
        DR_assortativity = -2.0
    
    g.graph['CA assortativity'] = CA_assortativity
    g.graph['DR assortativity'] = DR_assortativity

    # print(f'CA assortativity :{CA_assortativity}')
    # print(f'DR assortativity :{DR_assortativity}')

    nx.write_graphml(g, unitsavelocation)

    logging.info(f'process with PID:{os.getpid()} for the date{date} has finished assortativity calculation and saved the results with t={datetime.now() - start_time}')

    tqdm_bar.set_description(f"{switcherback[options.frequency]} of '{date.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)

    return f'process with PID:{os.getpid()} for the date{date} has finished'

     

if __name__ == "__main__":   
    options, args = parse_command_line()

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}

    logging.basicConfig(level=logging.DEBUG, filename=f"logfiles/logfile_networks_builder_final_heur_{options.heuristic}_{switcherback[options.frequency]}/logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

    chrono      = SimpleChrono()
    chain       = blocksci.Blockchain("/local/scratch/exported/blockchain_parsed/bitcoin_old.cfg")

    chrono.print(message="init")

    chrono.add_tic('proc')
    if options.start_date == None:
        start_date = datetime.fromtimestamp(chain.blocks[0].timestamp).date()
    else:
        start_date = datetime.strptime(options.start_date, "%Y-%m-%d").date()
    if options.end_date == None:
        end_date = datetime.fromtimestamp(chain.blocks[-1].timestamp).date()
    else:
        end_date = datetime.strptime(options.end_date, "%Y-%m-%d").date()

    print(f'start_date is set as: {start_date}')
    print(f'end_date is set as: {end_date}')    
    
    datelist = daterange(start_date, end_date, by=options.frequency)
    tqdm_bar = tqdm(datelist, desc="processed files")

    # pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()-5)         # start 50 worker processes

    # multiple_results = pool.map(build_network_with_attributes, tqdm_bar)

    # start = time.time()

    # # for timeunit in tqdm_bar:
    # #     # print(timeunit)
    # #     p.apply_async(build_network_with_attributes, [timeunit])
    
    # p.close()
    # p.join()
    # print("Complete")
    # end = time.time()
    # print('total time (s)= ' + str(end-start))

    r = process_map(build_network_with_attributes, datelist, max_workers=10)

    print(r)
            
        

    print('Process terminated, graphs and attributes created.')
    print(f"Graphs created in {chrono.elapsed('proc', format='%H:%M:%S')}")

        
