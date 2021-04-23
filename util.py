#!/usr/bin/env python3


import time
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict

import sys, os, os.path, socket


SYMBOLS = {
            "BTC": "bitcoin",
            "LTC": "litecoin",
            "DOGE": "dogecoin",
            "BCH": "bitcoin_cash",
            "BSV": "bitcoin_sv",
            "LCH": "litecoin_cash",
            "FTC": "feathercoin",
            "MONA": "monacoin",
            "PPC": "peercoin",
            "NMC": "namecoin"
          }

darknet = [
            'EvolutionMarket',
            'SilkRoadMarketplace',
            'SilkRoad2Market',
            'AlphaBayMarket',
            'NucleusMarket',
            'AbraxasMarket',
            'PandoraOpenMarket',
            'SheepMarketplace',
            'BlackBankMarket',
            'MiddleEarthMarketplace',
            'BlueSkyMarketplace',
            'CannabisRoadMarket',
            'BabylonMarket',
            'GreenRoadMarket'
        ]

if socket.gethostname() in {'abacus', 'later'}:
    DIR_BCHAIN="/mnt/hdd_data/blockchain_data/"
    DIR_PARSED="/mnt/hdd_data/blockchain_parsed/"
#elif socket.gethostname() == 'later':
#    DIR_BCHAIN="/mnt/hdd_data/blockchain_data/"
#    DIR_PARSED="/mnt/hdd_data/blockchain_parsed/"


class SimpleChrono:
    def __init__(self):
        n=time.perf_counter()
        self.tics = {0: n, "last":n}


    def add_tic(self, key):
        self.tics[key] = time.perf_counter()
        self.tics['last'] = self.tics[key]

    def elapsed(self, tic=0, format=None):
        n = time.perf_counter()
        if format == None:
            return "{:.3f}".format(n - self.tics[tic])
        else:
            return time.strftime(format, time.gmtime(n - self.tics[tic]))
        self.tics['last'] = n

    def print(self, tic=0, message=None, format=None):
        s = self.elapsed(tic, format)
        print(f"[CHRONO] {message}: {s}")

