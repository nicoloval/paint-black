#!/bin/bash

# echo "heur 1"
# python3 uniform_black.py --curr=BTC --heur=1 --output=uniform_black
# python3 ub_diffusion_net.py --curr=BTC --heur=1 --freq=day
echo "heur 2"
#python3 ub_ground_truth.py --curr=BTC --heur=2 --output=uniform_black
python3 ub_diffusion_block.py --curr=BTC --heur=2 --output=uniform_black
