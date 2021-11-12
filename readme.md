# code pipeline

1. `ub_ground_truth.py` darknet addresses -> black clusters 
	outputs:
	* `cluster_is_black_ground_truth.zarr`
	* `ground_truth_clust_id.csv`
2. `ub_diffusion_block` analysis of black diffusion per block
	outputs:
	* csv file: `diffusion_block.csv` info about diffusion stats per block
	* zarr file: `cluster_is_black_when_block.zarr` index is cluster id, value is int block when the cluster became black
3. `uniform_black_sanity_check.ipynb` to check that uniform black results are in line with previous papers results 

# remarks

- `ub_diffusion_block.py`: 110 h running time
- `ub_diffusion_net.py`: 58h

# obsolete

- `ub_diffusion_net` analysis of black diffusion via transaction networks.
	outputs:
	* csv file `diffusion_net_{frequency}.csv` in `uniform_black/{currency}_data/`:
	each row contains data on a specific network
	* zarr file `cluster_is_black_final_{frequency}.zarr` in `uniform_black/{currency}_data`:
	contians a numpy array. indexes are cluster ids, value is empty string if they were never black,
	they contain the first date they appear as black otherwise.
	* directory `black_active_nodes_{frequency}/` contains a pkl file per freuqnecy date of the active 
	black nodes on the corresponding date network, and the old black nodes at the start of each network

