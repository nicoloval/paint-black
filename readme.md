
# code
- `uniform_black.py` from currency heuristics and output dir,
	writes the `cluster_is_black.csv` df, and create a ground truth
	modified df with cluster and addresses id
- `run_uniform_black.sh` bash code to run `uniform_black.py` on all settings
- `uniform_black_sanity_check.ipynb` to check that uniform black results are in line with previous papers results 
- `ub_diffusion_block` analysis of black diffusion per block **in progress**
- `ub_diffusion_net` analysis of black diffusion via transaction networks.
	outputs:
	* csv file `diffusion_net_{frequency}.csv` in `uniform_black/{currency}_data/`:
	each row contains data on a specific network
	* zarr file `cluster_is_black_final_{frequency}.zarr` in `uniform_black/{currency}_data`:
	contians a numpy array. indexes are cluster ids, value is empty string if they were never black,
	they contain the first date they appear as black otherwise.
	* directory `black_active_nodes_{frequency}/` contains a pkl file per freuqnecy date of the active 
	black nodes on the corresponding date network, and the old black nodes at the start of each network

# directories
- `uniform_black`: subfolders by heuristic.
	each heur subfolder has: 
	- zarr file `cluster_is_black.zarr`, bool numpy.ndarray
		index is cluster, value is True if cluster has black addresses
		contains the original ground truth extracted from `ground_truth_clust_id.csv`
	- csv file `ground_truth_clust_id.csv`,
		sub df from original `ground_truth_id.csv`, reduced to darknet entities
		and with cluster ids column added
	- csv file `diffusion_net_{frequency}.csv`,
		each row contains data on a specific network
	- zarr file `cluster_is_black_final_{frequency}.zarr`, str numpy.ndarray
		contians a numpy array. indexes are cluster ids,
		value is empty string if they were never black,
		they contain the first date they appear as black otherwise.
	- directory `black_active_nodes_{frequency}/` contains a pkl file per freuqnecy date of the active 
		black nodes on the corresponding date network and the old black nodes at the start of the network:
		`[old_black_nodes, clust_is_black_active_set] = pkl.load(date)`
 
# remarks

- `ub_diffusion_block.py`: 180 h running time
