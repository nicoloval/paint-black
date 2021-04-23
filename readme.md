
# code
- `uniform_black.py` from currency heuristics and output dir,
	writes the `cluster_is_black.csv` df, and create a ground truth
	modified df with cluster and addresses id
- `run_uniform_black.sh` bash code to run `uniform_black.py` on all settings
- `sanity_check` to check that uniform bklack results are in line with previous papers results 

# directories
- `uniform_black`: subfolders by heuristic.
	each heur subfolder has 
	- bool array `cluster_is_black.zarr`,
		index is cluster, value is True if cluster has black addresses
	- pandas dataframe `ground_truth_clust_id.csv`,
		sub df from original `ground_truth_id.csv`, reduced to darknet entities
		and with cluster ids column added
	 
