# main scripts

1. `ub_ground_truth.py` darknet addresses -> black clusters 
	input:
	- `{DIR_PARSED/bitcoin_darknet/ground_truth_id.csv}` ground trught dataframe 
	- `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` clustering data
	- `{DIR_PARSED}/{options.currency}.cfg` blockchain data
	output:
	in `{options.output_folder}/heur_{options.heuristic}_data/`
	* `cluster_is_black_ground_truth.zarr`:
		index is cluster, value is bool: True if black ground truth, False otherwise
	* `ground_truth_clust_id.csv`:
		dataframe to relate entities, btc addresses and cluster ids
2. `ub_diffusion_block` run diffusion block by block and save first time black occurance per cluster
	input:
	- `{options.black_data_folder}/cluster_is_black_ground_truth.zarr` ground truth clusters from `ub_ground_truth.py` 
	- `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` clustering data
	- `{DIR_PARSED}/{options.currency}.cfg` blockchain data
	output:
	* `{options.output_folder}/cluster_is_black_when_block.zarr` index is cluster id, value is int block when the cluster became black
3. `ub_analysis_block.py`: analysis block per block of black diffusion and record results in a dataframe
	TODO: finish writing the code
	input:
	* `{options.black_data_folder}/cluster_is_black_when_block.zarr` index is cluster id, value is int block when the cluster became black
	- `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` clustering data
	- `{DIR_PARSED}/{options.currency}.cfg` blockchain data
	output:
	- `{options.output_folder}/diffusion_analysis_block.csv` csv containing a pd.dataframe of results on the diffusion

## code pipeline

- `run_ground_truth.sh`: run `ub_ground_truth.py` for all heuristics
- `run_ub_diffusion_block.sh`: run `ub-diffusion_block` for all heuristics

## jupyter notebook for check and visualisation
`uniform_black_sanity_check.ipynb` to check that uniform black results are in line with previous papers results 

# remarks

- `ub_diffusion_block.py`: 110 h running time
- `ub_diffusion_net.py`: 58h
