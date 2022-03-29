
### two stage evolutionary algorithm
#./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/iccad/two_stage_evo_custom_ea.yaml

### two stage oracle algorithm (takes long)
#./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/iccad/two_stage_oracle_custom_ea.yaml

### two stage bagnet algorithm
# ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/iccad/two_stage_bagnet_custom_ea_dropout_multi_sampling.yaml

### to plot figures on top of eachother use the following script
#./run.sh deep_ckt/efficient_ga/run_scripts/plot.py --log_paths --legend


CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_0  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_torch_fc20.yaml --seed 0
CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_10 ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_torch_fc20.yaml --seed 10
CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_20 ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_torch_fc20.yaml --seed 20

CUDA_VISIBLE_DEVICES=1 NGSPICE_TMP_DIR=/tmp/bagnet_0  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_randinit.yaml --seed 0
CUDA_VISIBLE_DEVICES=2 NGSPICE_TMP_DIR=/tmp/bagnet_10 ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_randinit.yaml --seed 10
CUDA_VISIBLE_DEVICES=3 NGSPICE_TMP_DIR=/tmp/bagnet_20 ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_randinit.yaml --seed 20

CUDA_VISIBLE_DEVICES=4 NGSPICE_TMP_DIR=/tmp/bagnet_0  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn.yaml --seed 0
CUDA_VISIBLE_DEVICES=5 NGSPICE_TMP_DIR=/tmp/bagnet_10 ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn.yaml --seed 10
CUDA_VISIBLE_DEVICES=6 NGSPICE_TMP_DIR=/tmp/bagnet_20 ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn.yaml --seed 20


CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_0  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_torch_fc20.yaml --seed 0


CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_0  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2.yaml --seed 0
CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_10  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2.yaml --seed 10
CUDA_VISIBLE_DEVICES=1 NGSPICE_TMP_DIR=/tmp/bagnet_20  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2.yaml --seed 20


CUDA_VISIBLE_DEVICES=2 NGSPICE_TMP_DIR=/tmp/bagnet_0   ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2_randinit.yaml --seed 0
CUDA_VISIBLE_DEVICES=3 NGSPICE_TMP_DIR=/tmp/bagnet_10  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2_randinit.yaml --seed 10
CUDA_VISIBLE_DEVICES=4 NGSPICE_TMP_DIR=/tmp/bagnet_20  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2_randinit.yaml --seed 20

CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_0   ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_torch_fc20_v2.yaml --seed 0
CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_10  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_torch_fc20_v2.yaml --seed 10
CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_20  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_torch_fc20_v2.yaml --seed 20


CUDA_VISIBLE_DEVICES=0 NGSPICE_TMP_DIR=/tmp/bagnet_0   ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2_frozen.yaml --seed 0
CUDA_VISIBLE_DEVICES=1 NGSPICE_TMP_DIR=/tmp/bagnet_10  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2_frozen.yaml --seed 10
CUDA_VISIBLE_DEVICES=2 NGSPICE_TMP_DIR=/tmp/bagnet_20  ./run.sh deep_ckt/efficient_ga/run_scripts/main.py specs/torch_specs/two_stage_bagnet_gnn_v2_frozen.yaml --seed 20