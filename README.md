## BagNet release for NGSPICE
This repo contains the demo code for demonstraing the algorithm of BagNet in ngspice environment.
BagNet results have been demonstrated in [ICCAD 2019](https://ieeexplore.ieee.org/document/8942062) and [DAC 2019(https://ieeexplore.ieee.org/document/8807032).

BagNet demo on BAG (Berkeley Analag Generator) with post layout simulations is comming soon.

# setup

Clone the repo and update the submodules.

```
git clone
cd repo
git submodule update --init --recursive
```

# NGSpice installation
NGspice 2.7 needs to be installed separately, via this [installation link](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/27/). Page 607 of the pdf manual on the website has instructions on how to install. Note that you might need to remove some of the flags to get it to install correctly for your machine.

# Code structure
This repo contains two submodule:

* (bb_envs)[https://github.com/kouroshHakha/bb_envs.git]
Contains example implementations of black-box environments used for optimization. For more info go to the link and read the documentation.
* (deep_ckt)[https://github.com/kouroshHakha/bag_deep_ckt.git]
Contains the submodules for black-box env framework and the algorithms used for circuit optimization.

# running experiements readily

`command.sh` contains some commands that can reproduce the results of ICCAD paper for the two stage opamp optimization problem. You can comment/un-comment the sections that you deem necessary.

```
./commands.sh
```

# running custom experiments
For custom experiments, a yaml file containing algorithm specifications should be passed to top level script located at `./deep_ckt/efficient_ga/run_scripts/main.py`.
For example:

```
./run.sh ./deep_ckt/efficient_ga/run_scripts/main.py specs/examples/cs_amp.yaml
```
