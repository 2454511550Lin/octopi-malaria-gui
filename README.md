## Environment
Check the nvidia installation by running:
`nvidia-smi`
and
`nvcc --version`

## Install

First time running:
```
git clone https://github.com/2454511550Lin/octopi-malaria-gui.git
cd octopi-malaria-gui
python3 run.py
```

If no error occurs, one can create desktop shortcut

``
bash create_shortcut.sh
``

Now the desktop has `octopi_malaria.sh` and `octopi_malaria_simulation.sh`, one can right click and `run as program`.

## New computer

Please refer to Cephla's [forum](https://forum.squid-imaging.org/t/setting-up-a-new-computer-with-ubuntu/41/2) to install Ubuntu 20.04.6.

### Software

```
wget https://raw.githubusercontent.com/hongquanli/octopi-research/master/software/setup_22.04.sh
chmod +x setup_22.04.sh
./setup_22.04.sh
```

`SciPy` has to be `1.12.0`, otherwise spot detection could have some issues.

### CUDA / Driver

```
wget https://raw.githubusercontent.com/hongquanli/octopi-research/master/software/setup_cuda_22.04.sh
chmod +x setup_cuda_22.04.sh
./setup_cuda_22.04.sh
```


