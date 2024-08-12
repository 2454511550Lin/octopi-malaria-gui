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
```

The main entry is `python3 run.py`. But one can also run `python3 single_fov.py` to test the the processing of a single FOV. Please configure the path to the data folder in `single_fov.py` in main function line `91 and 92`. The sinle fov file, including left and right half, fluorescent and should be places under sample_inputs folder.

If no error occurs, one can create desktop shortcut

``
bash create_shortcut.sh
``

Now the desktop has a octopi malaria icon, one should right click and select "Allow Lunching". Then one can double click the icon to lunch the program.

## Only for new computer and environment

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


