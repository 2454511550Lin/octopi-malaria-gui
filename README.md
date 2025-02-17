## Environment
Check the nvidia installation by running:
`nvidia-smi`
and
`nvcc --version`

## Install

First time running:
```
git clone https://github.com/octopi-project/octopi-malaria-gui.git
cd octopi-malaria-gui
python3 run.py
```

If no error occurs, one can create desktop shortcut

``
bash create_shortcut.sh
``

Now the desktop has a octopi malaria icon, one should right click and select "Allow Lunching". Then one can double click the icon to lunch the program.


## Google Cloud Storage

Optionally, one can use Google Cloud Storage to store the data. The configuration is as follows:
```
pip install google-cloud-storage
```
Then one can configure the bucket name and the path to the service account json key as:
```
echo 'export BUCKET_NAME="your_bucket_name"' >> ~/.bashrc
echo 'export SERVICE_ACCOUNT_JSON_KEY="path_to_your_service_account_json_key"' >> ~/.bashrc
```

## Only for new computer and environment

Please refer to Cephla's [forum](https://forum.squid-imaging.org/t/setting-up-a-new-computer-with-ubuntu/41/2) to install Ubuntu 20.04.6.

### Software

```
wget https://raw.githubusercontent.com/hongquanli/octopi-research/master/software/setup_22.04.sh
chmod +x setup_22.04.sh
./setup_22.04.sh
```

`SciPy` has to be `1.12.0`, otherwise spot detection could have some issues. Which can be installed by `pip install scipy==1.12.0`

### CUDA / Driver

```
wget https://raw.githubusercontent.com/hongquanli/octopi-research/master/software/setup_cuda_22.04.sh
chmod +x setup_cuda_22.04.sh
./setup_cuda_22.04.sh
```


