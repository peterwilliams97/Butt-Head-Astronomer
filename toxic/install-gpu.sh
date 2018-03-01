# This script is designed to work with ubuntu 16.04 LTS

# ensure system is updated and has basic build tools
sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common


# download and install GPU drivers
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-9-0_9.0.176-1_amd64.deb"http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-9-0_9.0.176-1_amd64.deb

sudo dpkg -i cuda-9-0_9.0.176-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda
sudo modprobe nvidia
nvidia-smi


# # install cudnn libraries
# wget "http://files.fast.ai/files/cudnn.tgz" -O "cudnn.tgz"
# tar -zxf cudnn.tgz
# cd cuda
# sudo cp lib64/* /usr/local/cuda/lib64/
# sudo cp include/* /usr/local/cuda/include/

