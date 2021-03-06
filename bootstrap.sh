#!/usr/bin/env bash

# Upgrade the package installer
sudo apt-get -y upgrade

# Update package manager
sudo apt-get update

# Common Packages
sudo apt-get install -y build-essential tcl tk --no-install-recommends
sudo apt-get install -y libpq-dev vim --no-install-recommends
sudo apt-get install -y software-properties-common python-software-properties

# Python 3.6 and packages
sudo apt-get -y install software-properties-common --no-install-recommends
sudo apt-add-repository universe
sudo apt-get update
sudo apt-get -y install python3.6 --no-install-recommends
sudo apt-get -y install python-dev --no-install-recommends
sudo apt-get -y install python-pip --no-install-recommends
sudo apt-get -y install python-setuptools --no-install-recommends
sudo pip3 install --upgrade pip
sudo pip3 install numpy
sudo pip3 install matplotlib



# Access Control Settings
sudo chmod 777 ./start
