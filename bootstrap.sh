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
sudo pip install --upgrade pip
sudo pip install numpy
sudo pip install scipy
sudo pip install scikit-learn

# Brew and Octave
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
brew install octave

# Access Control Settings
sudo chmod 777 ./start
