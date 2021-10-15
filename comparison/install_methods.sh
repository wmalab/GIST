#! /bin/bash
date
hostname
source activate env_cmp

# pastis
pip install pastis

# Shrec3d-extened
git clone https://github.com/jbmorlot/ShRec-Exented.git

# GEM
git clone https://github.com/GuangxiangZhu/GEM.git

# ChromSDE
wget https://www.comp.nus.edu.sg/~bioinfo/ChromSDE/ChromSDE_program2.2.zip
mkdir ChromSDE
unzip ChromSDE_program2.2.zip -d ./ChromSDE
