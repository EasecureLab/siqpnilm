The SIQP NILM solver

Copyright (c) 2016 by Weicong Kong

The optimisation based NILM solver. This project contains the source code that was use for my IEEE Transactions on Smart Grid journal paper.

If you use my code in your research please cite this paper. Current citation details are:

Title: An Extensible Approach for Non-Intrusive Load Disaggregation with Smart Meter Data
Authors: Weicong Kong, Zhao Yang Dong, Jin Ma, David J. Hill, Junhua Zhao and Fengji Luo
Journal: IEEE Transactions on Smart Grid
Vol/No/Pages: n/a
Accepted: 07-Nov-2016
DOI: n/a


使用anaconda的python3，用pycharm IDE开发调试，删除库上的.idea和 venv文件夹，便于pycharm到不同的Pc上重新配置工程环境



# data
redd_low.h5 文件比较大，需要存在其他地方

# conda 安装环境
## 1
conda create --name nilmtk-env 
## 2
conda config --add channels conda-forge
## 3
conda activate nilmtk-env
## 4
conda install -c nilmtk nilmtk
## 5
conda install gurobi
## 6
conda install pandas=0.25.1
## 
grbgetkey 49523af6-722f-11ea-ab05-020d093b5256
##
GRB_LICENSE_FILE=C:\WINDOWS\system32\gurobi\gurobi.lic
# 注意注意
pandas的版本最好回退到0.25.1
