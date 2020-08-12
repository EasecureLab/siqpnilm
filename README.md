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

## 总览
Anaconda3，配置以下创建的python interpreter及其相关的packages，可能原来的github连接使用的是python2的环境

## 数据集
redd_low.h5 文件比较大，需要存在其他地方

## conda 安装环境
## 官方说明
https://github.com/nilmtk/nilmtk/blob/master/docs/manual/user_guide/install_user.md
## 创建nilmtk-env环境
conda create --name siqpnilm
## 配置安装属性
conda config --add channels conda-forge
## 激活nilmtk-env环境
conda activate siqpnilm
## 安装nilmtk包(自动安装python 3.6.11)
conda install -c nilmtk nilmtk

#  增加gurobi

conda config --add channels http://conda.anaconda.org/gurobi

## 整数规划解析器gurobi
conda install gurobi
## 该学术license需要去申请
grbgetkey 49523af6-722f-11ea-ab05-020d093b5256
## 配置本地的lic文件跟系统环境变量
GRB_LICENSE_FILE=C:\WINDOWS\system32\gurobi\gurobi.lic
## 注意pandas版本，可能会影响数据读取和处理的部分函数（可能不需要）
conda install pandas=0.25.1
## 至此，本地电脑创建了conda的环境
Python 3.6 (nilmtk-env)
C:\ProgramData\Anaconda3\envs\nilmtk-env\python.exe
