addpath('../toolbox');

path = pwd;

directory = [path '/regression/Limit'];
dsv_validation(directory,'lc','','','ds_limit');
directory = [path '/regression/Overflow'];
dsv_validation(directory,'o','','','ds_overflow');
directory = [path '/regression/Minimum'];
dsv_validation(directory,'m','','','ds_minimum');
directory = [path '/regression/Stability'];
dsv_validation(directory,'s','','','ds_stability');