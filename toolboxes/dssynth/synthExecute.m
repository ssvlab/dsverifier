function synthExecute(type)
% Invoke DSSynth Tool in order to synthesis the controller for a given plant
%
% Function: synthExecute(type)
%
% type: 'ss' for state-space or 'tf' for transfer-function;
%
% Author: Lennon Chaves
% 
% March 2017
%

current = pwd;

user = userpath;

if strfind(user,'/Documents/MATLAB') %default folder installation
    install_folder = [user '/Add-Ons/Toolboxes/DSSynth/code'];
else
    install_folder = [user '/Toolboxes/DSSynth/code'];
end

if strcmp(type, 'ss')
  filesystem = 'system.ss';
elseif strcmp (type, 'tf')
  filesystem = 'system.c';
end

cd(install_folder);

%creating temporary directory

cd 'dssynth-tool'
mkdir 'benchmarks'
cd 'benchmarks'
mkdir 'system'
cd 'system'

benchmark = [current '/' filesystem];

copyfile(benchmark);

benchmark_path = pwd;

cd(current);

%running the synthesis

if (strcmp(type,'tf'))
runner = [install_folder '/dssynth-tool/compiler-transfer-function.sh'];
elseif (strcmp(type,'ss'))
runner = [install_folder '/dssynth-tool/compiler-state-space.sh'];
end

command = ['sh ' runner];
system(command);

if (strcmp(type, 'tf'))
logfile = [benchmark_path '/system_bound_simple.log'];
elseif (strcmp(type,'ss'))
logfile = [benchmark_path '/system_completeness-threshold-ss.log'];
end

copyfile(logfile);

tmp_directory = [install_folder '/dssynth-tool/benchmarks'];
rmdir(tmp_directory,'s');

end
