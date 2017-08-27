function extra_param = getExtraParams(n, var, type, property, realization)
% Support function to get the extra params in order to add in command line
% during the execution
%
% Function: extra_param = getExtraParams(n, var, type, property, realization)
%
%  n: number of arguments passed during the verification
%  var: vargin with all extra arguments
%  type: state-space, transfer-function or closed-loop
%  property: desired property to be verified
%  realization: desired realization to be employed
%
% Author: Lennon Chaves
% Federal University of Amazonas
% February 2017
%

global bmc_mode;
bmc_mode = 'CBMC';

extra_param = '';
indice = 1;

if strcmp(type,'tf')

if strcmp(property,'STABILITY') || strcmp(property,'MINIMUM_PHASE')
    nvarIndice = 6;
    indice = 1;
    if strcmp(realization,'DDFI') || strcmp(realization,'DDFII') || strcmp(realization,'TDDFII')
        nvarIndice = 7;
        indice = 2;
    end
elseif strcmp(property,'ERROR')
    nvarIndice = 8;
    indice = 1;
    if strcmp(realization,'DDFI') || strcmp(realization,'DDFII') || strcmp(realization,'TDDFII')
        nvarIndice = 9;
        indice = 2;
    end  
else     
    nvarIndice = 7;
    indice = 1;
    if strcmp(realization,'DDFI') || strcmp(realization,'DDFII') || strcmp(realization,'TDDFII')
        nvarIndice = 8;
        indice = 2;
    end    
end
end

if strcmp(type,'cl')
    if strcmp(property,'STABILITY_CLOSED_LOOP')
        nvarIndice = 8;
    elseif strcmp(property,'LIMIT_CYCLE_CLOSED_LOOP')
        nvarIndice = 9;
    elseif strcmp(property,'QUANTIZATION_ERROR_CLOSED_LOOP')
        nvarIndice = 10;
    end
end

if strcmp(type,'rb')
    if strcmp(property,'STABILITY_CLOSED_LOOP')
        nvarIndice = 10;
    elseif strcmp(property,'LIMIT_CYCLE_CLOSED_LOOP')
        nvarIndice = 11;
    elseif strcmp(property,'QUANTIZATION_ERROR_CLOSED_LOOP')
        nvarIndice = 12;
    end
end

if strcmp(type,'ss')
   nvarIndice = 7;
   if strcmp(property,'QUANTIZATION_ERROR')
       nvarIndice = 9;
   end
end

nvarIndice = nvarIndice + 1;

nvar = n;

bmc = ' --bmc ';
solver = ' --solver ';
omode = ' --overflow-mode ';
rmode = ' --rounding-mode ';
emode = ' --error-mode ';
timeout = ' --timeout ';

if nvar >= nvarIndice
if length(var{indice}) > 0
bmc_mode = var{indice};
extra_param = [extra_param bmc var{indice}];
end
end

if nvar >= nvarIndice + 1
if length(var{indice+1}) > 0
extra_param = [extra_param solver var{indice+1}];
end
end

if nvar >= nvarIndice + 2
if length(var{indice+2}) > 0
extra_param = [extra_param omode var{indice+2}];
end
end

if nvar >= nvarIndice + 3
if length(var{indice+3}) > 0
extra_param = [extra_param rmode var{indice+3}];
end
end

if nvar >= nvarIndice + 4
if length(var{indice+4}) > 0
extra_param = [extra_param emode var{indice+4}];
end
end

if nvar >= nvarIndice + 5
if length(var{indice+5}) > 0
extra_param = [extra_param timeout num2str(var{indice+5})];
end
end

end
