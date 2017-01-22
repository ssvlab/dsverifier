function reproduceError(path, error, varargin)
%
% Script developed to reproduce error property given a 'path' with all .out counterexamples.
%
% Function: reproduceError(path, error)
%
% You need inform the 'path', that is a directory with all counterexamples stored in a .out files and the maximum error value.
%
% The output is the feedback returned from simulation (successful or failed) and a .MAT file with counterexamples stored.
%
% Another usage form is adding other parameters (optional parameters) as follow:
%
% reproduceError(path, error, ovmode, rmode, filename);
%
% Where:
%  ovmode is related to overflow mode and it could be: 'saturate' or 'wrap'. By default is 'wrap';
%  rmode is related to rounding mode and it could be: 'round' or 'floor'. By default is 'round';
%  filename is the name of counterexample .MAT file generated. By default is 'digital_system'.
%
%  Example of usage:
%
%  reproduceError('/home/user/log/overflow/', 0.18);
%
%  reproduceError('/home/user/log/overflow/', 0.18,'saturate','floor','counterexample_file');
%
% Lennon Chaves
% January, 2017
% Manaus, Brazil

global max_error;
ovmode = '';
rmode = '';
filename = '';

nvar = nargin;
var = varargin;

if nvar >= 3
if length(var{1}) > 0
 ovmode = var{1};
end
end

if nvar >= 4
if length(var{2}) > 0
 rmode = var{2};
end
end

if nvar >= 5
if length(var{3}) > 0
 filename = var{3};
end
end

property = 'e';
max_error = error;
dsv_validation(path, property, ovmode, rmode, filename);

end
