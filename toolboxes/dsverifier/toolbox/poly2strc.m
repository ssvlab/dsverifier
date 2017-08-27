function output = poly2strc( poly )
% Translate a polynomial representation to a string representation
%
% Function: output = POLY2STRC( poly )
%
%  poly: the polynomial format, in a MATLAB vector.
%
% Author: Lennon Chaves
% Federal University of Amazonas
% October 2016
%

n = length(poly);
tmp = strings(1,n*2-1);
j = 1;

for i=1:length(tmp)
if mod(i,2) == 0  
tmp(i) = ',';
else
tmp(i) = poly(j);
j = j + 1;
end
end

output = cellstr(tmp);
output = cell2mat(output);

end

