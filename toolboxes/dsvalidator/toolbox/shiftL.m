function [z_out] = shiftL(zIn,z, nZ)
%
% Function developed to shift left during the realizations (DFI,DFII and TDFII)
%
% Function: [z_out] = shiftL(zIn,z, nZ)
%
% The main objective of this function is flip the vector in the right-left
% direction and including a value in the end of vector in each interaction of the realization.
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

for i=1:(nZ-1)
    z(i) = z(i+1);
end

z(nZ) = zIn;
z_out = z;

end
