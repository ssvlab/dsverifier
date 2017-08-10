function aq=fwl(a,l)

% 
% aq=fwl(a,l)
% 
% Obtains the FWL format of polynomial a with l fractional bits.
%     
% Iury Bessa
% Setembro 9, 2016
% Manaus

aq=(2^(-1*l))*floor(a*(2^l));
end