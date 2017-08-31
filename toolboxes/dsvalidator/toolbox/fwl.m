function aq=fwl(a,l)
%
% Obtains the FWL format of polynomial a with l fractional bits.
%
% aq=fwl(a,l)
%
% Iury Bessa
% Setembro 9, 2016
% Manaus

aq=(2^(-1*l))*round(a*(2^l),l,'significant');
end