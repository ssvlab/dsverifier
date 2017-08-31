function fxp_num = fxp_sub(asub, bsub, wl)
%
% Function to perform a fixed point subtraction out = a - b
%
% Function: [fxp_num]=fxp_add(aadd,badd,wl)
%
% where:
% asub is fixed point input
% bsub is fixed point input
% wl is fractional word lenght
%
% return result of subtracting the inputs
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

fxp_asub= fxp_rounding(asub,wl);
fxp_bsub= fxp_rounding(bsub,wl);

fxp_num = fxp_rounding(fxp_asub - fxp_bsub, wl);

end
