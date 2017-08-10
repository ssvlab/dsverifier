function fxp_num = fxp_mult(amult, bmult, wl)
%
% Function to perform a fixed point multiplication out = amult * bmult
%
% Function: [fxp_num]=fxp_mult(amult,bmult,wl)
%
% where:
% amult is fixed point input
% bmult is fixed point input
% wl is fractional word lenght
%
% Return the product result of fixed point inputs amult and bmult
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

fxp_amult= fxp_rounding(amult,wl);
fxp_bmult= fxp_rounding(bmult,wl);

fxp_num = fxp_rounding(fxp_amult*fxp_bmult, wl);

end
