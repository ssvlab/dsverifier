function fxp_num = fxp_div(adiv,bdiv,wl)
%
% Function to perform a fixed point division out = a / b
%
% Function: [fxp_num]=fxp_div(adiv,bdiv,wl)
%
% where:
% adiv is fixed point input
% bdiv is fixed point input
% wl is fractional word lenght
%
% return div result out
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

fxp_adiv= fxp_rounding(adiv,wl);
fxp_bdiv= fxp_rounding(bdiv,wl);

fxp_num = fxp_rounding(fxp_adiv/fxp_bdiv, wl);

end
