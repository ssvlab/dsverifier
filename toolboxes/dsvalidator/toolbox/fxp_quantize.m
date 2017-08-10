function num_quantized = fxp_quantize (value, k, l)
%
% Function to perform a fixed point quantization.
%
% Function: [num_quantized] = fxp_quantize(value, k, l)
%
% where:
% value is a input value to be quantized
% k is integer bits lenght
% l is fractional bits lenght
%
% return the parameter 'value' quantized according to implementation values and overflow mode.
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

global overflow_mode;

if (strcmp(overflow_mode,'wrap'))
    num_quantized = mode_wrap(value, k, l);
elseif (strcmp(overflow_mode,'saturate'))
    num_quantized = mode_saturate(value, k, l);
else
    num_quantized = value;
end

end
