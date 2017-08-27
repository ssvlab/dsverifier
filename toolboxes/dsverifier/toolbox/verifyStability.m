function verifyStability(system, intBits, fracBits, rangeMax, rangeMin, realization, varargin)
%
% Checks stability property violation for digital systems using a bounded model checking tool.
% Function: verifyStability(system, intBits, fracBits, rangeMax, rangeMin, realization)
%
% Where
%   system: represents a digital system represented in transfer-function;
%   intBits: represents the integer bits part;
%   fracBits: represents the fractionary bits part;
%   rangeMax: represents the maximum dynamical range;
%   rangeMin: represents the minimum dynamical range;
%   realization: set the realization for the Digital-System (DFI, DFII, TDFII, DDFI, DDFII, and TDDFII);
%
%  The 'system' must be structed as follow:
%  system = tf(den,num,ts): transfer function representation (den - denominator, num - numerator, ts - sample time);
%  or system = c2d(sys,ts): if the digital system should be discretized
%  
%  See also TF and C2D.
%
% For Delta Verification, the delta operator must be informed as:
%
% verifyStability(system, intBits, fracBits, rangeMax, rangeMin, realization, delta)
%
% Where
%   delta: the delta operator for a delta realization (DDFI, DDFII or TDDFII)
%
% Another usage form is adding other parameters (optional parameters) as follow:
%
% verifyStability(system, intBits, fracBits, rangeMax, rangeMin, realization, bmc, solver, ovmode, rmode, emode, timeout)
%
% For delta realization:
%
% verifyStability(system, intBits, fracBits, rangeMax, rangeMin, realization, delta, bmc, solver, ovmode, rmode, emode, timeout)
%
% Where
%  bmc: set the BMC back-end for DSVerifier (ESBMC or CBMC);
%  solver: use the specified solver in BMC back-end which could be 'Z3', 'boolector', 'yices', 'cvc4', 'minisat';
%  ovmode: set the overflow mode which could be 'WRAPAROUND' or 'SATURATE';
%  rmode: set the rounding mode which could be 'ROUNDING', 'FLOOR', or 'CEIL';
%  emode: set the error mode which could be 'ABSOLUTE' or 'RELATIVE';
%  timeout: configure time limit, integer followed by {s,m,h} (for ESBMC only).
%
% Example of usage:
%  num = [1 0.5 1];
%  den = [1 -1.5 -3];
%  system = tf(den,num,1);
%
%  verifyStability(system,2,10,1,-1,'DFI',0.25);
%  VERIFICATION FAILED!
%
% Author: Lennon Chaves
% Federal University of Amazonas
% December 2016
%

global property;
%setting the DSVERIFIER_HOME
verificationSetup();

%generating struct with sytem and its implementations.
digitalSystem.system = system;
digitalSystem.impl.frac_bits = fracBits;
digitalSystem.impl.int_bits = intBits;
digitalSystem.range.max = rangeMax;
digitalSystem.range.min = rangeMin;

delta = 0;
if strcmp(realization, 'DDFI') || strcmp(realization, 'DDFII') || strcmp(realization, 'TDDFII')
    if nargin >= 7
    delta = varargin{1};
    end
end

digitalSystem.delta = delta;

%translate to ANSI-C file
verificationParser(digitalSystem,'tf',0,realization);

%verifying using DSVerifier command-line
property = 'STABILITY';
extra_param = getExtraParams(nargin,varargin,'tf', property, realization);
command_line = [' --property ' property ' --realization ' realization extra_param];
verificationExecution(command_line,'tf');

%report the verification
output = verificationReport('output.out','tf');
disp(output);

end

