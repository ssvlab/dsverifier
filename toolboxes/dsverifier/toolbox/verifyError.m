function verifyError(system, intBits, fracBits, rangeMax, rangeMin, realization, kbound, error, varargin)
%
% Checks error property violation for digital systems using a bounded model checking tool.
% Function: verifyError(system, intBits, fracBits, rangeMax, rangeMin, realization, kbound, error)
%
% Where
%   system: represents a digital system represented in transfer-function;
%   intBits: represents the integer bits part;
%   fracBits: represents the fractionary bits part;
%   rangeMax: represents the maximum dynamical range;
%   rangeMin: represents the minimum dynamical range;
%   realization: set the realization for the Digital-System (DFI, DFII, TDFII, DDFI, DDFII, and TDDFII);
%   kbound: sets the k bound of verification;
%   error: the maximum error allowed.
%
%  The 'system' must be structed as follow:
%  system = tf(den,num,ts): transfer function representation (den - denominator, num - numerator, ts - sample time);
%  or system = c2d(sys,ts): if the digital system should be discretized
%  
%  See also TF and C2D.
%
% For Delta Verification, the delta operator must be informed as:
%
% verifyError(system, intBits, fracBits, rangeMax, rangeMin, realization, kbound, error, delta)
%
% Where
%   delta: the delta operator for a delta realization (DDFI, DDFII or TDDFII)
%
% Another usage form is adding other parameters (optional parameters) as follow:
%
% verifyError(system, intBits, fracBits, rangeMax, rangeMin, realization, kbound, error, bmc, solver, ovmode, rmode, emode, timeout)
%
% For delta realization:
%
% verifyError(system, intBits, fracBits, rangeMax, rangeMin, realization, kbound, error, delta, bmc, solver, ovmode, rmode, emode, timeout)
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
%  verifyError(system,2,10,1,-1,'DFI',0.25);
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
    if nargin >= 9
    delta = varargin{1};
    end
end

digitalSystem.delta = delta;

%translate to ANSI-C file
verificationParser(digitalSystem,'tf',error,realization);

%verifying using DSVerifier command-line
property = 'ERROR';
extra_param = getExtraParams(nargin,varargin,'tf',property,realization);
command_line = [' --property ' property ' --realization ' realization ' --x-size ' num2str(kbound) extra_param];
verificationExecution(command_line,'tf');

%report the verification
output = verificationReport('output.out','tf');
disp(output);

end

