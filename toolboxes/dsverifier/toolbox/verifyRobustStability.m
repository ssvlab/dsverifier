function verifyRobustStability(controller, plant, uncertainty_b, uncertainty_a, intBits, fracBits, rangeMax, rangeMin, realization, connectionMode, varargin)
%
% Checks stability property violation for closed-loop digital systems using a bounded model checking tool.
% Function: verifyRobustStability(controller, plant, uncertainty_b, uncertainty_a, intBits, fracBits, rangeMax, rangeMin, realization, connectionMode)
%
% Where
%   controller: represents a controller represented in transfer-function;
%   plant: represents a plant represented in transfer-function;
%   uncertainty_b: represents a vector with the uncertainty in numerator(only for the plant);
%   uncertainty_a: represents a vector with the uncertainty in denominator(only for the plant);
%   intBits: represents the integer bits part;
%   fracBits: represents the fractionary bits part;
%   rangeMax: represents the maximum dynamical range;
%   rangeMin: represents the minimum dynamical range;
%   realization: set the realization for the Digital-System (DFI, DFII, TDFII, DDFI, DDFII, and TDDFII);
%   connectionMode: set the connection mode for the closed-loop system (SERIES or FEEDBACK);
%
%  The 'controller' and 'plant' must be structed as follow:
%  system = tf(den,num,ts): transfer function representation (den - denominator, num - numerator, ts - sample time);
%  or system = c2d(sys,ts): if the digital system should be discretized
%  
%  See also TF and C2D.
%
% Another usage form is adding other parameters (optional parameters) as follow:
%
% verifyRobustStability(controller, plant, uncertainty_b, uncertainty_a, intBits, fracBits, rangeMax, rangeMin, realization, connectionMode, bmc, solver, ovmode, rmode, emode, timeout)
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
%  numplant = [...];
%  denplant = [...];
%  plant = tf(denplant,numplant,ts);
%  numcontrol = [...];
%  dencontrol = [...];
%  controller = tf(dencontrol,numcontrol,ts);
%
%  verifyRobustStability(controller, plant, [0 0 0], [0.5 0.2 0], 2, 10, 1, -1, 'DFI', 'SERIES');
%  VERIFICATION FAILED!
%
% Author: Lennon Chaves
% Federal University of Amazonas
% February 2017
%

global property;
%setting the DSVERIFIER_HOME
verificationSetup();

%generating struct with sytem and its implementations.
digitalSystem.controller = controller;
digitalSystem.plant = plant;
digitalSystem.impl.frac_bits = fracBits;
digitalSystem.impl.int_bits = intBits;
digitalSystem.range.max = rangeMax;
digitalSystem.range.min = rangeMin;
digitalSystem.uncertainty_a = uncertainty_a;
digitalSystem.uncertainty_b = uncertainty_b;

%translate to ANSI-C file
verificationParser(digitalSystem,'rb',0,realization);

%verifying using DSVerifier command-line
property = 'STABILITY_CLOSED_LOOP';
extra_param = getExtraParams(nargin,varargin,'rb',property,realization);
command_line = [' --property ' property ' --realization ' realization ' --connection-mode ' connectionMode extra_param];
verificationExecution(command_line,'rb');

%report the verification
output = verificationReport('output.out','rb');
disp(output);

end

