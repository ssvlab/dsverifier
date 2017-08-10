function controller = synthesize(system, intBits, fracBits, rangeMax, rangeMin)
% Synthesize Digital Controllers given a plant in transfer-function or state-space, numerical representation (integer and fractional bits), and the dynamic ranges (maximum and minimum ranges).
%
% Function: controller = SYNTHESIZE(system, intBits, fracBits, rangeMax, rangeMin)
%
%  system: plant designed as transfer-function or state-space representation (See also TF and SS.)
%  intBits: represents the integer bits part;
%  fracBits: represents the fractionary bits part;
%  rangeMax: represents the maximum dynamical range;
%  rangeMin: represents the minimum dynamical range;
%
%
%  Example of Usage:
%
%   plant = tf([1.0 0.865],(1.76 -0.9876),0.02);
%
%   controller = synthesize(plant, 8, 8, 1, -1);
%
% Author: Lennon Chaves
% 
% March 2017
%

 type = class(system);
 path = pwd;

%Step 1: Setup  
  ds = synthSetup(system, intBits, fracBits, rangeMax, rangeMin, type);
%Step 2: Parse
  synthParse(ds,type);
%Step 3: Execute
  synthExecute(type);
%Step 4: Extract
  synthExtract(path, type);
%Step 5: Report
  controller = synthReport(ds,type);

end
