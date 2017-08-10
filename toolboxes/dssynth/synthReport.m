function controller = synthReport(ds, type)
% Get the controller extracted and translate the system synthesized in MATLAB System.
%
% Function: controller = synthReport(ds, type)
%
%  ds: digital plant with specifications about plant, implementation and dynamical ranges;
%  type: 'ss' for state-space or 'tf' for transfer-function;
%
% Author: Lennon Chaves
% 
% March 2017
%

controller_file = 'controller.out';
fid = fopen(controller_file);

if (strcmp(type,'tf'))
 numLine = fgetl(fid);
 numerator = str2num(numLine);
 denLine = fgetl(fid);
 denominator = str2num(denLine);
 sample = ds.plant.Ts;
 if (isempty(numerator) | isempty(denominator))
   controller = '';
 else
   controller = tf(numerator, denominator, sample);
 end

elseif (strcmp(type,'ss'))
 valueLine = fgetl(fid);
 controller = str2num(valueLine);
 if (isempty(valueLine))
   controller = '';
 end
end

fclose(fid);

delete(controller_file);

end
