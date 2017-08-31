function validation(path, property, ovmode, rmode, filename)
%
% Script to run all steps to validate counterexamples
%
% Function: validation(path, property, ovmode, rmode, filename)
%
% To start the validation, the folder with all counterexamples should be
% informed.
%
% You need inform the path, e.g. 'home/user/dsv/counterexamples'
% The output files are generated at 'outputs' folder.
%
% property is the property of digital system to be validate:
%     'lc' is for limit cycle property
%     's' is for stability property
%     'm' is for minimum phase property
%     'o' is for overflow property
%     'scl' is for stability in closed-loop system
%     'sss' is for stability in state-space system
%     'sso' is for observability in state-space system
%     'ssc' is for controllability in state-space system
%
% ovmode is the overflow mode. The values could be:
%     'saturate' for saturate overflow
%     'wrap' for wrap around overflow
%     By default, the value is 'wrap'.
%
% rmode is the rounding mode. The values could be:
%     'round' to use round as rounded mode
%     'floor' to use floor as rounded mode
%      By default, the value is 'round'
%
% filename: the name of .MAT file generated as result from validation.
%      By default, the value is 'digital_system'
%
% Example of usage:
%  validation('/home/user/log/limit_cycle/','lc','wrap','round','ds_limit');
%  validation('/home/user/log/limit_cycle/','lc','saturate','floor','ds_limit');
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

global overflow_mode;
global round_mode;

overflow_mode = ovmode;
round_mode = rmode;

if isunix
    disp('Running Automatic Validation...');
else
    disp('Operating System not Supported for Automatic Validation Scripts!');
    return
end


if (isempty(property))
    disp('Error. The parameter "property" should be "m","lc","o","s","sss", "sso", "ssc" or "scl"!');
    return
elseif (strcmp(property,'m') || strcmp(property,'o') || strcmp(property,'lc') || strcmp(property,'s') || strcmp(property,'e') || strcmp(property,'scl'))
else
    disp('Error. The parameter "property" should be "m","lc","o","s","sss", "sso", "sscor "scl"!');
    return
end

if (isempty(path))
    disp('Error. You should inform a path of counterexamples!');
    return
end

if ~(strcmp(overflow_mode,'wrap') || strcmp(overflow_mode,'saturate'))
    overflow_mode = 'wrap';
end
if ~(strcmp(round_mode,'round') || strcmp(round_mode,'floor'))
    round_mode = 'round';
end

%function to extract the parameters from counterexamples output.
validationExtraction(path, property);

%parsing the paramaters to variables workspace
digital_system = validationParser(property);

%simulation automatically of all counterexamples

for i=1:length(digital_system)
    [output, time_execution] = validationSimulation(digital_system(i), property);
    digital_system(i).output.output_simulation = output;
    digital_system(i).output.time_execution = time_execution;
end

%comparison between matlab and dsverifier outputs
for i=1:length(digital_system)
    status = validationComparison(digital_system(i), property);
    digital_system(i).output.status = status;
    digital_system(i).status = status;
end

%report for user
validationReport(digital_system);

%saving all variables created in a file .MAT in order to be used later.

fname = 'counterexamples.mat';
varname = 'digital_system';

if length(filename) > 0
    fname = [filename '.mat'];
end

save (fname,varname);

end
