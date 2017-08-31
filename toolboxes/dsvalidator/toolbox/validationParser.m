function [system] = validationParser(p)
%
% Script to get the counterexamples parameters and transform them in variables on workspace
% Function>: [system] = validationParser(p)
%
% Where the parameter 'p' is the property to be analyzed: (m) for minimum phase, (s) for stability, (o) for overflow and (lc) for limit cycle.
% (scl) for stability in closed-loop systems, (sss) for stability in state-space format, (ssc) for controllability in state-space format and (sso) for observability in state-space format.
%
% The function validationExtraction must be done before this function!
% dsv_parser only read the extraction file and transform all counterexamples in variables.
%
% The output of this function is the counterexamples extracted in variables
% on MATLAB workspace.
%
% Federal University of Amazonas
% May 15, 2017
% Manaus, Brazil

if (strcmp(p ,'lc') || strcmp(p,'o')) %for overflow and LCO only.
    
    fid = fopen('dsv_counterexample_parameters.txt');
    tline = fgetl(fid);
    tline = fgetl(fid);
    count = 0;
    i = 1;
    while ischar(tline)
        
        switch count
            case 0
                name = tline;
            case 1
                realization = tline;
            case 2
                implementation = str2num(tline);
            case 3
                numerator = str2num(tline);
            case 4
                denominator = str2num(tline);
            case 5
                delta = str2num(tline);
            case 6
                sample_time = str2num(tline);
            case 7
                drange = str2num(tline);
            case 8
                inputs = str2num(tline);
            case 9
                initial_states = str2num(tline);
            case 10
                outputs = str2num(tline);
            case 11
                xsize = str2num(tline);
                
            otherwise
                warning('Unexpected error while reading file.')
        end
        
        count = count + 1;
        tline = fgetl(fid);
        if count == 12
            
            count = 0;
            system(i).test_case = name;
            system(i).sys.a = denominator;
            system(i).sys.b = numerator;
            if (length(numerator) > 0) && (length(denominator) > 0)
            system(i).sys.tf = tf(numerator,denominator,1);
            end
            system(i).impl.int_bits = implementation(1);
            system(i).impl.frac_bits = implementation(2);
            if length(sample_time) > 0
                system(i).impl.sample_time = sample_time;
            end
            if length(drange) > 0
                system(i).impl.range.max = drange(2);
                system(i).impl.range.min = drange(1);
            end
            if length(delta) > 0
                system(i).impl.delta = delta;
            else
                system(i).impl.delta = 0;
            end
            system(i).impl.realization_form = strtrim(realization);
            %if(strcmp(p,'o'))
            %    system(i).output.output_verification = 'Failed';
            %else
            system(i).output.output_verification = outputs;
            %end
            if length(initial_states) > 0
                system(i).inputs.initial_states = initial_states;
            end
            system(i).inputs.const_inputs = inputs;
            if length(xsize) > 0
                system(i).impl.x_size = xsize;
            else
                system(i).impl.x_size = length(inputs);
            end
            
            i = i + 1;
            
        end
    end
    
    fclose(fid);
    
else %for all the other properties in transfer-function format.
    
    fid = fopen('dsv_counterexample_parameters.txt');
    tline = fgetl(fid);
    tline = fgetl(fid);
    count = 0;
    i = 1;
    while ischar(tline)
        
        switch count
            case 0
                name = tline;
            case 1
                realization = tline;
            case 2
                implementation = str2num(tline);
            case 3
                size_numerator = str2num(tline);
            case 4
                size_denominator = str2num(tline);
            case 5
                numerator = str2num(tline);
            case 6
                denominator = str2num(tline);
            case 7
                delta = str2num(tline);
            case 8
                sample_time = str2num(tline);
            case 9
                drange = str2num(tline);
            case 10
                verification = tline;
            case 11
                xsize = str2num(tline);
            otherwise
                warning('Unexpected error while reading file.')
        end
        
        count = count + 1;
        tline = fgetl(fid);
        if count == 12
            
            count = 0;
            system(i).test_case = name;
            system(i).sys.a = denominator;
            system(i).sys.b = numerator;
            system(i).sys.tf = tf(numerator,denominator,1);
            system(i).impl.int_bits = implementation(1);
            system(i).impl.frac_bits = implementation(2);
            if length(sample_time) > 0
                system(i).impl.sample_time = sample_time;
            end
            if length(drange) > 0
                system(i).impl.range.max = drange(2);
                system(i).impl.range.min = drange(1);
            end
            if length(delta) > 0
                system(i).impl.delta = delta;
            else
                system(i).impl.delta = 0;
            end
            system(i).impl.realization_form = strtrim(realization);
            system(i).output.output_verification = verification;
            system(i).impl.x_size = xsize;
            
            i = i + 1;
            
        end
    end
    
    fclose(fid);
    
end

end
