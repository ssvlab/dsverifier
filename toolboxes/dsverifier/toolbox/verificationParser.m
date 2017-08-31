function verificationParser(ds, type, error, realization)
% Reads the system in struct form and translate to a ANSI-C file
%
% Function: verificationParser(ds, type, error, realization)
%
%  ds: digital system
%  type: 'ss' for state-space, 'tf' for transfer function and 'cl' for
%  closed-loop systems and 'rb' for robust closed-loop system (with uncertainty)
%  error: error for quantization error verification
%  realization: direct or delta form realization
%
% Author: Lennon Chaves
% Federal University of Amazonas
% February 2017
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(realization,'DDFI') || strcmp(realization,'DDFII') || strcmp(realization,'TDDFII')
delta = ds.delta;
else
delta = 0;
end

if strcmp(type, 'tf')
a = cell2mat(ds.system.Denominator);
nA = length(a);
b = cell2mat(ds.system.Numerator);
nB = length(b);
sample_time = ds.system.Ts;
frac_bits = ds.impl.frac_bits;
int_bits = ds.impl.int_bits;
max_range = ds.range.max;
min_range = ds.range.min;

fid = fopen('input.c', 'wt' );
fprintf(fid,'%s\n\n', '#include <dsverifier.h>');
fprintf(fid,'%s\n\t', 'digital_system ds = { ');
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(b));
fprintf(fid,'%s %d,\n\t','.b_size = ', nB);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(a));
fprintf(fid,'%s %d,\n\t','.a_size = ', nA);
fprintf(fid,'%s %d\n','.sample_time =', sample_time);
fprintf(fid,'%s\n\n','};');
fprintf(fid,'%s\n\t','implementation impl = { ');
if error > 0
fprintf(fid,'%s %f,\n\t','.max_error = ', error);
end
if delta > 0
fprintf(fid,'%s %f,\n','.delta =', delta);
end
fprintf(fid,'%s %d,\n\t','.int_bits = ', int_bits);
fprintf(fid,'%s %d,\n\t','.frac_bits =  ', frac_bits);
fprintf(fid,'%s %f,\n\t','.max = ', max_range);
fprintf(fid,'%s %f\n\t','.min = ', min_range);
fprintf(fid,'%s\n\n','};');
fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(type, 'cl')
ac = cell2mat(ds.controller.Denominator);
nAc = length(ac);
bc = cell2mat(ds.controller.Numerator);
nBc = length(bc);

ap = cell2mat(ds.plant.Denominator);
nAp = length(ap);
bp = cell2mat(ds.plant.Numerator);
nBp = length(bp);

sample_time = ds.controller.Ts;
frac_bits = ds.impl.frac_bits;
int_bits = ds.impl.int_bits;
max_range = ds.range.max;
min_range = ds.range.min;

fid = fopen('input.c', 'wt' );
fprintf(fid,'%s\n\n', '#include <dsverifier.h>');
fprintf(fid,'%s\n\t', 'digital_system controller = { ');
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(bc));
fprintf(fid,'%s %d,\n\t','.b_size = ', nBc);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(ac));
fprintf(fid,'%s %d,\n\t','.a_size = ', nAc);
fprintf(fid,'%s %d\n','.sample_time =', sample_time);
fprintf(fid,'%s\n\n','};');
fprintf(fid,'%s\n\t','implementation impl = { ');
if error > 0
fprintf(fid,'%s %f,\n\t','.max_error = ', error);
end
fprintf(fid,'%s %d,\n\t','.int_bits = ', int_bits);
fprintf(fid,'%s %d,\n\t','.frac_bits =  ', frac_bits);
fprintf(fid,'%s %f,\n\t','.max = ', max_range);
fprintf(fid,'%s %f\n\t','.min = ', min_range);
fprintf(fid,'%s\n\n','};');
fprintf(fid,'%s\n\t', 'digital_system plant = { ');
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(bp));
fprintf(fid,'%s %d,\n\t','.b_size = ', nBp);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(ap));
fprintf(fid,'%s %d \n\t','.a_size = ', nAp);
fprintf(fid,'%s\n\n','};');
fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
elseif strcmp(type, 'rb')
ac = cell2mat(ds.controller.Denominator);
nAc = length(ac);
bc = cell2mat(ds.controller.Numerator);
nBc = length(bc);

ap = cell2mat(ds.plant.Denominator);
nAp = length(ap);
bp = cell2mat(ds.plant.Numerator);
nBp = length(bp);

sample_time = ds.controller.Ts;
frac_bits = ds.impl.frac_bits;
int_bits = ds.impl.int_bits;
max_range = ds.range.max;
min_range = ds.range.min;

uncertainty_a = ds.uncertainty_a;
uncertainty_b = ds.uncertainty_b;

fid = fopen('input.c', 'wt' );
fprintf(fid,'%s\n\n', '#include <dsverifier.h>');
fprintf(fid,'%s\n\t', 'digital_system controller = { ');
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(bc));
fprintf(fid,'%s %d,\n\t','.b_size = ', nBc);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(ac));
fprintf(fid,'%s %d,\n\t','.a_size = ', nAc);
fprintf(fid,'%s %d\n','.sample_time =', sample_time);
fprintf(fid,'%s\n\n','};');
fprintf(fid,'%s\n\t','implementation impl = { ');
if error > 0
fprintf(fid,'%s %f,\n\t','.max_error = ', error);
end
fprintf(fid,'%s %d,\n\t','.int_bits = ', int_bits);
fprintf(fid,'%s %d,\n\t','.frac_bits =  ', frac_bits);
fprintf(fid,'%s %f,\n\t','.max = ', max_range);
fprintf(fid,'%s %f\n\t','.min = ', min_range);
fprintf(fid,'%s\n\n','};');

fprintf(fid,'%s\n\t', 'digital_system plant = { ');
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(bp));
fprintf(fid,'%s %s },\n\t','.b_uncertainty = { ', poly2strc(uncertainty_b));
fprintf(fid,'%s %d,\n\t','.b_size = ', nBp);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(ap));
fprintf(fid,'%s %s },\n\t','.a_uncertainty = { ', poly2strc(uncertainty_a));
fprintf(fid,'%s %d \n\t','.a_size = ', nAp);
fprintf(fid,'%s\n\n','};');
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
elseif strcmp(type, 'ss')
    
A = ds.system.A;
B = ds.system.B;
C = ds.system.C;
D = ds.system.D;
inputs = ds.inputs;
if isempty(ds.K) == 0
    K = ds.K;
else 
    K = [];
end


[rA,cA] = size(A);
nStates = rA;
[rB,cB] = size(B);
nInputs = cB;
[rC,cC] = size(C);
nOutputs = rC;

frac_bits = ds.impl.frac_bits;
int_bits = ds.impl.int_bits;
max_range = ds.range.max;
min_range = ds.range.min;

fid = fopen('file.ss', 'wt' );
fprintf(fid,'implementation <%d,%d>\n', int_bits, frac_bits);
fprintf(fid,'range [%d,%d]\n', max_range, min_range);
fprintf(fid,'states = %d;\n', nStates);
fprintf(fid,'inputs = %d;\n', nInputs);
fprintf(fid,'outputs = %d;\n', nOutputs);
fprintf(fid,'A = %s \n', matrix2string(A));
fprintf(fid,'B = %s \n', matrix2string(B));
fprintf(fid,'C = %s \n', matrix2string(C));
fprintf(fid,'D = %s \n', matrix2string(D));
fprintf(fid,'inputs = %s \n', matrix2string(inputs));
if isempty(K) == 0
   fprintf(fid,'K = %s \n', matrix2string(K));
end
fclose(fid);

end

end

