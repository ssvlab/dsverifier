function synthParse(ds, type)
% Translate the digital plant designed as transfer-function or state-space for an ANSI-C file
%
% Function: synthParse(plant, type)
%
%  plant: digital plant with specifications about plant, implementation and dynamical ranges;
%  type: 'ss' for state-space or 'tf' for transfer-function;
%
% Author: Lennon Chaves
% 
% March 2017
%

if strcmp(type,'ss')

A = ds.system.A;
B = ds.system.B;
C = ds.system.C;
D = ds.system.D;

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

fid = fopen('system.ss', 'wt' );
fprintf(fid,'implementation <%d,%d>\n', int_bits, frac_bits);
fprintf(fid,'states = %d;\n', nStates);
fprintf(fid,'inputs = %d;\n', nInputs);
fprintf(fid,'outputs = %d;\n', nOutputs);
fprintf(fid,'A = %s \n', matrix2string(A));
fprintf(fid,'B = %s \n', matrix2string(B));
fprintf(fid,'C = %s \n', matrix2string(C));
fprintf(fid,'D = %s \n', matrix2string(D));
fprintf(fid,'inputs = [%d,%d] \n', min_range, max_range);
fclose(fid);

elseif strcmp(type, 'tf')

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

fid = fopen('system.c', 'wt' );
fprintf(fid,'%s\n\n', '#include <dsverifier.h>');
fprintf(fid,'%s\n\t', 'digital_system controller = { ');
bc = zeros(1,nBc);
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(bc));
uncertainty_b = zeros(1,nBc);
fprintf(fid,'%s %s },\n\t','.b_uncertainty = { ', poly2strc(uncertainty_b));
fprintf(fid,'%s %d,\n\t','.b_size = ', nBc);
ac = zeros(1,nAc);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(ac));
uncertainty_a = zeros(1,nAc);
fprintf(fid,'%s %s },\n\t','.a_uncertainty = { ', poly2strc(uncertainty_a));
fprintf(fid,'%s %d,\n\t','.a_size = ', nAc);
fprintf(fid,'%s %d\n','.sample_time =', sample_time);
fprintf(fid,'%s\n\n','};');
fprintf(fid,'%s\n\t','implementation impl = { ');
fprintf(fid,'%s %d,\n\t','.int_bits = ', int_bits);
fprintf(fid,'%s %d,\n\t','.frac_bits =  ', frac_bits);
fprintf(fid,'%s %f,\n\t','.max = ', max_range);
fprintf(fid,'%s %f\n\t','.min = ', min_range);
fprintf(fid,'%s\n\n','};');

fprintf(fid,'%s\n\t', 'digital_system plant = { ');
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(bp));
uncertainty_b = zeros(1,nBp);
fprintf(fid,'%s %s },\n\t','.b_uncertainty = { ', poly2strc(uncertainty_b));
fprintf(fid,'%s %d,\n\t','.b_size = ', nBp);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(ap));
uncertainty_a = zeros(1,nAp);
fprintf(fid,'%s %s },\n\t','.a_uncertainty = { ', poly2strc(uncertainty_a));
fprintf(fid,'%s %d ,\n\t','.a_size = ', nAp);
fprintf(fid,'%s\n\n','};');
fclose(fid);

end

end
