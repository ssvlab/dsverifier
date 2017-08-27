function parserss2ss (system, inputs, intBits, fracBits, K, name)

A = system.A;
B = system.B;
C = system.C;
D = system.D;

[rA,cA] = size(A);
nStates = rA;
[rB,cB] = size(B);
nInputs = cB;
[rC,cC] = size(C);
nOutputs = rC;

frac_bits = fracBits;
int_bits = intBits;

file_name = [name '.ss'];

fid = fopen(file_name, 'wt' );
fprintf(fid,'implementation <%d,%d>\n', int_bits, frac_bits);
fprintf(fid,'states = %d;\n', nStates);
fprintf(fid,'inputs = %d;\n', nInputs);
fprintf(fid,'outputs = %d;\n', nOutputs);
fprintf(fid,'A = %s \n', matrix2string(A));
fprintf(fid,'B = %s \n', matrix2string(B));
fprintf(fid,'C = %s \n', matrix2string(C));
fprintf(fid,'D = %s \n', matrix2string(D));
fprintf(fid,'inputs = %s \n', matrix2string(inputs));
if isempty(K) == 0
    if K ~= 0
        fprintf(fid,'K = %s \n', matrix2string(K));
    end
end
fclose(fid);

end