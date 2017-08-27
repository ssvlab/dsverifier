function parsertf2c(system, intBits, fracBits, minRange, maxRange, error, delta, name)

a = cell2mat(system.Denominator);
nA = length(a);
b = cell2mat(system.Numerator);
nB = length(b);
sample_time = system.Ts;
frac_bits =  fracBits;
int_bits = intBits;
max_range = maxRange;
min_range = minRange;

file_name = [name '.c'];

fid = fopen(file_name, 'wt' );
fprintf(fid,'%s\n\n', '#include <dsverifier.h>');
fprintf(fid,'%s\n\t', 'digital_system ds = { ');
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(b));
fprintf(fid,'%s %d,\n\t','.b_size = ', nB);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(a));
fprintf(fid,'%s %d,\n\t','.a_size = ', nA);
fprintf(fid,'%s %d\n','.sample_time =', sample_time);
fprintf(fid,'%s\n\n','};');
fprintf(fid,'%s\n\t','implementation impl = { ');
if length(error) > 0
    if error ~= 0
    fprintf(fid,'%s %f,\n\t','.max_error = ', error);
    end
end
if length(delta) > 0
    if delta ~= 0
    fprintf(fid,'%s %f,\n','.delta =', delta);
    end
end
fprintf(fid,'%s %d,\n\t','.int_bits = ', int_bits);
fprintf(fid,'%s %d,\n\t','.frac_bits =  ', frac_bits);
fprintf(fid,'%s %f,\n\t','.max = ', max_range);
fprintf(fid,'%s %f\n\t','.min = ', min_range);
fprintf(fid,'%s\n\n','};');
fclose(fid);

end