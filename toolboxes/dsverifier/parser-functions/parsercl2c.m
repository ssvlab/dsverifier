function parsercl2c (plant, controller, intBits, fracBits, minRange, maxRange, error, name)

ac = cell2mat(controller.Denominator);
nAc = length(ac);
bc = cell2mat(controller.Numerator);
nBc = length(bc);

ap = cell2mat(plant.Denominator);
nAp = length(ap);
bp = cell2mat(plant.Numerator);
nBp = length(bp);

sample_time = controller.Ts;
frac_bits = fracBits;
int_bits = intBits;
max_range = maxRange;
min_range = minRange;

file_name = [name '.c'];

fid = fopen(file_name, 'wt' );
fprintf(fid,'%s\n\n', '#include <dsverifier.h>');
fprintf(fid,'%s\n\t', 'digital_system controller = { ');
fprintf(fid,'%s %s },\n\t','.b = { ', poly2strc(bc));
fprintf(fid,'%s %d,\n\t','.b_size = ', nBc);
fprintf(fid,'%s %s },\n\t','.a = { ', poly2strc(ac));
fprintf(fid,'%s %d,\n\t','.a_size = ', nAc);
fprintf(fid,'%s %d\n','.sample_time =', sample_time);
fprintf(fid,'%s\n\n','};');
fprintf(fid,'%s\n\t','implementation impl = { ');
if length(error) > 0
    if error ~= 0
    fprintf(fid,'%s %f,\n\t','.max_error = ', error);
    end
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

end