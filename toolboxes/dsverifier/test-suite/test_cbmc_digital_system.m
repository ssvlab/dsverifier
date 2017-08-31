%% Test 1: Verify Overflow
ds.system = tf([0.15, 0.05, 0.4], [1.0, 0.0, 0.3], 0.01);
ds.impl.frac_bits = 3;
ds.impl.int_bits = 4;
ds.range.max = 1;
ds.range.min = -1;
ds.delta = 0;
verify_error(ds,'CBMC','DFI','',4,0.18);
