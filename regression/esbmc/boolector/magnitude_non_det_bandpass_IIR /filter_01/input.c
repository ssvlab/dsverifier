//ipb2(1)

#include <dsverifier.h>

digital_system ds = {
            .b =  { 0.420807779837731821270807586188311688602 ,  0, -0.420807779837731821270807586188311688602 },                                
            .b_size = 3,
            .a =  { 1, -0.00000000000000008767503074033515886233,  0.158384440324536274191657980736636091024 },
            .a_size = 3
         };

implementation impl = {
         .int_bits = 2,
         .frac_bits = 5,
         .min = -1.6,
         .max = 1.6,
};

filter_parameters filter = {
         .Ac =  0.707945784384138, 
         .Ap =  0.707945784384138, 
         .Ar =  0.707945784384138, 

         .w1c = 0.3, 
         .w2c = 0.7,
         .wc = 0.8,

         .w1p = 0.31,
         .w2p = 0.69,

         .w1r = 0.29,
         .w2r = 0.71,

        // .1st_wr = 0.29,
        // .2nd_wr = 0.71,

         .type = 3
};


