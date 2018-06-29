#include <dsverifier.h>

hardware hw = {
	.clock = 20000000,
	.assembly = {
		.push = 2,
		.in = 1,
		.sbiw = 2,
		.cli = 1,
		.out = 1,
		.std = 2,
		.ldd = 2,
		.subi = 1,
		.sbci = 1,
		.lsl = 1,
		.rol = 1,
		.add = 1,
		.adc = 1,
		.adiw = 2,
		.rjmp = 2,
		.mov = 1,
		.sbc = 1,
		.ld = 2,
		.rcall = 4,
		.cp = 1,
		.cpc = 1,
		.ldi = 1,
		.brge = 2,
		.pop = 2,
		.ret = 5,
		.st = 2,
		.brlt = 2,
		.cpi = 1
	}
};

#if DS_ID == 1

	digital_system ds = {
		.b = { 1.5, -0.5 },
		.b_size = 2,
		.a = { 1.0, 0.0 },
		.a_size = 2,
		.sample_time = 0.02
	};

	#define	IMPLEMENTATION_COUNT 3

	/* carletta's implementation */
	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 2,
			.frac_bits = 14,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 4,
			.frac_bits = 12,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 6,
			.frac_bits = 10,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif

#if DS_ID == 2

	digital_system ds = {
		.b = { 60.0, -50.0 },
		.b_size = 2,
		.a = { 1.0, 0.0 },
		.a_size = 2,
		.sample_time = 0.02
	};

	#define	IMPLEMENTATION_COUNT 3

	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 6,
			.frac_bits = 10,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	/* carletta's implementation */
	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 8,
			.frac_bits = 8,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 10,
			.frac_bits = 6,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif

#if DS_ID == 3

	digital_system ds = {
		.b = { 110.0, -100.0 },
		.b_size = 2,
		.a = { 1.0, 0.0 },
		.a_size = 2,
		.sample_time = 0.02
	};

	#define	IMPLEMENTATION_COUNT 3

	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 7,
			.frac_bits = 9,
			.max = 1.0,
			.min = -1.0,
		};
	#endif


	/* carletta's implementation */
	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 9,
			.frac_bits = 7,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 11,
			.frac_bits = 5,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif

#if DS_ID == 4

	digital_system ds = {
		.b = { 135.0, -260.0, 125.0 },
		.b_size = 3,
		.a = { 1.0, -1.0, 0.0 },
		.a_size = 3,
		.sample_time = 0.02
	};

	#define	IMPLEMENTATION_COUNT 3

	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 8,
			.frac_bits = 8,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	/* carletta's implementation */
	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 10,
			.frac_bits = 6,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 11,
			.frac_bits = 5,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif

#if DS_ID == 5

	digital_system ds = {
		.b = { 2002.0, -4000.0, 1998.0 },
		.b_size = 3,
		.a = { 1.0, 0.0, -1.0 },
		.a_size = 3,
		.sample_time = 0.001
	};

	#define	IMPLEMENTATION_COUNT 3

	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 10,
			.frac_bits = 6,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 12,
			.frac_bits = 4,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	/* carletta's implementation */
	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 13,
			.frac_bits = 3,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif

/** PDPIBI */

/* pd */

#if DS_ID == 6

	digital_system ds = {
		.b = { 0.93, -0.87 },
		.b_size = 2,
		.a = { 1.0, 1.0 },
		.a_size = 2,
		.sample_time = 0.02
	};

	#define	IMPLEMENTATION_COUNT 3

	/* carletta's implementation (not possible, unstable) */

	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 4,
			.frac_bits = 12,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 8,
			.frac_bits = 8,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 10,
			.frac_bits = 6,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif

/* pi */

#if DS_ID == 7

	digital_system ds = {
		.b = { 0.1, -0.09998 },
		.b_size = 2,
		.a = { 1.0, -1.0 },
		.a_size = 2,
		.sample_time = 0.02
	};

	#define	IMPLEMENTATION_COUNT 3

	/* carletta's implementation  (not possible, unstable) */

	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 4,
			.frac_bits = 12,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 8,
			.frac_bits = 8,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 10,
			.frac_bits = 6,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif

/** PDPIBE */

/* pd */

#if DS_ID == 8

	digital_system ds = {
		.b = { 0.0096, -0.009 },
		.b_size = 2,
		.a = { 0.02, 0.0 },
		.a_size = 2,
		.sample_time = 0.02
	};

	#define	IMPLEMENTATION_COUNT 3

	/* carletta's implementation (ok) */
	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 3,
			.frac_bits = 13,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 4,
			.frac_bits = 12,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 5,
			.frac_bits = 11,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif

/* pi */

#if DS_ID == 9

	digital_system ds = {
		.b = { 0.1, -0.1 },
		.b_size = 2,
		.a = { 1.0, -1.0 },
		.a_size = 2,
		.sample_time = 0.02
	};

	#define	IMPLEMENTATION_COUNT 3

	/* carletta's implementation (not possible, unstable) */
	#if IMPLEMENTATION_ID == 1
		implementation impl = {
			.int_bits = 4,
			.frac_bits = 12,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 2
		implementation impl = {
			.int_bits = 8,
			.frac_bits = 8,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

	#if IMPLEMENTATION_ID == 3
		implementation impl = {
			.int_bits = 10,
			.frac_bits = 6,
			.max = 1.0,
			.min = -1.0,
		};
	#endif

#endif
