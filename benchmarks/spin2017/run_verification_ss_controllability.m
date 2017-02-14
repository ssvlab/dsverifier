makebenchmarks

clear all
clc

load('benchmark_ss.mat')

%DC Motor - 8 benchmarks - dcmotor_ss_disc
%Pendulum - 8 benchmarks - pendulum_ss_disc
%Inverted Cart Pendulum - 8 benchmarks - invpendulum_cartpos_ss_disc 
%Simple Magnetic Suspension System - 8 benchmarks - magsuspension_ss_disc
%Car Cruise Control - 7 benchmarks - cruise_ss_disc
%Computer Tape Driver - 8 benchmarks - tapedriver_ss_disc
%Helicopter Longitudinal Motion - 8 benchmarks - helicopter_ss_disc
%USCG cutter Tampa Heading Angle - 8 benchmarks - uscgtampa_ss_disc
%Magnetic Pointer - 8 benchmarks -magneticpointer_ss_disc

%Inputs

inputs = [1.0];

verifyStateSpaceControllability(dcmotor_ss_disc1, inputs, 12,4,1,-1,'');
verifyStateSpaceControllability(dcmotor_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(dcmotor_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(dcmotor_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(dcmotor_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(dcmotor_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(dcmotor_ss_disc7, inputs, 9,7,1,-1,'');
verifyStateSpaceControllability(dcmotor_ss_disc8, inputs, 11,5,1,-1,'');

verifyStateSpaceControllability(pendulum_ss_disc1, inputs, 12,4,1,-1,'');
verifyStateSpaceControllability(pendulum_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(pendulum_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(pendulum_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(pendulum_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(pendulum_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(pendulum_ss_disc7, inputs, 9,7,1,-1,'');
verifyStateSpaceControllability(pendulum_ss_disc8, inputs, 11,5,1,-1,'');

verifyStateSpaceControllability(invpendulum_cartpos_ss_disc1, inputs, 12,4,1,-1,'');
verifyStateSpaceControllability(invpendulum_cartpos_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(invpendulum_cartpos_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(invpendulum_cartpos_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(invpendulum_cartpos_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(invpendulum_cartpos_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(invpendulum_cartpos_ss_disc7, inputs, 9,7,1,-1,'');
verifyStateSpaceControllability(invpendulum_cartpos_ss_disc8, inputs, 11,5,1,-1,'');

verifyStateSpaceControllability(magsuspension_ss_disc1, inputs, 12,4,1,-1,'');
verifyStateSpaceControllability(magsuspension_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(magsuspension_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(magsuspension_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(magsuspension_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(magsuspension_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(magsuspension_ss_disc7, inputs, 9,7,1,-1,'');
verifyStateSpaceControllability(magsuspension_ss_disc8, inputs, 11,5,1,-1,'');

verifyStateSpaceControllability(cruise_ss_disc1, inputs, 12,4,1,-1,'');
verifyStateSpaceControllability(cruise_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(cruise_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(cruise_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(cruise_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(cruise_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(cruise_ss_disc7, inputs, 9,7,1,-1,'');

verifyStateSpaceControllability(tapedriver_ss_disc1, inputs, 12,4,1,-1,'');
verifyStateSpaceControllability(tapedriver_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(tapedriver_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(tapedriver_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(tapedriver_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(tapedriver_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(tapedriver_ss_disc7, inputs, 9,7,1,-1,'');
verifyStateSpaceControllability(tapedriver_ss_disc8, inputs, 11,5,1,-1,'');

verifyStateSpaceControllability(helicopter_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(helicopter_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(helicopter_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(helicopter_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(helicopter_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(helicopter_ss_disc7, inputs, 9,7,1,-1,'');
verifyStateSpaceControllability(helicopter_ss_disc8, inputs, 11,5,1,-1,'');

verifyStateSpaceControllability(uscgtampa_ss_disc1, inputs, 12,4,1,-1,'');
verifyStateSpaceControllability(uscgtampa_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(uscgtampa_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(uscgtampa_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(uscgtampa_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(uscgtampa_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(uscgtampa_ss_disc7, inputs, 9,7,1,-1,'');
verifyStateSpaceControllability(uscgtampa_ss_disc8, inputs, 11,5,1,-1,'');

verifyStateSpaceControllability(magneticpointer_ss_disc1, inputs, 12,4,1,-1,'');
verifyStateSpaceControllability(magneticpointer_ss_disc2, inputs, 2,14,1,-1,'');
verifyStateSpaceControllability(magneticpointer_ss_disc3, inputs, 10,6,1,-1,'');
verifyStateSpaceControllability(magneticpointer_ss_disc4, inputs, 8,8,1,-1,'');
verifyStateSpaceControllability(magneticpointer_ss_disc5, inputs, 6,10,1,-1,'');
verifyStateSpaceControllability(magneticpointer_ss_disc6, inputs, 7,9,1,-1,'');
verifyStateSpaceControllability(magneticpointer_ss_disc7, inputs, 9,7,1,-1,'');
verifyStateSpaceControllability(magneticpointer_ss_disc8, inputs, 11,5,1,-1,'');
