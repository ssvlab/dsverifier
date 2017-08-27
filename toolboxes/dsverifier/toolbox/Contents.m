% TOOLBOX
%
% Files
%   genCounterexample                 - Translate .out file generated after verification to a .MAT file with
%   getExtraParams                    - Support function to get the extra params in order to add in command line
%   matrix2string                     - Translate a matrix representation to a string representation
%   poly2strc                         - Translate a polynomial representation to a string representation
%   verificationExecution             - Function to get the parameters, system and implementatio to start the verification
%   verificationParser                - Reads the system in struct form and translate to a ANSI-C file
%   verificationReport                - Shows the report about the verification: successful or failed
%   verificationSetup                 - Setup the DSVERIFIER_HOME and model checkers visibility (ESBMC or CBMC) in a variable environment
%   verifyClosedLoopLimitCycle        - Checks limit cycle property violation for closed-loop digital systems using a bounded model checking tool.
%   verifyClosedLoopQuantizationError - Checks quantization error property violation for closed-loop digital systems using a bounded model checking tool.
%   verifyError                       - Checks error property violation for digital systems using a bounded model checking tool.
%   verifyLimitCycle                  - Checks limit cycle property violation for digital systems using a bounded model checking tool.
%   verifyMinimumPhase                - Checks minimum phase property violation for digital systems using a bounded model checking tool.
%   verifyOverflow                    - Checks overflow property violation for digital systems using a bounded model checking tool.
%   verifyRobustStability             - Checks stability property violation for closed-loop digital systems using a bounded model checking tool.
%   verifyStability                   - Checks stability property violation for digital systems using a bounded model checking tool.
%   verifyStateSpaceControllability   - Checks controllability property violation for digital systems (state-space representation) using a bounded model checking tool.
%   verifyStateSpaceObservability     - Checks observability property violation for digital systems (state-space representation) using a bounded model checking tool.
%   verifyStateSpaceQuantizationError - Checks quantization error property violation for digital systems (state-space representation) using a bounded model checking tool.
%   verifyStateSpaceStability         - Checks stability property violation for digital systems (state-space representation) using a bounded model checking tool.
%   verifyTiming                      - Checks timing property violation for digital systems using a bounded model checking tool.
