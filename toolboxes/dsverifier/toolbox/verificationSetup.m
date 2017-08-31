function verificationSetup()
% Setup the DSVERIFIER_HOME and model checkers visibility (ESBMC or CBMC) in a variable environment
%
% Function: verificationSetup()
%
% Author: Lennon Chaves
% Federal University of Amazonas
% December 2016
%

home = pwd;

user = userpath;

if strfind(user,'/Documents/MATLAB') %default folder installation
    install_folder = [user '/Add-Ons/Toolboxes/DSVerifier/code'];
else
    install_folder = [user '/Toolboxes/DSVerifier/code'];
end


cd(install_folder);

current = pwd;

cd(home);

setenv('DSVERIFIER_HOME',current);

setenv('PATH', [getenv('PATH') ':' current]);

end

