function decision=eiginside(sys,testtype)
% 
% decision=eiginside(sys,testtype)
% 
% For a LTI system (sys) in transfer function format, eiginside(sys,testtype) 
% decides about the stability and the minimum phase of sys. It returns 
% decision = 1 if the system presents stability (or minimum-phase), and 
% returns decision = 0 in other case.
% 
% Inputs testtype='s' for stability test, and testype='m' for minimum-phase 
% test.
%     
% Iury Bessa
% August 24, 2016
% Manaus

bpoly=cell2mat(sys.num);
apoly=cell2mat(sys.den);
rootsb=roots(bpoly);
rootsa=roots(apoly);
if testtype == 's'
    for i=1:length(rootsa)
        if abs(rootsa(i))>=1
            decision=0;
            display('The system is UNSTABLE')
            break
        end
        decision=1;    
    end
    if decision==1
        display('The system is STABLE')
    end
elseif testtype == 'm'
    for i=1:length(rootsb)
        if abs(rootsb(i))>=1
            decision=0;
            display('The system is a NON-MINIMUM-PHASE system')
            break
        end
        decision=1;
    end
    if decision==1
        display('The system is a MINIMUM-PHASE system')    
    end
else
    display('Invalid testtype')
end
end