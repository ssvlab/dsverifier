function result=overflowtest(a,b,u,l,n,domain,delta)

% 
% result=overflowtest(a,b,u,l,n,domain,delta)
% 
% Verifies the overflow occurrence in a digital system implemented in FXP
% format with n bits and l frractional bits. The result will be equal to 1
% if an overflow occurs, and it will be zero otherwise.
% 
% a is the denominator polynomial
% b is the numerator polynomial
% u is the input sequence (from couter example)
% l is the number of fractional bits
% n is the number of bits (total)
% domain have to equal to 'd' if it will be an delta implementation
% delta is transformation step factor
% 
% 
%     
% Iury Bessa
% September 04, 2016
% Manaus


if domain=='d'
    [at,bt]=deltapoly(a,b,delta);
else
    at=a;
    bt=b;
end
uf=(2^(-1*l))*u;
[y,x]=dlsim(bt,at,uf);
stairs(y);
hold on;
plot([1:length(y)],-1*(((2^n)-1)/(2^l))*ones(length(y),1),'r')
plot([1:length(y)],(((2^n)-1)/(2^l))*ones(length(y),1),'r')
hold off

for i=1:length(y)
    if (y(i)>(((2^n)-1)/(2^l))) || (y(i)<-1*(((2^n)-1)/(2^l)))
        result=1;
        display('An overflow occurred');
        break;
    else
        result=0;
    end
end

if result==0
    display('There were no overflow');
end

end