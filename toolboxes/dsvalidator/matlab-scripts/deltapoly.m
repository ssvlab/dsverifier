function [Da, Db]=deltapoly(b,a,delta)

% 
% [Da, Db]=deltapoly(b,a,delta)
% 
% Performs the delta transfor over a transfer function b(z)/a(z)
% 
% a is the denominator polynomial
% b is the numerator polynomial
% delta is transformation step factor
%     
% Iury Bessa
% September 03, 2016
% Manaus



N=length(a)-1;
M=length(b)-1;

for i=0:N
    suma=0;
    for j=0:i
        suma=suma+a(j+1)*nchoosek(N-j,i-j);
    end
    Da(i+1)=delta^(N-i)*suma;
end

for i=0:M
    suma=0;
    for j=0:i
        suma=suma+b(j+1)*nchoosek(M-j,i-j);
    end
    Db(i+1)=delta^(M-i)*suma;
end

end