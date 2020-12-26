

%Multivariate Gaussian example generation
Pmax=0.5;
Pmin=0.1;
Ti=500;
T = zeros(3,Ti,90);

config=[ones(1,30),2*ones(1,30),3*ones(1,30)];%Les trois �tats de graph 1-30, 30-60, 60-90

for i =1:size(T,3)

e = rand(1,1)*(Pmax-Pmin)+Pmin;
a = (1-e);
Y = zeros(3,Ti);
step = config(i);

%%%%
if step==1
    Y(2,:)=randn(1,Ti);
    Y(1,:) = e*Y(2,:)+a*randn(1,Ti);
    Y(3,:) = e*Y(2,:)+a*randn(1,Ti);
elseif step==2
     Y(1,:)=randn(1,Ti);
     Y(2,:) = e*Y(1,:)+a*randn(1,Ti);
     Y(3,:) = e*Y(1,:)+a*randn(1,Ti);
elseif step==3
    Y(3,:)=randn(1,Ti);
    Y(1,:) = e*Y(3,:)+a*randn(1,Ti);
    Y(2,:) = e*Y(3,:)+a*randn(1,Ti);
else
    'problem'
end

T(:,:,i)=Y;
end




%Compute pairwise synchrony matrix {R^(1), R^(2), ..., R^(N)}
S = size(T);
P = zeros(S(1),S(1),S(3));

for i =1:size(T,3)
  X = T(:,:,i);  
  Z = hilbert(X')';
  Phi = angle(Z);
  for j = 1:S(1)
    for k =1:S(1)
        P(j,k,i) = 1/S(2)*sum(exp(1i*(Phi(j,:)-Phi(k,:))),2);
    end
  end
end


%Use rpPlV_ADMM to compute precision matrices {Omega^(1), Omega^(2), ..., Omega^(N)}
Mat = rpPLV_ADMM(P,0.1,0.5);%%%%% <-------

Result=[];
Underdiag  = (1:size(Mat(:,:,1),1)).' > (1:size(Mat(:,:,1),2));%Under diagonal elements selection
for i =1:size(T,3)
   A =  abs(Mat(:,:,i));
   B = diag(diag(A).^(-1/2));
   M = B*A*B;  %Normalize
   Result(:,i) =M(Underdiag);%(select upper diagonal elements)
end



imagesc(-Result,[-1,0])
colormap('pink')