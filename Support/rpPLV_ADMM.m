function [Theta, e,rnorm,snorm,eprimal,edual,iter] = rpPLV_ADMM(K,lam,beta)

%Matlab code to solve time varying graphical lasso from the paper
%Regularized partial phase synchrony index applied to dynamical functional connectivity %estimation, Gaëtan Frusque, Julien Jung, Pierre Borgnat, Paulo Gonçalves\
%ICASSP 2020, 01-2020

%inspired by :
%D. Hallac, Y. Park, S. Boyd, and J. Leskovec, “Network
%inference via the time-varying graphical lasso,” in Proceedings
%of the 23rd ACM SIGKDD International Conference
%on Knowledge Discovery and Data Mining. ACM, 2017,
%pp. 205–213.
%and the code from :
%Federico Tomasi, Veronica Tozzo, Alessandro Verri, and Saverio Salzo. Forward-Backward
%Splitting for Time-Varying Graphical Models. In International Conference on Probabilistic
%Graphical Models, pages 475–486, 2018.


%Set parameters
rho=0.01;%ADMM parameter
Name = 3;% temporal regularization utilis�e

%Initialisation
Theta = zeros(size(K));
for i =1:size(K,3)
    Theta(:,:,i) = rand(size(Theta(:,:,i)));
end

A = Theta;
U = zeros(size(Theta));



B = Theta(:,:,2:end);
V = zeros(size(Theta,1),size(Theta,2),size(Theta,3)-1);


C = Theta(:,:,1:(end-1));
W = zeros(size(Theta,1),size(Theta,2),size(Theta,3)-1);

A_b = zeros(size(Theta));
B_b = zeros(size(Theta,1),size(Theta,2),size(Theta,3)-1);
C_b = zeros(size(Theta,1),size(Theta,2),size(Theta,3)-1);

eabs = 10^-4;
erel = 10^-4;


iterMax = 5000;
e = [];
rnorm = [];
snorm = [];
eprimal= [];
edual= [];

ni=1;%%%%%
decr = 2;%%%%%

    a = ni*real(-log(det(Theta(:,:,1)))+trace(K(:,:,1)*Theta(:,:,1)))...
       +lam*funct(A(:,:,1),0);
    for j=1:(size(Theta,3)-1)
    a = a+ni*real(-log(det(Theta(:,:,j+1)))+trace(K(:,:,j+1)*Theta(:,:,j+1)))+...
           lam*funct(A(:,:,1),0)+beta*funct(B(:,:,j)-C(:,:,j),Name);
    end
  e(1) = a;



  
BS = sqrt(numel(K)+2*numel(W))*eabs;


if size(K,3)<3
  ndiv = ones(1,size(K,3));
else
  ndiv = [2,3*ones(1,size(K,3)-2),2]; 
end




%Iteration
for i=1:iterMax
    
    %Step1
Const1=lam/rho;
Const2 = (2*beta)/rho;
 
    eta = rho*ndiv/ni;
    Z = A-U;
    Z(:,:,2:end) =Z(:,:,2:end)+B-V;
    Z(:,:,1:(end-1)) =Z(:,:,1:(end-1))+C-W;

    for j =1:size(K,3)
        D = (Z(:,:,j)+ctranspose(Z(:,:,j)))/(2*ndiv(j));
        D = eta(j)*D-K(:,:,j);
        [Q,L] = eig(D);
         Theta(:,:,j) = (1/(2*eta(j)))*Q*(L+sqrt(L.^2+...
             4*eta(j)*eye(size(Theta,1))))*ctranspose(Q);  
    end

    %Step2

    
    Z2 = Theta+U;
    Z3=[];
    for j =1:size(Theta,3)
       Z3(:,:,j) = ctranspose(Z2(:,:,j)); 
    end
    Z = (Z2+Z3)/2;
    A = proxy(Z,Const1,1);

    T1 = Theta(:,:,1:(end-1))+Theta(:,:,2:end)+V+W;
    T2 = -Theta(:,:,1:(end-1))+Theta(:,:,2:end)+V-W;
    for j =1:(size(T1,3))
        B(:,:,j) = 1/2*(T1(:,:,j)+proxy(T2(:,:,j),Const2,Name));
        C(:,:,j) = 1/2*(T1(:,:,j)-proxy(T2(:,:,j),Const2,Name));
    end

    %Step3

    U = U+Theta-A;
    V = V+Theta(:,:,2:end)-B;
    W = W+Theta(:,:,1:(size(Theta,3)-1))-C;


    %Scores 

    %Main
    a = ni*real(-log(det(Theta(:,:,1)))+trace(K(:,:,1)*Theta(:,:,1)))...
       +lam*funct(A(:,:,1),0);
    for j=1:(size(Theta,3)-1)
    a = a+ni*real(-log(det(Theta(:,:,j+1)))+trace(K(:,:,j+1)*Theta(:,:,j+1)))+...
  lam*funct(A(:,:,j),0)+beta*funct(B(:,:,j)-C(:,:,j),Name);
    end
    e(i+1)=a;
    
    %rho score
    
    rnorm(i) = sqrt(sum3(Theta-A)+sum3(Theta(:,:,2:end)-B)+ sum3(Theta(:,:,1:(end-1))-C));
    snorm(i) = rho*sqrt(sum3(A-A_b)+sum3(B-B_b)+ sum3(C-C_b));
    eprimal(i) = BS+erel*max(sqrt(sum3(A)+sum3(B)+sum3(C)),...
        sqrt(sum3(Theta)+sum3(Theta(:,:,2:end))+sum3(Theta(:,:,1:(end-1)))));
    edual(i) = BS+erel*rho*(sqrt(sum3(U)+sum3(V)+sum3(W)));

    
    if(rnorm(i)<=eprimal(i) && snorm(i)<=edual(i))
       break 
    end
    
    if(rnorm(i)>10*snorm(i))
        rho_b=rho;
        rho = decr*rho;
        rappor = rho_b/rho;
        U = U*rappor;
        V = V*rappor;
        W = W*rappor;
    elseif(snorm(i)>10*rnorm(i))
        rho_b=rho;
        rho = rho/decr;
        rappor = rho_b/rho;
        U = U*rappor;
        V = V*rappor;
        W = W*rappor;
    end
    
    A_b = A;
    B_b = B;
    C_b = C;

    

    
end

iter=i;



end



function Res = proxy(A,b,Name)

if Name == 1%Soft thresholding
    R = A;
    if isreal(A)
        Res=max(abs(A)-b,0).*sign(A);
    else
        Res=max(abs(A)-b,0).*exp(1i*angle(A));
    end
 %UU = Res;   
for i =1:size(Res,3)
Res(:,:,i)=Res(:,:,i)-diag(diag(Res(:,:,i)))+diag(diag(A(:,:,i)));
end
% subplot(121)
% imagesc(abs(A(:,:,i)))
% subplot(122)
% imagesc(abs(Res(:,:,i)))



elseif Name == 2%Laplacian
    Res = 1/(1+2*b)*A;
elseif Name == 3%Group lasso
    s = sqrt(trace(ctranspose(A)*A));
    %filt = repmat(double(s-b>0),size(A,1),1).*(repmat(1-b./s,size(A,1),1));
    Res = repmat(max(1-b./s,0),size(A,1),1).*A;%A voir
else
    'Problem'
end

end


function Psi = funct(M,Name)

if Name == 1%Soft thresholding
    Psi = sum(sum(abs(M)));
elseif Name == 2%Laplacian
    Psi = sqrt(trace(M*ctranspose(M)));
elseif Name == 3%Group lasso
    Psi = sum(sqrt(sum(abs(M))));
elseif Name == 0
    Psi = sum(sum(abs(M-diag(diag(M)))));
else
    'Problem'
end


end

function R = proxy2(B,b)
if length(size(B))==2
    A=B;
a = diag(diag(A));
A = A-a;
    if isreal(A)
        Res=max(abs(A)-b,0).*sign(A);
    else
        Res=max(abs(A)-b,0).*exp(1i*angle(A));
    end

R = Res+a;  
else
    R=zeros(size(B));
for i =1:size(B,3)
    A =B(:,:,i);
a = diag(diag(A));
A = A-a;
    if isreal(A)
        Res=max(abs(A)-b,0).*sign(A);
    else
        Res=max(abs(A)-b,0).*exp(1i*angle(A));
    end

Res = Res+a;
R(:,:,i)=Res;
end
end
end



function r = sum3(T)

r = sum(sum(sum(abs(T).^2)));

end

