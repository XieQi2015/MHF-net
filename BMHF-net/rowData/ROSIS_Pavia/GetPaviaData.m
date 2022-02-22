function GetPaviaData
% X = double(imread('original_rosis.tif'))/5000;
% % sizeX = size(X);
% % max(mean(reshape(X, [sizeX(1)*sizeX(2),sizeX(3)])))
% % 1709.2
% X = X(1:608,1:336,11:end);
% sizeX = size(X);
% load('R')
% R = R(:,1:end-10);
% C = fspecial('gaussian',[12 12],5);
% [Y,Z] =  DowsamplingRC2(X,R',C,12,8);
% [U,S,V] = svd(reshape(X, [sizeX(1)*sizeX(2),sizeX(3)]),'econ');
% S = S(1:30,1:30);
% V = V(:,1:30);
% U = U(:,1:30);
% mkdir('../RealData/Pavia1/');
% save('../RealData/Pavia1/XYZVS', 'X','Y','Z','V','Z','R','C')

% 
% X = double(imread('original_rosis.tif'))/5000;
% % sizeX = size(X);
% % max(mean(reshape(X, [sizeX(1)*sizeX(2),sizeX(3)])))
% % 1709.2
% X = X(1:608,1:336,11:end);
% sizeX = size(X);
% load('R')
% R = R(:,1:end-10);
% R = bsxfun(@times, R, 1./sum(R,2));
% C = fspecial('gaussian',[12 12],5);
% [Y,Z] =  DowsamplingRC2(X,R',C,12,8);
% [U,S,V] = svd(reshape(X, [sizeX(1)*sizeX(2),sizeX(3)]),'econ');
% S = S(1:30,1:30);
% V = V(:,1:30);
% U = U(:,1:30);
% mkdir('../RealData/Pavia1/');
% save('../RealData/Pavia1/XYZVS2', 'X','Y','Z','V','Z','R','C')

X = double(imread('original_rosis.tif'))/5000;
% sizeX = size(X);
% max(mean(reshape(X, [sizeX(1)*sizeX(2),sizeX(3)])))
% 1709.2
X = X(1:608,1:336,11:end);
sizeX = size(X);
load('R')
R = R(:,1:end-10);
R = bsxfun(@times, R, 1./sum(R,2))*2;
C = fspecial('gaussian',[12 12],5);
[Y,Z] =  DowsamplingRC2(X,R',C,12,8);
[U,S,V] = svd(reshape(X, [sizeX(1)*sizeX(2),sizeX(3)]),'econ');
S = S(1:30,1:30);
V = V(:,1:30);
U = U(:,1:30);
mkdir('../../RealData/Pavia/');
save('../../RealData/Pavia/XYZVS3', 'X','Y','Z','V','Z','R','C')
end


function [Y,Z] =  DowsamplingRC2(X,R,C,sizeC,ratio)
sizeX = size(X);
uX = reshape(X,[sizeX(1)*sizeX(2),sizeX(3)]);
Y = reshape(uX*R,[sizeX(1:2),4]);
Z = zeros([sizeX(1:2)/ratio, sizeX(3)]);
C = C(end:-1:1,end:-1:1);
Xpad = mypadding(X,round((sizeC-ratio)/2));

for i=1:sizeC
    for j = 1:sizeC
        Z = Z + Xpad(i:ratio:i+sizeX(1)-1,j:ratio:j+sizeX(2)-1,:)*C(i,j);
    end
end
end

function Xpad = mypadding(X,num)
X1 = [X(num:-1:1,:,:);X;X(end:-1:end-num+1,:,:)];
X0 = [X(num:-1:1,num:-1:1,:);X(:,num:-1:1,:);X(end:-1:end-num+1,num:-1:1,:)];
X2 = [X(num:-1:1,end:-1:end-num+1,:);X(:,end:-1:end-num+1,:);X(end:-1:end-num+1,end:-1:end-num+1,:)];
Xpad = [X0,X1,X2];
end