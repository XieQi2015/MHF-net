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
save('../RealData/Pavia/XYZVS3', 'X','Y','Z','V','Z','R','C')