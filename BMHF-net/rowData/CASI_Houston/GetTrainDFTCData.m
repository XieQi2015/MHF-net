clear;
%% getTestData, 我们做8倍的下采样吧，这样训练集要大一些
tempX = single(imread('2013_IEEE_GRSS_DF_Contest_CASI.tif'));% 0.364 to 1046 um
tempX = tempX(5:end-5,:,:);
sizeX = size(tempX);
tempX = tempX/max(mean(reshape(tempX, [sizeX(1)*sizeX(2),sizeX(3)]))) *1709.2/5000;%%这里要注意所有的数据要变成 X/max(X)*8/5;

X1 = tempX(:,1:2*sizeX(1),:);
X2 = tempX(:,end-sizeX(1)+1:end,:);

save('../../RealData/trainX','X1','X2');


uX = [reshape(X1, [size(X1,1)*size(X1,2), sizeX(3)]); reshape(X2, [size(X2,1)*size(X2,2), sizeX(3)])];
Rk = 30;

[U,S,V] = svd(uX,'econ');
S = S(1:Rk,1:Rk);
V = V(:,1:Rk);
U = U(:,1:Rk);
reX = U*S*V';

error = sum((uX(:)-reX(:)).^2)/sum(uX(:));

save('../../RealData/VS','V','S');


X = tempX(:,2*sizeX(1)+1:end-sizeX(1),:);

save('../../RealData/testX','X');


