clear;
% information of normal sensors, range of four bands
WV2  = [442,515; 506,586; 624,694; 765, 901];
WV3  = [427.9, 535.9; 485.3,608.9; 601.6,718.6; 723.6000, 924.4];
QuikBird  = [430, 545; 466,620; 590,710; 715, 918];

center = [mean(WV2,2),mean(WV3,2),mean(QuikBird,2)];

wide = [];
a = WV2;
wide = [wide, a(:,2)-a(:,1)];
a = WV3;
wide = [wide, a(:,2)-a(:,1)];
a = QuikBird;
wide = [wide, a(:,2)-a(:,1)];

mmC = [min(center,[],2),max(center,[],2)];
mmW = [min(wide,[],2),max(wide,[],2)];

mkdir('../../RealData');
save('../../RealData/SRFinfo','mmC','mmW');

%% getTrainData
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

%% getTestData
X = tempX(:,2*sizeX(1)+1:end-sizeX(1),:);

save('../../RealData/testX','X');


