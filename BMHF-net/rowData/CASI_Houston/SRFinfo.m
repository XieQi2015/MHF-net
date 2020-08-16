
%
% wavelength = [450 520; 520 600; 630 690; 760 900; 1550 1750; 2080 2350]; % ETM/Landsat
% B G R NIR

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
