% R = permute(allR(1,:,:),[2,3,1]);
% X = 364:(1046-364)/143:1046;
% hold off;
% plot(X,R(:,1),'linewidth',3,'color',[0.2,0.2,0.8])
% hold on;
% plot(X,R(:,2),'linewidth',3,'color',[0.2,0.8,0.2])
% plot(X,R(:,3),'linewidth',3,'color',[0.8,0.2,0.2])
% plot(X,R(:,4),'linewidth',3,'color',[0.8,0.2,0.8])
% axis([364,1046, 0, 0.05])
% legend('Blue','Green','Red','NIR');
% xlabel('Wavelength')
% ylabel('Response')

load('R.mat')
% load('XYZVS.mat')
% R = R(:,1:93)';
R = bsxfun(@times, R, 1./sum(R,2))*2;
R = R';
X = 430:(838-430)/102:838;
hold off;
plot(X,R(:,1),'linewidth',3,'color',[0.2,0.2,0.8])
hold on;
plot(X,R(:,2),'linewidth',3,'color',[0.2,0.8,0.2])
plot(X,R(:,3),'linewidth',3,'color',[0.8,0.2,0.2])
plot(X,R(:,4),'linewidth',3,'color',[0.8,0.2,0.8])
% axis([430,850, 0, 0.07])
% axis([430,850, 0, 1.2])
axis([430,850, 0, 0.14])
legend('Blue','Green','Red','NIR');
xlabel('Wavelength')
ylabel('Response')