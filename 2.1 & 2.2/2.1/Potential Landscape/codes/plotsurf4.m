close all

%% load Fokker Plank result
a = 'n_00100_all_over.dat';    
nd=load(a);

%% get the sizes and shape
N2=size(nd);
N=sqrt(N2(1));
NX=N; 
NY=N;
P=reshape(nd(:,1),NY,NX);
P = P';

%% get x y z data
y=15000./N*(1:N);
x=450/N*(1:N);
[X,Y] = meshgrid(x,y);
E = -log(abs(P));

%% crop it
Z_limit_h = 26;
Z_limit_l = -5;
E2 = E;

E3 = E2>25;
E4 = bwlabel(E3);
E2(E4==1) = nan;
E5 = ~isnan(E2);
xx=[150 360];
yy=[12500 3000];
para = polyfit(xx,yy,1);

xcutoff = 365;
ycutoff = 12500;
E2(X>xcutoff|Y>ycutoff) = nan;
E2(Y>(para(1)*X+para(2))) = nan;

%% plotting
bf=figure(1);
colormap(jet);
surf(X,Y,E2,'FaceColor','interp',...
   'EdgeColor','none',...
   'FaceLighting','gouraud');

%% set plotting parameters
zlim([Z_limit_l Z_limit_h]);
ylim([0 15000])
% xlim([0 380])
hold on

%% plot contour
[M,hContour] = contour(X,Y,E);
hContour.ContourZLevel = Z_limit_l;
hContour.LevelList = -10:2:Z_limit_h;
caxis([Z_limit_l+2 Z_limit_h]);
view(110,15)
%% other parameters 
ah = gca;
ah.LineWidth = 1.5;
ah.GridAlpha = 0.35;
ah.XTickLabel=[];
ah.YTickLabel=[];
ah.ZTickLabel=[];
%% save fig
set(gcf,'renderer','opengl');
saveas(gcf, 'all_over_labeled.eps','epsc2');




