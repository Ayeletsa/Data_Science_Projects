%% Final assignment - Ayelet Sarel
close all;
clear all;
%% Q1
%=========================================================
% load the data for the question:
load('observations_1.mat')
%x=rabbit
%y=sheep

% 1. a. plot the scatter:
%------------------------------------------------------
figure('units','normalized','outerposition',[0 0 1 1])
set(gcf,'DefaultAxesFontSize',8);
label_font_size=10;
title_font_size=12;

subplot(3,2,1)
scatter(x,y)
xlabel('Rabbits','FontSize',label_font_size)
ylabel('Sheep','FontSize',label_font_size)
title('Observations of different populations of R&S','FontSize',title_font_size)
axis([1 3.5 0.5 2.5])

% 1. b. k-means clustering
%------------------------------------------------------
% Define k means options:
replic_num=10;
k=9;
opts = statset('Display','final');
% perform k-means:
[idx,C] = kmeans([x;y]',k,'Distance','sqeuclidean' ...
    ,'Replicates',replic_num,'Options',opts);

% 1. inside the loop I have answers to b c and d:
%    b      plot the observation in different colors
%    c      find radius and plot the circles
%    d      find the cluster disatnces from centroid std
%------------------------------------------------------

% Plot:
subplot(3,2,2)
std_distance=NaN*ones(1,length(C));
centroid_r=NaN*ones(1,length(C));

for cluster_i=1:length(C)
    %plot clusters in different colors
    plot(x(cluster_i==idx),y(cluster_i==idx),'.');hold all;
    
    % adding the cluster number at the centroid
    cx=C(cluster_i,1);
    cy=C(cluster_i,2);
    text(cx,cy,sprintf('%d',cluster_i),'FontSize',9,'HorizontalAlignment','center')
    data_points=[x;y];% combine x and y for short calculation of the radius
    if length(x(cluster_i==idx))>1 %if the cluster has only one data point there is no meaning to the radius
        % compute radius length
        centroid_r(cluster_i)=max(sqrt(sum(((repmat(C(cluster_i,:),sum(cluster_i==idx),1)-data_points(:,cluster_i==idx)').^2),2)));
        % plot circle using circle.m
        circle(cx,cy,centroid_r(cluster_i))
    end
    % for d - computing cluster disatnces from centroid std
    std_distance(cluster_i)=std(sqrt(sum(((repmat(C(cluster_i,:),sum(cluster_i==idx),1)-data_points(:,cluster_i==idx)').^2),2)));
end
axis([1 3.5 0.5 2.5])
xlabel('Rabbits','FontSize',label_font_size)
ylabel('Sheep','FontSize',label_font_size)
title('Clustering using k-means','FontSize',title_font_size)
% d. subplot with several plots as requested:
%----------------------------------------------------------------------
subplot(3,2,3)
%  Plot the cluster radius as function of cluster number
plot(centroid_r,'o'); hold all;
xlabel('cluster #','FontSize',label_font_size)
% plot std as function of cluster number
plot(std_distance,'o')
% ratio between std and radius:
alpha_ratio=centroid_r./std_distance;
% choose one alphs out of all ratios - choosing the closest integer to
% most (in this case it is actually one integer that its all):
alpha_approx=mode(nearest(alpha_ratio));
% plot:
plot(alpha_approx*std_distance,'o')
legend1=legend('radius','std','\alpha\sigma')
set(legend1,...
    'Position',[0.0250263435194938 0.43853854488793 0.0713909378292938 0.124503525978763]);
set(legend1,'FontSize',10);
xlim([0.5 9.5])

% find the probability for observation larger than the radius in each
% cluster:
p_larger_than_alphasigma=NaN*ones(1,length(C));
p_larger_than_radius=NaN*ones(1,length(C));
for c_i=1:length(C)
    p_larger_than_alphasigma(c_i)=1-normcdf(alpha_approx*std_distance(c_i),0,std_distance(c_i));
    p_larger_than_radius(c_i)=1-normcdf(centroid_r(c_i),0,std_distance(c_i));
end
% Give title according to the result:
p_requested=0.001;
if ~isempty(find(p_larger_than_radius<p_requested))
    title(sprintf('For all clusters the probability that there is an observation\nlarger than the radius is lower than %.3f',p_requested),'FontSize',title_font_size)
else
    title(sprintf('Not for all clusters the probability that\nthere is an observation larger than the radius\nis lower than %.3f',p_requested),'FontSize',title_font_size)
end


%% Q2
%=========================================================
%x=rabbit
%y=sheep

% a. find a1 and a2 for all clusters:
%------------------------------------------------------------
% clusters centroid are the fixed point - the calculation of a1 a2 are from
% the
a1=NaN*ones(1,length(C));
a2=NaN*ones(1,length(C));
%subplot(3,2,4)

for cluster_i=1:length(C)
    cx=C(cluster_i,1);
    cy=C(cluster_i,2);
    a1(cluster_i)=(2*(cx.^2)+cx.*cy-2)./cx;
    a2(cluster_i)=(cy.^2+cx.*cy-2)./cy;
end
% plot:
subplot(3,2,4)
plot(a2,a1,'x')
xlabel('a2','FontSize',label_font_size)
ylabel('a1','FontSize',label_font_size)
title('Parameters values for the different clusters','FontSize',title_font_size)
axis([0.5 3.5 3.5 6.5])


% d. Plot the real and the imaginary parts:
%--------------------------------------------------------------------------
for cluster_i=1:length(C)
    cx=C(cluster_i,1);
    cy=C(cluster_i,2);
    % the Jacobian calculation from q2c:
    jacobian=[a1(cluster_i)-4*cx-cy,-cx;...
        -cy,a2(cluster_i)-cx-2*cy];
    % finding the eigenvalues:
    [V,D] = eig(jacobian);
    
    subplot(3,2,5)
    plot([cluster_i cluster_i],real(diag(D)),'x');hold on;
    xlabel('cluster #','FontSize',label_font_size)
    ylabel('Real part value','FontSize',label_font_size)
    title('Real part of eigenvalues per cluster','FontSize',title_font_size)
    xlim([0.5 9.5])
    
    subplot(3,2,6)
    plot([cluster_i cluster_i],imag(diag(D)),'x');hold on;
    xlabel('cluster #','FontSize',label_font_size)
    ylabel('Imaginary part value','FontSize',label_font_size)
    title('Imaginary part of eigenvalues per cluster','FontSize',title_font_size)
    xlim([0.5 9.5])
    
end

%% Q3
figure('units','normalized','outerposition',[0 0 1 1])
set(gcf,'DefaultAxesFontSize',8);
label_font_size=12;
title_font_size=14;
% 3a. Solve for specific IC and paramteres:
%--------------------------------------------------------
subplot(1,2,1)
% Possible values of a1 and a2:
a1_vec=[5,4];
a2_vec=[1,2];
% Initial conditions:
x0=4;
y0=6;
% Integration time:
tEnd=0.5;
dt=0.05;

for rep_i=1:length(a1_vec)
    
    a1_cur=a1_vec(rep_i);
    a2_cur=a2_vec(rep_i);
    Y=SolveY_nonlinear(x0,y0,a1_cur,a2_cur,tEnd,dt);
    plot(Y(1,:),Y(2,:));hold all
    end_points(:,rep_i)=[Y(:,end)];
   
    str{rep_i}=['a1=',num2str(a1_cur),' a2=',num2str(a2_cur)];
   
end
plot(end_points(1,:),end_points(2,:),'k.','MarkerSize',15)
str{rep_i+1}='Final points';
xlabel('Rabbits','FontSize',label_font_size)
ylabel('Sheep','FontSize',label_font_size)
title('Chanage of populations sizes over time','FontSize',title_font_size)
legend1=legend(str);
set(legend1,...
    'Position',[0.140937829293993 0.777528977871442 0.0866701791359323 0.0900948366701788]);
set(legend1,'FontSize',10);

% 3.b plot the cost function
%--------------------------------------------------------------------------
%load the data
load('observations_2.mat')

%possible range of parameters based on q2:
num_points=100;
possible_range_a1=linspace(min(a1),max(a1),num_points);
possible_range_a2=linspace(min(a2),max(a2),num_points);
% The final population size observed:
Y_obs=[x;y];
% compute cost function for different stes of parameters:
for ii=1:length(possible_range_a1)
    for jj=1:length(possible_range_a2)
        a1_cur=possible_range_a1(ii);
        a2_cur=possible_range_a2(jj);
        %solve the equation
        Y=SolveY_nonlinear(x0,y0,a1_cur,a2_cur,tEnd,dt);
        L_all(ii,jj)=0.5*(Y(:,end)-Y_obs)'*(Y(:,end)-Y_obs);
    end
end
%plot:
subplot(1,2,2)
contourf(possible_range_a1,possible_range_a2,L_all',40);
colormap(flipud(gray))
hold on
xlabel('a1','FontSize',label_font_size)
ylabel('a2','FontSize',label_font_size)
% plot the + where cost function is minimal:
plot(max(a1),2.12,'k+','MarkerSize',15)

% 3c. steepest descent
%--------------------------------------------------------------------------
% Initial guess for the parameters:
a1_initial_guess=4.5;
a2_initial_guess=1;
use_line_search=1; %non linear

% set x - with a1 and a2 of initial guess:
x=[a1_initial_guess;a2_initial_guess];
%solve the equation
Y=SolveY_nonlinear(x0,y0,a1_initial_guess,a2_initial_guess,tEnd,dt);
%plot initial guess point on the graph:
L_new=0.5*(Y(:,end)-Y_obs)'*(Y(:,end)-Y_obs);
plot(a1_initial_guess,a2_initial_guess,'ro','MarkerFaceColor','r')
text(a1_initial_guess,a2_initial_guess+0.1,'Steapest Decent','Color','r')

%steapest decend with optimal line search
dx=1e-10;
tol=10^-6;

while L_new>tol
    
    x_old=x;
    L_old=L_new;
  
    Y=SolveY_nonlinear(x0,y0,x(1),x(2),tEnd,dt);
    L_base=0.5*(Y(:,end)-Y_obs)'*(Y(:,end)-Y_obs);
    Y=SolveY_nonlinear(x0,y0,x(1)+dx,x(2),tEnd,dt);
    L_1=0.5*(Y(:,end)-Y_obs)'*(Y(:,end)-Y_obs);
    Y=SolveY_nonlinear(x0,y0,x(1),x(2)+dx,tEnd,dt);
    L_2=0.5*(Y(:,end)-Y_obs)'*(Y(:,end)-Y_obs);
    
    g0=[-(L_1-L_base)/dx;-(L_2-L_base)/dx]; %calc gradient in a general way
    r0=g0;
    
    %calc gradient
    Yr0=SolveY_nonlinear(x0,y0,x(1)+r0(1),x(2)+r0(2),tEnd,dt)-SolveY_nonlinear(x0,y0,x(1),x(2),tEnd,dt);
    
    a=r0'*r0/(Yr0(:,end)'*Yr0(:,end));
    
    if use_line_search
        %line search
        x_new=x+a*r0;
        Y=SolveY_nonlinear(x0,y0,x_new(1),x_new(2),tEnd,dt);
        L_new=0.5*(Y(:,end)-Y_obs)'*(Y(:,end)-Y_obs);
        while ((L_base-L_new)<=0.5*a*r0'*r0)
            a=a*0.9;
            x_new=x+a*r0;
            Y=SolveY_nonlinear(x0,y0,x_new(1),x_new(2),tEnd,dt);
            L_new=0.5*(Y(:,end)-Y_obs)'*(Y(:,end)-Y_obs);
        end
    end
    x=x+a*r0;
    
    Y=SolveY_nonlinear(x0,y0,x(1),x(2),tEnd,dt);
    L_new=0.5*(Y(:,end)-Y_obs)'*(Y(:,end)-Y_obs);
    plot(x(1),x(2),'ro','MarkerFaceColor','r')
    delta_x=x-x_old;
    quiver(x_old(1),x_old(2),delta_x(1),delta_x(2),1.0,'r')
    drawnow
end

% 3d. add the title to the graph according to the cluster:
%----------------------------------------------------------------------------
%find the closest cluster:
dist=pdist([x(1),a1;x(2),a2]');
sq_dist=squareform(dist);
[~,closest_cluster_ind]=min(sq_dist(1,2:end));
title(sprintf('Cost Function\nThe solution is a_1=%.2f a_2=%.2f, fits best to cluster %d',x(1),x(2),closest_cluster_ind),'FontSize',title_font_size)

%%
function Y=SolveY_nonlinear(x,y,a1,a2,tEnd,dt)
% solve pendolum equation

% constants

t0   =   0;         % starting time [years]
Time = t0:dt:tEnd;
NumSteps=floor(tEnd./dt);
var=[x;y];
Fun=@(var) [var(1).*(a1-2.*var(1)-var(2))+2; var(2).*(a2-var(1)-var(2))+2];

Y=zeros(2,NumSteps);
Y(:,1)=[x;y];
%solve the equation
for n=2:1:NumSteps+1
    k1=dt.*Fun(Y(:,n-1)       );
    k2=dt.*Fun(Y(:,n-1)+0.5*k1);
    k3=dt.*Fun(Y(:,n-1)+0.5*k2);
    k4=dt.*Fun(Y(:,n-1)+k3    );
    Y(:,n)=Y(:,n-1)+1/6*(k1+2*k2+2*k3+k4);
end
end
