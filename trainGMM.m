function [g1 g2]=trainGMM
% Gaussian Mixture Models Training with Expectation-Maximumization Algorithm
%
% This function trains 2 GMMs using the EM algoritm. The 2 GMMs are Whale
% Present, g1, and No Whale Present, g2. The input to the function are
% labeled audio recordings in .aiff format labeled as Whale Present or No
% Whale Present. From these recordings, audio spectrum feature vectors are
% extracted using a modified Mel-Frequency Cesptral Coefficient approach.
% The first 6 modified MFCC vectors are then used as input to the EM
% algorithm to estimate the parameters of the GMM: mu, sigma, weighting
% coefficients (prior probabilities). The GMM parameter estimates for each 
% model are initialized using kmeans clustering from 1 recording with a
% whale and 1 recording without a whale respectively. The trained GMMs are
% then saved and used in the testing phase.
%
% Key Variables:
%       fV - feature vector extracted from recording
%       g1 - GMM structure for whale, has fields
%                   - mu    = mean
%                   - sigma = covariance matrix
%                   - prior = prior probabilities, weighting coeff for GMM
%       g2 - GMM structure for no whale, has fields
%                   - mu    = mean
%                   - sigma = covariance matrix
%                   - prior = prior probabilities, weighting coeff for GMM
%

%load labels to identify recordings
labels=csvread('C:\School\Pattern Recogition\data\train_wo_headers.csv',0,1);
nGMM=3;

% if no model input
if nargin==0
    % initialize GMM for Whale, g1
    filename='C:\School\Pattern Recogition\data\test\test7.aiff';
    %filename='C:\School\Pattern Recogition\data\test\test12.aiff';
    fV=extractFeatureVector(filename); 
    [prior1, mu1, sigma1] = init_EM(fV, nGMM);
    %[prior1, mu1, sigma1] = EM_init_kmeans(fV, nGMM);
    clear fV
    
    % initialize GMM for No Whale,g1
    filename='C:\School\Pattern Recogition\data\test\test1.aiff';
    fV=extractFeatureVector(filename); 
    [prior2, mu2, sigma2] = init_EM(fV, nGMM);
    %[prior2, mu2, sigma2] = EM_init_kmeans(fV, nGMM);
    clear fV
end

%loop through aiff files and train GMMs
nTrainingSamples=1000;
whaleCount=1;
nowhaleCount=1;
for i=3:nTrainingSamples
    
    % check label on recording
    if labels(i)==1;
        whalePresent=1;
        fprintf('File %i Whale \n',i);
    else
        whalePresent=0;
        fprintf('File %i No Whale \n',i);
    end
    
    % extract feature vector from recording
    filename=['C:\School\Pattern Recogition\data\test\test' int2str(i) '.aiff'];
    if i==38
        x=1;
    end
    fV=extractFeatureVector(filename); 
    
    % train GMM for whale and no whale recordings
    if ~isempty(fV)
        if whalePresent
            %train Whale GMM with EM Algorithm
            %g1=trainEM(fV, g1);
            %fprintf(' Whale \n');
            [g1(whaleCount).prior, g1(whaleCount).mu, g1(whaleCount).sigma] = EM_Algo(fV, prior1, mu1, sigma1);
            if ~isnan(mu1t(1,1))
%                 prior1 = prior1t;
%                 mu1 = mu1t;
%                 sigma1 = sigma1t;
            else
                disp('Bad data')
            end
            clear fV
            whaleCount=whaleCount+1;
        else
            %train No Whale GMM with EM Algorithm
            %g2=trainEM(fV, g2);
            %fprintf(' No Whale \n');
            [g2(nowhaleCount).prior, g2(nowhaleCount).mu, g2(nowhaleCount).sigma] = EM_Algo(fV, prior2, mu2, sigma2);
            % check if update parameters are NaNs
            if ~isnan(mu2t(1,1))
%                 prior2 = prior2t;
%                 mu2 = mu2t;
%                 sigma2 = sigma2t;
            else
                disp('Bad data')
            end
            clear fV
            nowhaleCount=nowhaleCount+1;
        end
    end
    
end

% %save GMM for testing
% g1.prior=prior1;
% g1.mu=mu1;
% g1.sigma=sigma1;
% 
% g2.prior=prior2; 
% g2.mu=mu2;
% g2.sigma=sigma2;

%save results for testing phase
save('C:\Program Files\MatLAB v7.14 with Simulink v7.9\GMM_10Oct.mat','g1','g2');


function fV=extractFeatureVector(filename)
%read aiff file
x=double(aiffread(filename));
x=real(x);
n=length(x);

i_common1=0; i_common2=0;
%constants and initialze spectrum vector
nfft=512;
nt=20;
fs=2000;
dt=1/fs;
nf=nfft/2+1;
tt=dt*(nfft-1)/2:nt*dt:(n-1)*dt-(nfft/2)*dt;
ntt=length(tt);
y=zeros(nf,ntt);
s_energy=zeros(1,ntt);
%f=linspace(0,fs/2,nf)'*ones(1,ntt);
%t=ones(nf,1)*tt;

% x_f=zeros(n,1);
% x_f(1)=.03*x(1);
% for i=2:n
%     x_f(i)=x(i)-0.97*x(i-1);
% end

%lowpass filter at 500 Hz
%highfreq=500;fa=2000;[b,a] = butter(4, highfreq/fa*2, 'low')
b = [0.0940    0.3759    0.5639    0.3759    0.0940];
a =[1.0000   -0.0000    0.4860   -0.0000    0.0177];
x=filter(b,a,x);
%x=x_f;

% spectrogram of signal 
xw=(0:nfft-1)';
wind=.5*(1-cos((xw*2*pi)/(nfft-1))); %window
for i=1:ntt
    %fft
    zi=(i-1)*nt+1:nfft*i-(i-1)*(nfft-nt);
    xss=fft(x(zi).*wind,nfft)/nfft;
    yy=2*abs(xss(1:(nfft/2)+1));
    s_energy(i)=sum(abs(x(zi).^2));
    y(:,i)=yy; 
end
clear x

%remove silent periods and noise from spectrum
noise_threshold=1.2*mean(s_energy);
i_max=0;
ii_max=0;
ii_max2=0;
s_max=0;
ii=1;
y_n=[];
i_n=[];
for i=1:ntt
    if s_energy(i)>noise_threshold
        if s_energy(i)>s_max
            s_max=s_energy(i);
            ii_max2=ii_max;
            i_max=ii;
            ii_max=ii;
        end
        %y_n(:,ii)=y(:,i);
        i_n(ii)=i;
        %f_n(:,ii)=f(:,i);
        %t_n(:,ii)=t(:,i);
        ii=ii+1;
        
    end
end

if isempty(i_n)
    fV=[];
    return;
else
    y_n=y(:,i_n);
end
clear y
% %t=ones(nf,1)*(1:ntt);
% set(gca,'YTickLabel',[200 400 600 800 1000]);
% xlabel('Frame Number');
% ylabel('Frequency (Hz)');
% shading flat

%calculate modified MFCC
n_MFCC=25;
n_CC=12;
ntt=size(y_n,2);
nff=size(y_n,1);
nn=round(nff/n_MFCC);
%x_c=linspace(0,1000,nn); %approx freq centers
xw=[0:nn-1]';
L=length(xw)*2;
%L=nn;
twind=zeros(1,L);
freq_max_c=zeros(1,n_MFCC);
%create triangular window function, change to constant later
if mod(L,2)
    for i=1:L
        if i<((L+1)/2+1)
            twind(i)=2*i/(L+1);
        else
            twind(i)=2-(2*i)/(L+1);
        end
    end
else
    %twind=[0.1,0.3,0.5,0.7,0.9,0.9,0.7, 0.5,.3 .1];
    for i=1:L
        if i<((L)/2+1)
            twind(i)=(2*i-1)/L;
        else
            twind(i)=2-(2*i-1)/L;
        end
    end
end
%wind=.5*(1-cos((xw*2*pi)/(nn-1)));
c=zeros(n_MFCC-1,ntt);
cc=zeros(n_CC,ntt);
for j=1:ntt
    for i=1:n_MFCC-1
    ii=(nn*(i-1)+1):(nn*(i-1)+L);
    %iii=((i-1)*(L/2)+1):((i-1)*(L/2)+L)
    y_temp=y_n(ii,j).*twind';
    c(i,j) =sum(y_temp);
    %twind_x(:,i)=ii;
    %twind_y(:,i)=twind;
   % c(i,j)=log10(sum_wind);
    %c(i,j)=(sum_wind);
    end
    
%     twind_x=4*twind_x;
%     figure; hold on;
%     for i=1:size(twind_y,2)
%         if mod(i,2)
%             plot(twind_x(:,i),twind_y(:,i),'r');
%         else
%             plot(twind_x(:,i),twind_y(:,i),'b');
%         end
%     end
%     xlabel('Frequency (Hz)');
%     ylabel('Magnitude');
    
    %Cepstral analysis
    for ci=1:n_CC
    sum_CC=0; dCC=0;
        for i=1:15%n_MFCC-1
            %dCC=log10(c(i,j))*cos(ci*(i-.5)*pi/(n_MFCC-1));
            if i==1
                w=1/sqrt((n_MFCC-1));
            else
                w=sqrt(2/(n_MFCC-1));
            end
            dCC=w*log10(c(i,j))*cos(pi*(2*i-1)*(ci-1)/(2*(n_MFCC-1)));
            sum_CC=sum_CC+dCC;
        end
        cc(ci,j)=sum_CC;
    
    end  
    
    %most frequent max cc
    try
    i_max_c=find(c(:,j)==max(c(:,j)));
    freq_max_c(1,i_max_c)=freq_max_c(1,i_max_c)+1;
    catch error
        disp(error);
    end
end
%clear c
% if signal is all noise, exit
if isempty(cc)||(size(cc,2)<2)
    %disp('Whale not found')
    fV=[];
    return; 
end

% %first stage classifer
% %Check 4th and 5th filter bacnk c_n with max energy 
% c_threshold=4.5*mean(mean(c));
% if ((c(7,ii_max)>c_threshold)||(c(6,ii_max)>c_threshold))&&(c(8,ii_max)<c_threshold)  %and not 5 or 6
%     disp('The white whale')
% end
% figure; plot(c(:,ii_max));hold on; plot([0 size(c,1)],[c_threshold c_threshold]);

%find top two most common frequencies
i_common1=find(freq_max_c==max(freq_max_c));
freq_max_c(i_common1)=0;
i_common2=find(freq_max_c==max(freq_max_c));
if ((i_common1(1)==7)&&(i_common2(1)==6))||((i_common1(1)==5)&&(i_common2(1)==6))
    disp('The white whale NEW')
end
close all

%    plot([0 size(c,1)],[c_threshold c_threshold]);

% demean and return feature vector, for better convergence of EM
%fV=cc(1:3,:)-repmat(mean(cc(1:3,:),2)',ntt,1)';
cc=cc-repmat(mean(cc,2)',ntt,1)';
fV=cc;%(1:10,:);


% data=cc_n(1:5,:);
% [prior, mu, sigma] = EM_init_kmeans(data, 3);
% [prior, mu, sigma, Pix] = EM(data, prior, mu, sigma);
% p=evalGMM(data,prior,mu,sigma)




function p=evalGMM(data,prior,mu,sigma)
    nGMM=size(sigma,3);
    %[nVar,nDim] = size(data);
    [D,N] = size(data);
    %pComp=zeros(nGMM,1);
    for i=1:nGMM
        tdata = data' - repmat(mu(:,i)',N,1);
        %data = data' - repmat(mu(:,i)',N,1);
%p = sum((data*inv(sigma)).*data, 2);
        ptemp = sum((tdata/sigma(:,:,i)).*tdata, 2);
        %ptemp = ptemp/(sum(ptemp));
        pComp(:,i) = prior(i)*exp(-0.5*ptemp) / sqrt((2*pi)^D * (abs(det(sigma(:,:,1)))+realmin));
        
    end
    p=sum(pComp);
