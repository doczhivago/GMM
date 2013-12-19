function results=batchTest
% Batch testing for sample recordings

labels=csvread('C:\School\Pattern Recogition\data\train_wo_headers.csv',0,1);
istart=3;
istop=100;
results=zeros((istop-istart+1),5); %stores results for ID rate calc
%whale=zeros((istop-istart+1),3);
%nowhale=zeros((istop-istart+1),3);

%test each sample and record result
i=1;
for ii=istart:istop
    results(i,1)=testSample(ii);
    results(i,2)=labels(ii);
    
    if results(i,1)~=results(i,2)
        % if classification error, classify as miss or false alarm
        if results(i,2)==0
            results(i,4)=1; %false alarm
        else
            results(i,5)=1; %miss
        end
    else
        %if correct classification, set result 3 to 1
        results(i,3)=1;
%         if labels(ii)==1
%             whale(i,1)=1;
%         end
    end
    i=i+1;
    
end

idrate=sum(results(:,3))/size(results,1); fprintf('\n ID rate %f \n',idrate);
falseAlarmRate=sum(results(:,4))/size(results,1);
fprintf('FA rate %f \n',falseAlarmRate);
missRate=sum(results(:,5))/size(results,1);
fprintf('Miss rate %f \n',missRate);
x=1;


function result=testSample(filenumber,demo)
% This function test a sample recording and returns whether it contains a
% whale or not.
%
%

if nargin==1
    demo=0;
else 
    demo=1;
    labels=csvread('C:\School\Pattern Recogition\data\train_wo_headers.csv',0,1);
end

% load GMM parameters
load('C:\Program Files\MatLAB v7.14 with Simulink v7.9\GMM.mat');

%extract feature vector from test sample
filename=['C:\School\Pattern Recogition\data\test\test' int2str(filenumber) '.aiff'];
fprintf('File %i ',filenumber);
[fV i_common1 i_common2]=extractFeatureVector(filename,demo);

if ~isempty(fV)
    result=classifyfV(fV, i_common1, i_common2,g1,g2);
else
    fprintf(' \n');
    result=0;
end

%plot results
if demo
    if labels(filenumber)==1
        ltext='Whale Present';
    else
        ltext='Whale Not Present';
    end
    if result
        wtext='Whale Detected';
    else
        wtext='Whale Not Detected';
    end
    screen_size = get(0, 'ScreenSize');
    f1=figure; 
    set(f1, 'Position', [10 150 screen_size(3)/2 screen_size(4)/1.5 ] );
    %subplot(3,1,1); pcolor(fV);
    %subplot(3,1,2); pcolor(fV);
    %subplot(3,1,3);
    pcolor(fV);
    title(sprintf('%s And %s ',ltext,wtext),'FontWeight','bold')
end



function result=classifyfV(fV,i_common1,i_common2,g1,g2)
%if ~isempty(fV)
    %first stage, threshold
    if ((i_common1(1)==7)&&(i_common2(1)==6))||((i_common1(1)==5)&&(i_common2(1)==6))
        % recording within threshold, classify as whale
        fprintf(' Whale detected by 1st Stage \n');
        result=1; return;
        %whaleFound=1;
    end
    if (i_common1(1)~=7)&&(i_common1(1)~=6)&&(i_common1(1)~=5)%((i_common1(1)~=7)&&(i_common2(1)~=6))||((i_common1(1)~=5)&&(i_common2(1)~=6))
        % recording completely outside threshold, classify as not whale
        fprintf(' \n');
        result=0; return;
        %whaleFound=1;
    end
    
    %if~whaleFound
        %second stage, GMM Classifer
        %evaluate test sample with
        for j=1:length(g1)
            p1(j)=evalGMM(fV,g1(j).prior,g1(j).mu,g1(j).sigma);
        end
        for j=1:length(g2)
            p2(j)=evalGMM(fV,g2(j).prior,g2(j).mu,g2(j).sigma);
        end
        
        % OR IF P for both is really low, label as whale not present!
        if sum(sum((p1)))>sum(sum((p2)))
            fprintf(' Whale detected by 2nd Stage \n');
            result=1; return;% whale is present
        else
            fprintf(' \n');
            result=0; return; % whale is not present
        end
    %else
        %fprintf(' \n');
        %result=0;
    %end
%else
%    fprintf(' \n');
%    result=0;
%end


