clear all; close all; clc;  tic
currentFolder = pwd;
addpath(genpath(currentFolder))

load db821.mat db10
load db_label821.mat db_label10
  data1= db10{1};
data2=data1(:,142:951);
Xdata=data1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 label_all0=db_label10{1};
 aa=0;
 aa=0;
 labels0=[]; labels01=[]; l1=[];l2=[];l3=[];l4=[];l5=[];l6=[];l7=[];
% 
     for i1=1:size(label_all0,1)
       
              if label_all0(i1)==1
              label1=label_all0(i1);
              label00=[1 0 0 0 0 0 0]';
              l1=[l1;i1];
              end
               if label_all0(i1)==2
              label1=label_all0(i1);
              label00=[0 1 0 0 0 0 0]'
              l2=[l2;i1];
               end;
               if label_all0(i1)==3
              label1=label_all0(i1);
              label00=[0 0 1 0 0 0 0]'
              l3=[l3;i1];
               end;
               if label_all0(i1)==4
              label1=4;
              label00=[0 0 0 1 0 0 0]';
              l4=[l4;i1];
               end
              if label_all0(i1)==5
              label1=5;
              label00=[0 0 0 0 1 0 0]';
              l5=[l5;i1];
              end
              if label_all0(i1)==6
              label1=6;
              label00=[0 0 0 0 0 1 0]';
              l6=[l6;i1];
              end
              if label_all0(i1)==7
              label1=7;
              label00=[0 0 0 0 0 0 1]';
              l7=[l7;i1];
              end
    
     labels0=[labels0  label00];         
     labels01=[labels01;  label1];           
     end  
     label_all= labels0';

 result2=[];
 H=[];
 sum_acc=0;
for j=1:1
number=randperm(size(label_all,1));
number10{j,1}=number;
for i=1:10
if i~=10
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     writePath = ['idx_sdaPath'];
  logname = ['./' writePath '/' strcat('idx_sda',num2str(round(rand()*100000)))];
  if ~exist(logname,'dir')
        mkdir(logname);
  end    
train_x = [Xdata(number(1:241*(i-1)),:); Xdata(number(241*i+1:end),:)]';
test_x = Xdata(number(241*(i-1)+1:241*i),:)';

train_y =[label_all0(number(1:241*(i-1)),:) ; label_all0(number(241*i+1:end),:)]';

test_y =label_all0(number(241*(i-1)+1:241*i),:)';
 end
if i==10
    
train_x = Xdata(number(1:241*(i-1)),:)';
test_x = Xdata(number(241*(i-1)+1:end),:)';
train_y =label_all0(number(1:241*(i-1)),:)';
test_y =label_all0(number(241*(i-1)+1:end),:)';
end
  trainLabels=train_y';
        testLabels=test_y';
        traindata=train_x';
        testdata=test_x';
        trainLabels1{i,j}=trainLabels;
         testLabels11{i,j}=testLabels;

         [train_data,test_data] = featnorm(traindata,testdata);
         train_data = double(train_data*2-1);
         test_data = double(test_data*2-1);
%       [train_data,inputps]=mapminmax(train_data);
%      [test_data,outputps]=mapminmax(test_data);

     trainLabels =trainLabels;

    testLabels1 =  testLabels;%测试集标签  
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%尝试产生SDA
target= trainLabels;
  %SDA: stepwise discriminant analysis
 train_data_sda=train_data(:,142:951);

  idx_sda = sda2(train_data_sda,target,logname)   
   
   trainData=[train_data(:,1:36)  train_data(:,75:110) train_data(:,112:115) train_data_sda(:,idx_sda) train_data(:,952:end)];            %trainSet

  
     trainX=[trainData];
    test_data_sda=test_data(:,142:951);

       testData =[ test_data(:,1:36)   test_data(:,75:110)  test_data(:,112:115) test_data_sda(:,idx_sda)  test_data(:,952:end)];               %testSet

        
      testData1 = testData;            %testSet
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
 
   inputSize =size(trainX,1); 



 
 net=patternnet([50 30]);
 net.trainParam.epochs=1000; %最大训练次数（缺省值为10）
net.trainParam.show=25;%显示训练迭代过程（NaN表示不显示，缺省为25），每25次显示一次
net.trainParam.showCommandLine=0;%显示命令行（默认值是0） 0表示不显示
net.trainParam.showWindow=0; %显示GUI（默认值是1） 1表示显示
net.trainParam.goal=0.001;%训练要求精度（缺省为0）
net.trainParam.time=inf;%最大训练时间（缺省为inf）
net.trainParam.min_grad=1e-4;%最小梯度要求（缺省为1e-10）
net.trainParam.max_fail=5;%最大失败次数（缺省为5）
net.performFcn='mse';%性能函数

    %net.trainParam.showWindow=0;
    trainLabels7 = ind2vec( trainLabels');
    testLabels7 = ind2vec( testLabels');
    net = train(net,trainData',trainLabels7);
    y = net(testData');
    perf = perform(net,testLabels7,y);
  CatNames = {'1 Cytopl', '2 ER', '3 Gol', '4 Lyso', '5 Mito', ...
        '6 Nucl', '7 Vesi'};
    saveDir='./output/';
 ROCraw.true = testLabels7; ROCraw.predicted =y';
  AUC = showMyROC(ROCraw,  'ROC_Classify_BP network', CatNames,  saveDir); 
    meanAUC = mean(cell2mat(AUC))
    stdAUC = std(cell2mat(AUC))

%% 结果分析
%根据网络输出找出数据属于哪类
for i=1:size(testData1,1)
    if y(1,i) ~= nan
    output_fore(i)=find(y(:,i)==max(y(:,i)));
    else
      output_fore(i)=1;
    end
end

%BP网络预测误差
error=output_fore-test_y

sum_error=sum(error~=0)
sum_acc=sum_acc+sum_error
 result2=[result2,error];
 H=[];

figure(1)
plot(output_fore,'r')
hold on
plot(label_all0(number(inputSize+1:size(number,2)))','b')
legend(' Predicted Classes','Ture Classes')

%画出误差图
figure(2)
plot(error)
title('BP network classification error','fontsize',12)
xlabel('Human Tissue Image Numbers','fontsize',12)
ylabel('Classification Error','fontsize',12)
toc
end
acc=(size(number,2)-sum_acc)/size(number,2)
save bp_acc acc
%画出误差图
figure(3)
plot(result2)
title('BP network classification error','fontsize',12)
xlabel('Human Tissue Image Numbers','fontsize',12)
ylabel('Classification Error','fontsize',12)
toc

end