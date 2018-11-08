clear all; close all; clc;  tic
% addpath('../data');  
% addpath('../util'); 
currentFolder = pwd;
addpath(genpath(currentFolder))
%load PT;
load db821.mat db10
load db_label821.mat db_label10
  data1= db10{1};
data2=data1(:,142:951);
Xdata=data1;
 
% data2=data1(:,142:951);
% Xdata=double(reshape(data1,256,256,size(data1,2)))/255;

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
% load LETTERS;
%load LT;
%   
% train_x = double(reshape(train_x,28,28,140))/255;%训练样本为28*28的矩阵，总共有30个样本
% test_x = double(reshape(test_x,28,28,1))/255;  
% train_y = double(train_y'); 
% test_y =double(test_y');  

% train_x = double(reshape(p,28,28,30))/255; 
% test_x = double(reshape(test_x,28,28,10))/255; 
% train_y = double(t');  
% test_y =double(test_y');
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
train_y =[label_all(number(1:241*(i-1)),:) ; label_all(number(241*i+1:end),:)]';
test_y =label_all(number(241*(i-1)+1:241*i),:)';
 end
if i==10
    
train_x = Xdata(number(1:241*(i-1)),:)';
test_x = Xdata(number(241*(i-1)+1:end),:)';
train_y =label_all(number(1:241*(i-1)),:)';
test_y =label_all(number(241*(i-1)+1:end),:)';
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



% train_x = double(reshape(train_x,256,256,size(train_x,2)))/255;%训练样本为28*28的矩阵，总共有30个样本
% test_x = double(reshape(test_x,256,256,size(test_x,2)))/255;  
% train_y = double(train_y); 
% test_y =double(test_y);  

%从1到2000间随机排序
% k=rand(1,2000);
% [m,n]=sort(k);
[m,n]=sort(number);
%找出训练数据和预测数据
% input_train=input(n(1:1900),:)';
% output_train=output(n(1:1900));
% input_test=input(n(1901:2000),:)';
% output_test=output(n(1901:2000));
input_train=trainData';
output_train=trainLabels';
input_test=testData';
output_test= testLabels';
%选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
% [outputn,outputps]=mapminmax(output_train);

innum=size(input_train,1);
midnum=60;
outnum=7;
 

%权值初始化
w1=rands(midnum,innum);
b1=rands(midnum,1);
w2=rands(midnum,outnum);
b2=rands(outnum,1);

w2_1=w2;w2_2=w2_1;
w1_1=w1;w1_2=w1_1;
b1_1=b1;b1_2=b1_1;
b2_1=b2;b2_2=b2_1;

%学习率
xite=0.05
alfa=0.01;

%% 网络训练
for ii=1:50
    E(ii)=0;
    for i=1:1: inputSize
       %% 网络预测输出 
        x=inputn(:,i);
        % 隐含层输出
        for j=1:1:midnum
            I(j)=inputn(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        % 输出层输出
        yn=w2'*Iout'+b2;
        
       %% 权值阀值修正
        %计算误差
        e=output_train(:,i)-yn;     
        E(ii)=E(ii)+sum(abs(e));
        
        %计算权值变化率
        dw2=e*Iout;
        db2=e';
        
        for j=1:1:midnum
            S=1/(1+exp(-I(j)));
            FI(j)=S*(1-S);
        end      
        for k=1:1:innum
            for j=1:1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4));
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4));
            end
        end
           
        w1=w1_1+xite*dw1';
        b1=b1_1+xite*db1';
        w2=w2_1+xite*dw2';
        b2=b2_1+xite*db2';
        
        w1_2=w1_1;w1_1=w1;
        w2_2=w2_1;w2_1=w2;
        b1_2=b1_1;b1_1=b1;
        b2_2=b2_1;b2_1=b2;
    end
end
 

%% 语音特征信号分类
inputn_test=mapminmax('apply',input_test,inputps);

for ii=1:1
    for i=1:size(testData1,1)%1500
        %隐含层输出
        for j=1:1:midnum
            I(j)=inputn_test(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        
        fore(:,i)=w2'*Iout'+b2;
    end
end



%% 结果分析
%根据网络输出找出数据属于哪类
for i=1:size(testData1,1)
    output_fore(i)=find(fore(:,i)==max(fore(:,i)));
end

%BP网络预测误差
error=output_fore-label_all0(number(inputSize+1:size(number,2)))'

sum_error=sum(error~=0);
sum_acc=sum_acc+sum_error
 result2=[result2,error];
 H=[];
%画出预测语音种类和实际语音种类的分类图
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
%print -dtiff -r600 1-4
% %% 结果分析
% 
% figure(1)
% plot(BPoutput,':og')
% hold on
% plot(output_test,'-*');
% legend('预测输出','期望输出')
% title('BP网络预测输出','fontsize',12)
% ylabel('函数输出','fontsize',12)
% xlabel('样本','fontsize',12)
% %预测误差
% error=BPoutput-output_test;
% 
% 
% figure(2)
% plot(error,'-*')
% title('BP网络预测误差','fontsize',12)
% ylabel('误差','fontsize',12)
% xlabel('样本','fontsize',12)
% 
% figure(3)
% plot((output_test-BPoutput)./BPoutput,'-*');
% title('神经网络预测误差百分比')
% 
% errorsum=sum(abs(error));
% % --------------------- 
% 作者：kiomi_kiomi 
% 来源：CSDN 
% 原文：https://blog.csdn.net/qq_24182661/article/details/81254373 
% 版权声明：本文为博主原创文章，转载请附上博文链接！

% acc= mean(label_all(number) == result2)
% save   acc333 acc
end