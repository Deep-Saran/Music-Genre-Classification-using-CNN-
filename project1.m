clear; clc;
addpath C:\Users\Deep_HP\Desktop\MI_Project\toolbox\machineLearning
addpath C:\Users\Deep_HP\Desktop\MI_Project\toolbox\sap
addpath C:\Users\Deep_HP\Desktop\MI_Project\toolbox\utility
auDir='C:\Users\Deep_HP\Desktop\MI_Project\genres';
opt=mmDataCollect('defaultOpt');
opt.extName='au';
auSet=mmDataCollect(auDir, opt, 1);
if ~exist('ds.mat', 'file')
	myTic=tic;
	opt=dsCreateFromMm('defaultOpt');
	opt.auFeaFcn=@mgcFeaExtract;	% Function for feature extraction
	opt.auFeaOpt=feval(opt.auFeaFcn, 'defaultOpt');	% Feature options
	opt.auEpdFcn='';		% No need to do endpoint detection
	ds=dsCreateFromMm(auSet, opt, 1);
	fprintf('Time for feature extraction over %d files = %g sec\n', length(auSet), toc(myTic));
	fprintf('Saving ds.mat...\n');
	save ds ds
else
	fprintf('Loading ds.mat...\n');
	load ds.mat
end
%%
auFile=[auDir, '/disco/disco.00001.au'];
figure; mgcFeaExtract(auFile, [], 1);
figure;
[classSize, classLabel]=dsClassSize(ds, 1);
figure; dsFeaVecPlot(ds); figEnlarge;
%%
c=[ds.input;ds.output]';
Completedata=shuffleRow(c);
TrainData=Completedata(1:800,:);
TestData=Completedata(801:1000,:);
%%
[trainedClassifier, TrainAccuracy] = projectSVM(TrainData);
yfit = trainedClassifier.predictFcn(TestData(:,1:156));
con=confusionmat(yfit,TestData(:,157))
accuracy = sum(diag(con))/sum(sum(con))*100