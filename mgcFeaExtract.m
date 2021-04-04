function [fea, mfcc]=mgcFeaExtract(au, opt, showPlot)
% feaExtract: Feature extraction for audio
%
%	Usage:
%		fea=mgcFeaExtract(au, opt, showPlot)
%
%	Example:
%		auFile='disco.00001.au';
%		au=myAudioRead(auFile);
%		opt=mgcFeaExtract('defaultOpt');
%		fea=mgcFeaExtract(au, opt, 1);

%	Category: Audio feature extraction
%	Roger Jang, 20150527

if nargin<1, selfdemo; return; end
if ischar(au) && strcmpi(au, 'inputName')	% Return the input names
	fea={'mean', 'std', 'min', 'max'};
	return
end
if ischar(au) && strcmpi(au, 'defaultOpt')	% Set the default options
	fea.dim=8;		% Dummy field to be added later
	return
end
if nargin<2||isempty(opt), opt=feval(mfilename, 'defaultOpt'); end
if nargin<3, showPlot=0; end

if ischar(au), au=myAudioRead(au); end

% === MFCC
mfccOpt=mfccOptSet(au.fs);
mfccOpt.frameSize=1024;
mfccOpt.overlap=512;
mfccOpt.useDelta=2;
mfcc=wave2mfcc(au.signal, au.fs, mfccOpt);

% === Feature extraction
fea(:,1)=mean(mfcc, 2);	% Mean
fea(:,2)=var(mfcc, 1, 2);	% Standard deviation
fea(:,3)=min(mfcc, [], 2);	% Min
fea(:,4)=max(mfcc, [], 2);	% Max
%fea(:,5)=mean(abs(diff(mfcc, 1, 2)), 2);		% Mean of abs. diff
%fea(:,6)=var(abs(diff(mfcc, 1, 2)), 2);		% Std of abs. diff
fea=fea(:);

if showPlot
	time=(1:length(au.signal))/au.fs;
	subplot(211); plot(time, au.signal); xlabel('Time (sec)'); title('Audio signals');
	subplot(212); plot(fea); xlabel('Feature index'); title('File-based features');
end

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);