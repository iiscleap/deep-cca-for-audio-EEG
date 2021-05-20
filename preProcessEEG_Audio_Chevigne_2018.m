
function preProcessEEG_Audio_Chevigne_2018(subjectID)
% function to load the TRF data and plot the model for a single subject.

% addpath /Users/sri/Documents/speech/tools/rastamat/
% addpath /Users/sri/Documents/proposals/2019/Telluride-2019/sideProjects/TRF/mTRF_1.5
% addpath /Users/sri/Documents/speech/EEG_Analysis/EEGLab/eeglab14_1_1b/
% addpath ~/Documents/tools/NoiseTools/

addpath /home/sriram/speech/eegAnalysis/rastamat/
addpath /home/sriram/speech/eegAnalysis/TRF/mTRF_1.5/
addpath /data2/jaswanthr/github/working/codes2/NoiseTools/
% addpath /data2/jaswanthr/github/working/codes2/NoiseTools2/

% ----- Parameter Definition ----- %
% N denotes the number of trials per subject. 
N = 20;
fs = 64;
map=1 ;
tmin = 0;
tmax = 400;
lambda = 0.1;
eps=1e-6;
scale = 1e3 ;
stim = cell (1,N);
resp = cell (1,N);
hpcutoff = 0.1/(fs/2) ;
lpcutoff = 12/(fs/2) ;
[b1,a1] = butter(2,hpcutoff,'high');
[b2,a2] = butter(2,lpcutoff,'low');
signalFs=16000;

% Load the stimuli and response data
for I = 1 : N 
    if ~mod(I,5)
        disp (['Loading from trial ' num2str(I)]);
    end 
%      stm = load(['../Spectrogram/audioST_aespa_speech_' num2str(I) '.mat']);
%      stim{I} = (downsample((stm.STRFstimuli(:,:))',4)) ;
%      T = size(stim{I},1); 


%% %%%%%% Load the audio signal and compute the envelope 

     [x1,aFs] = audioread(['/home/sriram/speech/eegAnalysis/Telluride/TRF/audio/audio' num2str(I) '.wav']);
     x1 = x1 ;
     
     
%% Generation of Envelope Features     
     x1Square = mean(x1,2).^2 ;    % Square The Signal 
     squareWindow = rectwin(round(15.6*44.1)); % Rectangular window.
     x1Square = filter(squareWindow, 1, x1Square); % Convolution with rectangular window
     env = nt_resample_interp1(x1Square,fs,aFs);
     env(env<0) = 0;
     env = env.^(1/3);

   %  save(['../Spectrogram/melSpec/speech_' num2str(I) '.mat'],spec
     stim{I} = env ; 
     T = size(stim{I},1); 
     
%      %% Generation of Spectrogram Features.
%      % %%%%%% Spectrogram computation from rastamat
%      x1_16k = resample(mean(x1,2),signalFs,aFs);
%      [ceps,aspec,pspec] = melfcc(x1_16k,signalFs);
%     % aspec = aspec.^(0.1) ;
%      aspec = log(aspec+eps); 
%    %  save(['../Spectrogram/melSpec/speech_' num2str(I) '.mat'],spec
%      stim{I} = (aspec')/(max(max(((aspec)))));  % normalize the spectrogram magnitude
%      T = size(stim{I},1); 

%% %%%%%% Load the EEG signal and preprocess

    rsp = load(['/home/sriram/speech/eegAnalysis/TRF/Natural_Speech/EEG/Subject' num2str(subjectID) '/Subject' num2str(subjectID) '_Run' num2str(I) '.mat']);
  
    rsp.eegData = nt_resample_interp1(rsp.eegData,fs, rsp.fs); 
     
    if size(rsp.eegData,1) >= T
        currResp = rsp.eegData (1:T,:);
    else
        T = size(rsp.eegData,1);
        stim{I} = stim{I}(1:T,:);
        currResp = rsp.eegData (1:T,:);
    end
    
    currResp = double(nt_detrend(currResp,10));  % De-trend with polynomial
    currResp = nt_star2(currResp);
    currResp_high = filtfilt(b1,a1,currResp);  % High-pass filter
    currResp_band = filtfilt(b2,a2,currResp_high); % Low-pass filter
    
    % re-referencing using average
    currResp_band = currResp_band - repmat(mean(currResp_band,2),1,size(currResp_band,2));
    
    resp{I} = currResp_band ; 
end


save(['Subject' num2str(subjectID) '_Preprocessed_ENV_EEG.mat'],'stim','resp');



% % Perform TRF cross validation
% dbq
% disp('TRF Model Estimation and Cross Validation');
% 
% [R,P,MSE,PRED,MODEL] = mTRFcrossval(stim,resp,fs,map,tmin,tmax,lambda);
% disp(mean(mean(R,1),3));
% 
% disp ('Saving the model');
% save(['Subject' num2str(subjectID) '_origSpec.mat'],'R','P','MSE','PRED','MODEL');
% 
% disp ('Done');

    
    
    