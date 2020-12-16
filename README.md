# DEEP CCA ANALYSIS FOR AUDIO-EEG DATA

We have implemented the presented CCA methods on two datasets.
1. SPEECH - EEG Dataset by Liberto et al.
2. NMEDH (MUSIC-EEG) Dataset by Kaneshiro et al.

Can be easily extended to any other dataset with a common stimulus for multiple subjects' brain signals.

(Tested for both the linear methods, being tested for deep methods).

From speech dataset, 6 subjects are chosen and experimented on. Each subject has 20 blocks of Audio-EEG data.
From NMEDH, all subjects were used.

## WHAT DO WE HAVE HERE?
Codes for performing four CCA methods on audio-EEG datasets.
1. Linear CCA
2. Deep CCA
3. Linear MCCA + Linear/Deep CCA
4. Deep MCCA + Linear/Deep CCA

## HOW TO USE THE CODES?
All of our methods are performed on presaved data as follows.

### Speech Dataset
1. We have chosen 6 subjects from the dataset.
2. Preprocessed the EEG and the common stimuli as mentioned by Alain de Cheviegné et al.
3. Saved the 20 cross-fold validation data of all subjects' EEG Recordings and common stimuli as f"data_raw_{block}.pkl".
   1. It contains a list of 7 elements. 6 subject's EEG as first six elements and the last element as the common stimuli.
   2. Each element is a list of 3 elements. [Training_data, Validation data, Test_data]. Training_data has 18 blocks, Validation_data has 1 block and Test_data has 1 block.
   3. Each of these lists' elements are in the shape: Time-length x Dimen
   4. sion
   5. Dimension = 128D for EEG and 1D for the preprocessed Auditory Envelope.
4. Processed the EEG and stimuli using PCA and filterbank as proposed for LCCA3 method by Cheviegné et al.
5. Saved these 20 cross-fold validation data also similar to the preprocessed data. Saved as f"data_{block}.pkl".
   1. It contains a list of 7 elements. 6 subject's EEG as first six elements and the last element as the common stimuli.
   2. Each element is a list of 3 elements. [Training_data, Validation data, Test_data]. Training_data has 18 blocks, Validation_data has 1 block and Test_data has 1 block.
   3. Each of these lists' elements are in the shape: Time-length x Dimension
   4. Dimension = 139D for EEG and 21D for the preprocessed Auditory Envelope.
6. These files are used to perform the four CCA methods.
 
### NMEDH Dataset
1. We have chosen the "Clean EEG" files, for all the 16 stimuli.  
2. Extracted 20 features for all the stimuli as proposed by Alan et al. Extracted three features from them: PC1, Spectral Flux, RMS.
3. Therefore, four stimuli features are considered: [ "ENVELOPE", "PC1", "FLUX", "RMS" ].
4. For LCCA/DCCA tasks, each subject's responses and their corresponding stimuli are aggregated.
5. We have saved such 48 files as f"sub_{sub_num}.mat".
6. Its contents are as follows:
   1. resp_data_a - All Trial A data. Shape: [4, T, 125] (Each are the 4 trials)
   2. resp_data_b - All Trial A data. Shape: [4, T, 125] (Each are the 4 trials)
   3. resp_tr_a, resp_tr_b , resp_val_a , resp_val_b , resp_te_a , resp_te_b - Training, validation and test data of each trail, after dividing each trial into 90-5-5 splits.
   4.  stim_data  - All four stimuli preprocessed 3D stimuli. ("PC1", "FLUX", "RMS"). Shape: [4, T, 3]
   5.  stim_tr_3d, stim_val_3d, stim_te_3d    - Training, validation and test data of the 3D stimuli (repeated twice for the trials), after dividing each trial into 90-5-5 splits.
   6.  stimulus_tr, stimulus_val, stimulus_te - Training, validation and test data of the 1D ENV stimuli (repeated twice for the trials), after dividing each trial into 90-5-5 splits.
7. Loaded each of them for each subject's LCCA and DCCA Analysis.
8. For inter-subject analysis, Each stimulus' 12 EEG responses are aggregated and saved in a f"mcca_{stim}.pkl" file.
   1. Contains a list of 2 elements.
   2. First element: A list of 12 elements. Each element is a subject's EEG response to the common stimuli.
   3. Second element: A list.
   4. These 13 elements from the two lists are arranged as : [Training_data, Validation_data, Test_data].
   5. Each subject's EEG and the stimuli (from each trial) is split into 90-5-5 for the training-validation-test data. And then both trials' data is concatenated.
   6. The EEG data are of shape T x 125D
   7. The stimuli data are of shape T x 4. The 4 features are "ENV", "PC1", "FLX" and "RMS" respectively.
9.  Now both MCCA are applied on each stimulus feature (of 4) for each stimulus (of 16). 
10. After that, LCCA and DCCA are performed for intra-subject analysis.


## What is present in each file?
1. cca_functions: 
   1. cca_model  - MAIN CCA Method file.
   2. linear_mcca_with_stim - MAIN MCCA Method file. Performing LMCCA for all subjects' responses + common stimuli, and then LCCA for each subject's new stimulus and response.
   3. linear_mcca_resps_only - Performing LMCCA for all subjects' responses and then LCCA for each subject.
   4. linear_cca  - a custom linear cca  model
   5. linear_mcca - a custom linear mcca model
   6. my_corr    - for calculating correlation between two signals.
   7. my_mcca    - for performing linear mcca analysis and returning the transforms.
   8. pca_stim_filtem_pca_resp, pca_filt_pca_resp - for performing PCA, filterbank as proposed by Cheviegné et al.
   9. filtem, filtone - for filtering data by the FILTERBANK.
   10. lagGen     - for generating time-lags for a given data.
   11. my_standardize, my_PCA, apply_PCA, standardize_1 are helper functions for standardizing data and performing PCA.
2. deep_nets   - All the deep CCA networks are available here.
3. deep_losses - Losses required for the deep CCA and MCCA methods. 
4. deep_models - Deep CCA models
5. dcca.py  - For performing Deep CCA   method on Speech dataset or NMEDH dataset or any custom dataset.
6. dmcca.py - For performing Deep MCCA   + L/D CCA on Speech dataset or NMEDH dataset or any custom dataset.
7. lcca.py  - For performing Linear CCA method on Speech dataset or NMEDH dataset or any custom dataset.
8. lmcca.py - For performing Linear MCCA + LCCA   on Speech dataset or NMEDH dataset or any custom dataset.
9. lmdc.py  - For performing Linear MCCA + L/DCCA on Speech dataset or NMEDH dataset or any custom dataset.
10. music_helper.py  - Helper function for aggregating all the trails of subject into a single stimulus-response format.
11. speech_helper.py - Helper functions to load the presaved CCA and MCCA speech data.


