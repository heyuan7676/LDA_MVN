    
training_input = loadmat('/work-zfs/abattle4/heyuan/tissue_spec_eQTL/MF/input/cb_cis_eQTL_v7_filterGenes_tagSNPs_r20.8.mat')
Y_toFit_training = np.transpose(training_input['X'])
Y_SD_training = np.transpose(training_input['SD'])
P_training = np.array([x[0] for x in training_input['pairs'][0]])

missing_idx = np.where(np.isnan(Y_SD_training))[0]
complete_idx = np.sort(list(set(range(len(Y_SD_training))) - set(missing_idx)))

Y_toFit_training = Y_toFit_training[complete_idx,:]
Y_SD_training = Y_SD_training[complete_idx, :]
P_training = P_training[complete_idx]

Y_SD_training = 1/Y_SD_training
Y_SD_training = (Y_SD_training - np.min(Y_SD_training)) / (np.max(Y_SD_training) - np.min(Y_SD_training))








    
original_input = loadmat('/work-zfs/abattle4/heyuan/tissue_spec_eQTL/MF/input/cb_cis_eQTL_v7_filterGenes.mat')
Y_toFit = np.transpose(original_input['X'])
Y_SD = np.transpose(training_input['SD'])
PAIRS = np.array([x[0] for x in training_input['pairs'][0]])


missing_idx = np.where(np.isnan(Y_SD))[0]
complete_idx = np.sort(list(set(range(len(Y_SD))) - set(missing_idx)))

Y_toFit = Y_toFit[complete_idx,:]
Y_SD = Y_SD[complete_idx, :]
PAIRS = PAIRS[complete_idx]



Y_SD = 1/Y_SD
Y_SD = (Y_SD - np.min(Y_SD)) / (np.max(Y_SD) - np.min(Y_SD))



