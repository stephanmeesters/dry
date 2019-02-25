from dry import Dry

drys = Dry()

# model = drys.train_model('20180719_104240DTI150050SENSEs301a1003.bval')
# drys.save_model(model, 'model.h5')

model = drys.load_model('model.h5')

drys.fwe(['/Users/stephan/Research/2.PostDoc/Data/Pat1/nifti/dti/mrtrix/free-water-correction/DWI_proc.nii.gz'], 
		 model, 
		 '/Users/stephan/Research/2.PostDoc/Data/Pat1/nifti/dti/mrtrix/free-water-correction/20180719_104240DTI150050SENSEs301a1003.bval',
		 '/Users/stephan/Research/2.PostDoc/Data/Pat1/nifti/dti/mrtrix/free-water-correction/'
		 # '/Users/stephan/Research/2.PostDoc/Data/Pat1/nifti/dti/mrtrix/free-water-correction/rtumor_mask.nii'
)
