
#First start with fslvie, load the mri file, and write down coordinates

ssh -X cn1-hpc
freeview
#from FS subjects folder open mri/T1.mgz, find landmarks on volume and write down their TK... coords




ssh -X cn1-hpc
module purge
module load Anaconda3/2020.11
conda activate mne

export LIBGL_DRIVERS_PATH="/home/keyvan.mahjoory/Downloads/lib"
export LD_LIBRARY_PATH="/home/keyvan.mahjoory/Downloads/lib"


mne coreg or mne.gui.coregistration().


#Specify Landmark coordinates based on the previous plot and lock it
#Look up raw meg file (from the first block preferebly) for the "digitization source path" part
#Click on  Fit fiducials
# add meg helmet and see how it fits
#See how those two look like
# Save the transformation matrix