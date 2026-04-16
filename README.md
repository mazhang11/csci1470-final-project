# csci1470-final-project

////// DATA PROCESSING STEP //////
Preprocessed data gotten from: http://preprocessed-connectomes-project.org/adhd200/

./scripts/download_adhd200.py accesses the open source pre-processed (noise from MRI scan like skull, head movement, heartbeat removed) data from the ADHD200 from the Amazon S3 bucket. 

./scripts/run_downloads.sh bash script that runs the download and puts it in our data directory





Citations:
  Pierre Bellec, Carlton Chu, François Chouinard-Decorte, Yassine Benhajali, Daniel S. Margulies, R. Cameron Craddock (2017). The Neuro Bureau ADHD-200 Preprocessed repository. NeuroImage, 144, Part B, pp. 275 - 286. doi:10.1016/j.neuroimage.2016.06.034
