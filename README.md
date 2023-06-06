# Neuroflex


### check the analysis status
from nf_tools import utils  
utils.status(subj_ids=range(1, 29))

### Copy data from the server to the local machine and organize them in the right folder structure
from src import utils
utils.cat_cp_meg_blocks(src='',
                        dest='', 
                        name_pattern='')

### Annotate blocks based on Behavioral data and MEG stimulus channel
from src import utils
utils.annotate_blocks(subjs_dir='datasets/data/', subj_id=25, write=True)