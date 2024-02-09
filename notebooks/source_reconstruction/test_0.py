
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--subj_ids', action='store', type=int, nargs=2)
my_parser.add_argument('--dim_red', action='store', type=str, nargs='*', default=['ssd', 'pca'])
args = my_parser.parse_args()
subjects_id = range(args.subj_ids[0], args.subj_ids[1])
dim_red = args.dim_red

print(subjects_id)
print(dim_red)