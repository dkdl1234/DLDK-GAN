from img_manip import merge_image_summaries, merge_mp4s
import argparse
from os.path import join

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('img_dir')
	parser.add_argument('opt')
	args = parser.parse_args()

	dest_dir = args.img_dir
	fake_imgs_dir = join(args.img_dir, 'fake')
	real_imgs_dir = join(args.img_dir, 'real')
	
	if args.opt == "video":
		merge_mp4s(real_imgs_dir, fake_imgs_dir, dest_dir)

	elif args.opt == "summary":
		merge_image_summaries(real_imgs_dir, fake_imgs_dir, dest_dir)

	else:
		print("opt should be 'video' or 'summary'")
