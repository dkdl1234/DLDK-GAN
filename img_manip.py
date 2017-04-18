from PyPDF2 import PdfFileWriter, PdfFileReader
from scipy.misc import imread
from os import listdir
from os.path import abspath, splitext, join, isfile, basename
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def append_pdf(inputs, outputs):
	[outputs.addPage(inputs.getPage(page_num)) for page_num in range(inputs.numPages)]



def merge(files, out_path):
	output = PdfFileWriter()
		
	[append_pdf(PdfFileReader(open(file_path,'rb')), output) for file_path in files]	
	
	output.write(open(out_path, "wb"))


def convert_png_to_pdf(fake_files_names,real_file_name, saving_directory):
	files = [(file_ + ".png", join(saving_directory, basename(file_) + ".pdf")) for file_ in fake_files_names]
	real_img = imread(real_file_name + ".png")
	counter = 0
	for (png_file, pdf_file) in files:
		#read the fake file
		fake_img = imread(png_file)
		
		#fuze the fake and real image
		fuzed = np.concatenate((real_img, fake_img), axis=1)
		
		#plot the data		
		plot = plt.figure(counter)
		plt.title("Real-Fake, epoch %d" % counter)
		_ = plt.imshow(fuzed)

		#save the data to a pdf file
		pp = PdfPages(pdf_file + "." + str(counter))
		pp.savefig(plot, pdi=300, transparent=True)
		pp.close()

		counter += 1

	pdf_files = [pdf for (_, pdf) in files]
	indx = [i for i in range(len(pdf_files))]
	ziped = zip(pdf_files, indx)
	return [pdf_file + "." + str(i) for (pdf_file, i) in ziped]



def merge_image_summaries(real_dir, fake_dir, targ_dir):
	
	#get all the images indices from the real image directory
	img_names = [splitext(file_)[0] for file_ in listdir(real_dir) if ".png" in file_]
	fake_dirs = listdir(fake_dir)
	fake_dirs = [join(fake_dir, dir_) for dir_ in fake_dirs]
	

	tmp_pdf_directory = join(targ_dir, "tmp")
	if not os.path.exists(tmp_pdf_directory):
		os.makedirs(tmp_pdf_directory)

	#for each indexed image:
	for file_ in img_names:
		#getting all the files with the same generated image in them (from epoch to epoch)
		epoch_files = [join(epoch_dir, file_) for epoch_dir in fake_dirs]

		#get the path of the real file (sould be the same name, different path) 
		real_file = join(real_dir, file_)

		#convert all png files to pdfs
		pdf_files = convert_png_to_pdf(epoch_files, real_file, tmp_pdf_directory)

		#merge all the pdf files into a single summary
		merge(pdf_files, join(targ_dir, "Summary-" + basename(file_) + ".pdf"))
		
		#clean all the partial pdf's from the tmp directory
		for pdf_file in pdf_files:
			if isfile(pdf_file):
				try:
					os.unlink(pdf_file)
				except Exception as p:
					print (p)

	try:
		shutil.rmtree(tmp_pdf_directory)

	except Exception as c:
		pass



def convert_png_to_mp4(fake_files_names,real_file_name, name, saving_directory):
	'''
		Create a single mp4 summary from png image sequence.
	'''
	import imageio
	
	files = [file_ + ".png" for file_ in fake_files_names]
	real_img = imread(real_file_name + ".png")
	counter = 0
	
	seq_name = name + ".mp4"
	with imageio.get_writer(os.path.join(saving_directory, seq_name), mode='I') as writer:
		for png_file in files:
			#read the fake file
			fake_img = imread(png_file)
		
			#fuze the fake and real image
			fuzed = np.concatenate((real_img, fake_img), axis=1)
		
			#add the fuzed file to the writer
			writer.append_data(fuzed)



def merge_mp4s(real_dir, fake_dir, targ_dir):
	'''
		Create an mp4 summary video presenting fake images near to real images
		This requires a directory that holds the real images, and directories that hold the fake images
	'''
	#get all the images indices from the real image directory
	img_names = [splitext(file_)[0] for file_ in listdir(real_dir) if ".png" in file_]
	fake_dirs = listdir(fake_dir)
	fake_dirs = [join(fake_dir, dir_) for dir_ in fake_dirs]
	
	tmp_pdf_directory = join(targ_dir, "tmp")
	if not os.path.exists(tmp_pdf_directory):
		os.makedirs(tmp_pdf_directory)

	for file_ in img_names:
		#getting all the files with the same generated image in them (from epoch to epoch)
		epoch_files = [join(epoch_dir, file_) for epoch_dir in fake_dirs]

		#get the path of the real file (sould be the same name, different path) 
		real_file = join(real_dir, file_)

		#create the mp4 file
		convert_png_to_mp4(epoch_files, real_file, file_, targ_dir)
