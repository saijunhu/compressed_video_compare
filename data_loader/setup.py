from distutils.core import setup, Extension
import numpy as np

#this file is a script used for Disutil Module set c extension

FFMPEG_PATH_INCLUDE = '/home/husaijun/storage/anaconda3/include'
FFMPEG_PATH_LIB = '/home/husaijun/storage/anaconda3/lib'

coviar_utils_module = Extension('coviar',
		sources = ['coviar_data_loader.c'],
		include_dirs=[np.get_include(), FFMPEG_PATH_INCLUDE],
		extra_compile_args=['-DNDEBUG', '-O3'],
		library_dirs= [FFMPEG_PATH_LIB],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale']
)

setup ( name = 'coviar',
	version = '0.1',
	description = 'Utils for coviar training.',
	ext_modules = [ coviar_utils_module ]
)
