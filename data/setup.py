from distutils.core import setup, Extension
import numpy as np

coviar_utils_module = Extension('coviexinfo',
		sources = ['test1.c'],
		include_dirs=[np.get_include(), '/home/sjhu/env/ffmpeg_gjy/include/'],
		extra_compile_args=['-DNDEBUG', '-O3'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-L/home/sjhu/env/ffmpeg_gjy/lib']
)

setup ( name = 'coviexinfo',
	version = '0.1',
	description = 'Utils for coviar training.',
	ext_modules = [ coviar_utils_module ]
)
