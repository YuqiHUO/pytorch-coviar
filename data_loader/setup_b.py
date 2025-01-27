from distutils.core import setup, Extension
import numpy as np

coviar_utils_module = Extension('coviar_b',
		sources = ['coviar_data_loader_b.c'],
		include_dirs=[np.get_include(), '/home/yuqi_huo/lab/FFmpeg_build/include/'],
		extra_compile_args=['-DNDEBUG', '-O3'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-L/home/yuqi_huo/lab/FFmpeg_build/lib/']
)

setup ( name = 'coviar_b',
	version = '0.1',
	description = 'Utils for coviar training.',
	ext_modules = [ coviar_utils_module ]
)
