from distutils.core import setup, Extension
import numpy as np

coviar_utils_module = Extension('coviar_p',
		sources = ['coviar_data_loader_p.c'],
		include_dirs=[np.get_include(), '/home/yuqi_huo/anaconda3/include/'],
		extra_compile_args=['-DNDEBUG', '-O3'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-L/home/yuqi_huo/anaconda3/lib/']
)

setup ( name = 'coviar_p',
	version = '0.1',
	description = 'Utils for coviar training.',
	ext_modules = [ coviar_utils_module ]
)
