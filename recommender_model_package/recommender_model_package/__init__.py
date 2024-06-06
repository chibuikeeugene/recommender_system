import os

from recommender_model_package.recommender_model_package.config.core import PACKAGE_ROOT


with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()