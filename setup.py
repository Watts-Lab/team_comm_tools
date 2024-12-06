from setuptools import setup
from setuptools.command.build_py import build_py
import toml
import os

def get_version_from_pyproject():
    with open("pyproject.toml", "r") as f:
        pyproject_data = toml.load(f)
    return pyproject_data["project"]["version"]

version = get_version_from_pyproject()
version_file = os.path.join(os.path.dirname(__file__), "src/team_comm_tools", "_version.py")
if not os.path.exists(version_file):
    with open(version_file, "w") as f:
        f.write(f'__version__ = "{version}"\n')

class BuildPyCommand(build_py):
    def run(self):
        # Generate _version.py before building, which helps automatically update the .__version__ parameter
        version = get_version_from_pyproject()
        version_path = os.path.join(self.build_lib, "src/team_comm_tools", "_version.py")
        os.makedirs(os.path.dirname(version_path), exist_ok=True)
        with open(version_path, "w") as f:
            f.write(f'__version__ = "{version}"\n')
        super().run()

setup(
    cmdclass={
        'build_py': BuildPyCommand,
    }
)