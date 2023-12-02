clean:
	rm -rf dist build *.egg-info *.so

reinstall:
	pip uninstall cugae -y && make clean && pip install -e .