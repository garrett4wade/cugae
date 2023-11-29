clean:
	rm -rf dist build *.egg-info *.so

reinstall:
	pip uninstall cu_gae -y && make clean && pip install -e .