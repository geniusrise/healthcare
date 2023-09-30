setup:
	@pip install -r ./requirements.txt

test:
	@coverage run -m pytest -v ./tests

publish:
	@python setup.py sdist bdist_wheel
	@twine upload dist/-$-* --verbose
