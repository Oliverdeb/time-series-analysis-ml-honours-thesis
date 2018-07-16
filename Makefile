
develop: requirements.txt
	test -d venv || virtualenv --python=/usr/bin/python3 venv
	venv/bin/pip install -Ur requirements.txt
	source venv/bin/activate	

test: @pytest