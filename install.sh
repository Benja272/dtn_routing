#!/bin/bash
python3 brufn-0.0.2/setup.py install && cd spark-3.0.3-bin-hadoop2.7/python/ && python3 setup.py install || true && cd ../../ && pip install -r requirements.txt
