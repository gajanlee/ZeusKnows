#!/usr/bin/bash
python generate_net.py --tp=train
python generate_net.py --tp=test
python generate_net.py --tp=dev
