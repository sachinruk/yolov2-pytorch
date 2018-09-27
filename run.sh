nvidia-docker run -d -p 8888:8888 -p 6006:6006 -v ${PWD}:/notebook -v ${PWD}/data/:/notebook/data sachinruk/pytorch_gpu

