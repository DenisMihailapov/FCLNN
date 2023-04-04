mkdir data
# shellcheck disable=SC2164
cd data
wget -c http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -c http://ufldl.stanford.edu/housenumbers/test_32x32.mat
