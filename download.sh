mkdir -p data/ data/train data/test
wget -N http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar -P ./data/train/
wget -N http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar -P ./data/train/

wget -N http://pjreddie.com/media/files/VOC2012test.tar -P ./data/test/
wget -N http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar -P ./data/test/

curr_dir=$(pwd)
# untar train files
cd data/train
for f in `find . -type f -name "*.tar"`
do
    tar -xvf $f 
done
rm *.tar

# untar test files
cd $curr_dir
cd data/test
for f in `find . -type f -name "*.tar"`
do
    tar -xvf $f
done
rm *.tar

