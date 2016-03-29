#split the sets to train, dev, test according to the ratio of 8:1:1

loc=$1
language=$2

cd $loc
mkdir  -p train dev test
for file in ${language}_wrd.text ${language}_wrd.utt ${language}_wrd.wav.scp; do
    cat $file | sed -e '9~10d' -e '10~10d' > train/train_$file 
    cat $file | sed -e '1~10d' -e '2~10d' -e '3~10d' -e '4~10d' -e '5~10d' -e '6~10d' -e '7~10d' -e '8~10d' -e '10~10d' > dev/dev_$file 
    cat $file | sed -e '1~10d' -e '2~10d' -e '3~10d' -e '4~10d' -e '5~10d' -e '6~10d' -e '7~10d' -e '8~10d' -e '9~10d' > test/test_$file 
done


