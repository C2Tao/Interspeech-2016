#sort files' contents
loc=$1

for file in $(ls $loc); do
  rm -rf $loc/sorted_* 
  sort -k1,1V -k2,3V $loc/$file >$loc/sorted_$file
  mv $loc/sorted_$file $loc/$file
done
