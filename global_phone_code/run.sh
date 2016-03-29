#Input: global phone corpus location and language
#src: phone and wrd ctm files
#output: ../std_global_phone

# Usage: bash run.sh <language> </global/phone/corpus/path>
# e.g. bash run.sh Czech /media/hdd/ubuntuData/corpus
language='Czech'
corpus_loc='/home/c2tao'

#Config
#set up directory
dir="/home/c2tao/interspeech_2016/global_phone_std/$language"
ori_header="/home/speech/share/corpus"

query_num=30
start_query_idx=100  #we sort our wrds, from frequency high to low. We ignore the wrds before the idx.

#Should be specified as absolute path
wrd_corpus="/home/c2tao/interspeech_2016/global_phone_wrd"  #The location where to put wrd wav files.
wrd_feats="/home/c2tao/interspeech_2016/global_phone_feat/${language}_wrd_feats" #The location where to put wrd features.
#End of config

mkdir -p $dir

echo "### Phase 1: Basic file extraction ###"
#get utt_id_set
echo ""
echo "Generating .utt"
python utils/get_utt.py $dir $language

#get wav.scp
#Note: needs to know ori header of original wav.scp, e.g. /ori/header/GlobalPhone
#Usage: pyhton get_wavSCP.py $dir $language ori_header
echo ""
echo "Generating .wav.scp"
python utils/get_wav_scp.py $dir $language /home/speech/share/corpus $corpus_loc

#get text
echo ""
echo "Generating .text"
python utils/get_text.py $dir $language $corpus_loc

#sort files
echo "Sorting .text, .wav.scp, .utt files"
bash utils/sort_files.sh $dir

#validate files in sorted order
echo "File validation, check their uttID orders"
python utils/validate_contents_file.py $dir $language

### Phase 1 Done. There shoud be three sorted files including .text, .wav.scp, .utt ####
echo "### Phase 1 Done. ###"

echo "### Phase 2: get chosen queries, and the refined corpus ###"

#get text needed by tf-idf.py
echo "Generating files needed by utils/tfidf.py"
python utils/get_tfidf_text.py $dir $language

#get lexicon
echo "Copy lexicon from original global phone corpus"
cp $corpus_loc/GlobalPhone/$language/material/lexicon.word.txt $dir/

#get chosen_queries
echo "Choosing proper queries"
python utils/tfidf.py $dir $language $query_num $start_query_idx

#get refined corpus
echo "According to the chosen queries, refine our original corpus"
python utils/refine_sets.py $dir $language

echo "### Phase 2 Done ###"
echo""
echo "### Phase 3: create wrd corpus ###"

#needs specify the wrd corpus location
#Usage: python utils/wrd_corpus.py $dir $language </location/of/wrd/corpus>
echo "Create wrd corpus files."
python utils/wrd_corpus.py $dir $language $wrd_corpus

wrd_file_loc=$dir/selected/wrd_corpus
#split the wrd_corpus into train, dev, test according to 8:1:1
#Usage: bash split_sets.sh </path/to/target/wrd_files> <language>

echo "Spliting .text, .utt, .wav.scp files into train, dev, test"
bash utils/split_sets.sh $wrd_file_loc $language

#extract wrd features
echo "Extract wrd features."
bash utils/keep_order_extract_feat.sh $wrd_file_loc $language $wrd_feats

#produce labels
query_idx=0 #The target query
echo "Producing labels for query_$query_idx"
python utils/produce_label.py $dir $language $query_idx

echo "### Phase 3 Done ###"
