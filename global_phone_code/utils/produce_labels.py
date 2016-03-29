import sys

corpus_path=sys.argv[1]
language=sys.argv[2]
query_idx = int(sys.argv[3])

#corpus_path = '/home/allyoushawn/Documents/STD_globalPhone/Czech'

queries = []
with open(corpus_path+'/chosen_queries','r') as f:
  for line in f.readlines():
    line = line.rstrip()
    queries.append(line)

#query_idx = 0;
for target in ['train','dev','test']:
  selected_query = queries[:query_idx]
  print selected_query
    
  transcription_path = corpus_path+'/selected/wrd_corpus/{}/{}_{}_wrd.text'.format(target,target,language)
  label = open('{}/selected/wrd_corpus/{}/{}_{}_query{}_label'.format(corpus_path,target,target,language,query_idx),"w")
  
  with open(transcription_path,"r") as f:
    for line in f.readlines():
      line = line.rstrip()
      segs = line.split()
      label.write(segs[0]+ ' ');
      if segs[1] in selected_query:
        label.write('1\n')
      else:
        label.write('0\n')
  
  label.close()
  #query_idx += 1



