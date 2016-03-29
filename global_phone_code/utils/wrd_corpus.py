import subprocess
import sys
from IPython.core.debugger import Tracer

loc = sys.argv[1]
language=sys.argv[2]
wrd_corpus_wav_loc =sys.argv[3]

subprocess.call('mkdir -p {}/{}/wav'.format(wrd_corpus_wav_loc,language),shell=True)
query_set = []
with open(loc+'/chosen_queries','r') as f:
  for line in f.readlines():
    query = line.rstrip('\n')
    query_set.append(query)

selected_loc = loc + '/selected'
wrd_corpus_loc = selected_loc + '/wrd_corpus'
subprocess.call('mkdir -p {}'.format(wrd_corpus_loc),shell=True)
uttID = ''
utt_query_count = 1

wrd_text_f = open(wrd_corpus_loc+'/{}_wrd.text'.format(language),'w')
wrd_wav_f = open(wrd_corpus_loc+'/{}_wrd.wav.scp'.format(language),'w')
wrd_utt_f = open(wrd_corpus_loc+'/{}_wrd.utt'.format(language),'w')
ori_wav_scp = open(selected_loc+'/{}.wav.scp'.format(language),'r')
ori_line = (ori_wav_scp.readline()).rstrip('\n')
with open(selected_loc + '/{}_wrd.ctm'.format(language), 'r') as f:
  for line in f.readlines():
    segs = (line.rstrip('\n')).split()

    wrd = segs[-1]
    if wrd in query_set:
      
      if segs[0] != uttID:
	uttID = segs[0]
	utt_query_count = 1

      while uttID != (ori_line.split())[0]:
	ori_line = (ori_wav_scp.readline()).rstrip('\n')
	if not ori_line:
	  print "end of original scp file, exit..."
	  ori_wav_scp.close()
	  exit()
	if (ori_line.split())[0] == uttID:
	  break

      #write the query into wrd_corpus_loc file and cut its wav out.
      wrd_uttID = uttID + '_wrd_' + str(utt_query_count)
      wrd_utt_f.write(wrd_uttID+'\n')
      wrd_text_f.write(wrd_uttID + ' '+wrd + '\n')
      wrd_wav_loc = '{}/{}/wav/{}.wav'.format(wrd_corpus_wav_loc,language,wrd_uttID)
      wrd_wav_f.write(wrd_uttID + ' '+ wrd_wav_loc +'\n')
      utt_query_count += 1        

      #do sox
      assert((ori_line.split())[0] == uttID)
      ori_wav_loc = (ori_line.split())[1]
      start_time = segs[2]
      duration = segs[3]
      command = 'sox {} {} trim {} {}'.format(ori_wav_loc,wrd_wav_loc,start_time,duration)
      subprocess.call(command,shell=True)


      
wrd_text_f.close()
wrd_wav_f.close()
wrd_utt_f.close()
ori_wav_scp.close()

#build query table
query_utt_map = {}
for query in query_set:
  query_utt_map[query] = []

with open(wrd_corpus_loc+'/{}_wrd.text'.format(language),'r') as f:
  for line in f.readlines():
    line = line.rstrip('\n')
    query = (line.split())[1]
    utt = (line.split())[0]
    query_utt_map[query].append(utt)

q2u_f = open(wrd_corpus_loc+'/{}.query2utt'.format(language),'w')
for query in query_utt_map.keys():
  q2u_f.write(query)
  for utt in query_utt_map[query]:
    q2u_f.write(' '+utt)
  q2u_f.write('\n')
  
