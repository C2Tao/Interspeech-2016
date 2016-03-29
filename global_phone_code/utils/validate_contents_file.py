import sys
loc=sys.argv[1]
language = sys.argv[2]

def compare_uttid(utt1,utt2): #return True if utt1 < utt2
  sentenceID1 = ((utt1.split('_'))[0])[2:]
  sentenceID2 = ((utt2.split('_'))[0])[2:]
  wavID1 = (utt1.split('_'))[1]
  wavID2 = (utt2.split('_'))[1]

  language1 = utt1[:2]
  language2 = utt2[:2]
  assert(language1 == language2)
  if int(sentenceID1) < int(sentenceID2):
    return True
  elif int(sentenceID1) > int(sentenceID2):
    return False

  if int(wavID1) < int(wavID2):
    return True
  elif int(wavID1) > int(wavID2):
    return False

  print "Error in comparison"
  quit()
  


file_type_set = ['.utt','.wav.scp','.text']
for file_type in file_type_set:
  file_beVal = '{}/{}{}'.format(loc,language,file_type)
  f = open(file_beVal,'r')
  line = f.readline()
  line = line.rstrip()
  previous_uttID = (line.split())[0]
  
  for line in f.readlines():
    line = line.rstrip()
    uttID = (line.split())[0]
  
    if uttID != previous_uttID:
      if compare_uttid(uttID,previous_uttID) == True:
	print "uttID {}, previous_uttID {}".format(uttID,previous_uttID)
      assert(compare_uttid(uttID,previous_uttID) == False)
      previous_uttID = uttID
  
  print "Done Validation {}{}".format(language,file_type)
  f.close()
