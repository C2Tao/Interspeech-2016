# Check if the ctm is sorted and it's correct.
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
  

language_set = [ 'Czech' ,'French','German','Spanish']
val_type_set = ['phn','wrd']
for language in language_set:
  for val_type in val_type_set:
    loc = '/home/allyoushawn/Documents/STD_globalPhone/{}'.format(language)
    ctm_file = '{}/{}_{}.ctm'.format(loc,language,val_type)
    f = open(ctm_file,'r')
    line = f.readline()
    line = line.rstrip()
    previous_uttID = (line.split())[0]
    previous_startPoint = (line.split())[2]
    
    for line in f.readlines():
      line = line.rstrip()
      uttID = (line.split())[0]
      startPoint = (line.split())[2]
    
      if uttID == previous_uttID:
        assert(float(previous_startPoint) < float(startPoint))
        previous_startPoint = startPoint
    
      if uttID != previous_uttID:
        if compare_uttid(uttID,previous_uttID) == True:
          print "uttID {}, previous_uttID {}".format(uttID,previous_uttID)
        assert(compare_uttid(uttID,previous_uttID) == False)
        previous_uttID = uttID
        previous_startPoint = startPoint
        #assert(float(startPoint) == 0)
    
    print "Done Validation {}_{}".format(language,val_type)
    f.close()

file_type_set = ['.utt','.wav.scp','.text']
for language in language_set:
  for file_type in file_type_set:
    loc = '/home/allyoushawn/Documents/STD_globalPhone/{}'.format(language)
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
