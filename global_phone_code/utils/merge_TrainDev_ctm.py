# Merge phone ali of dev.ctm into train.ctm

def compare_uttid(utt1,utt2):
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
  

ctm_loc ='/home/allyoushawn/Documents/Global_ali/phn_ctm'
language = 'Czech'
ctm_loc = ctm_loc + '/'+language
dev_f = open(ctm_loc+'/dev.ctm','r')
train_f = open(ctm_loc+'/train.ctm','r')
new_train_f = open(language+'_train.ctm','w')

dev_line = (dev_f.readline()).rstrip()
dev_uttID = (dev_line.split())[0]
dev_end = False

for train_line in train_f.readlines():
  train_line = train_line.rstrip()
  train_uttID = (train_line.split())[0]

  while compare_uttid(dev_uttID , train_uttID) == True and dev_end != True:
    new_train_f.write(dev_line+'\n')
    dev_line = dev_f.readline()
    if not dev_line:
      dev_end = True
      break
    dev_line = dev_line.rstrip()
    dev_uttID = (dev_line.split())[0]
  
  if dev_end == False:
    assert( compare_uttid(dev_uttID , train_uttID) == False)
  
  new_train_f.write(train_line+'\n')

while dev_end == False:
  new_train_f.write(dev_line+'\n')
  dev_line = dev_f.readline()
  if not dev_line:
    dev_end = True
    break
  dev_line = dev_line.rstrip()

dev_f.close()
train_f.close()
new_train_f.close()
