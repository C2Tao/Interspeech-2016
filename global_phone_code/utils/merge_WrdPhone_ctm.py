#make wrd_ctm and phn_ctm have the same content of utterance

language = 'Spanish'
ctm_loc = '/home/allyoushawn/Documents/Global_ali'
check = True

if check == False:
  wrd_ctm = ctm_loc+'/wrd/'+language+'.ctm'
  phone_ctm = ctm_loc+'/phn/'+language+'.ctm'
else:
  wrd_ctm = language+'_wrd.ctm'
  phone_ctm = language+'_phn.ctm'

def utt_equal(utt1, utt2):
  label1 = (utt1.split('_'))[0]
  number1 = int((utt1.split('_'))[1])
  label2 = (utt2.split('_'))[0]
  number2 = int((utt2.split('_'))[1])
  if label1 == label2 and number1 == number2:
    return True

  return False
  

phone_utt_list = []
wrd_utt_list = []
with open(wrd_ctm,'r') as f:
  for line in f.readlines():
    uttID = ((line.rstrip()).split())[0]
    label = (uttID.split('_'))[0]
    number = (uttID.split('_'))[1]
    converted_number = int(number)
    converted_uttID = label+'_'+str(converted_number)
    if converted_uttID not in wrd_utt_list:
      wrd_utt_list.append(converted_uttID)

with open(phone_ctm,'r') as f:
  for line in f.readlines():
    uttID = ((line.rstrip()).split())[0]
    if uttID not in phone_utt_list:
      phone_utt_list.append(uttID)

miss_1 = 0
miss_2 = 0
for uttID in phone_utt_list:
  if uttID not in wrd_utt_list:
    print uttID
    miss_1 += 1

for uttID in wrd_utt_list:
  if uttID not in phone_utt_list:
    miss_2 += 1
if check == True:
  print "miss1 is {}, miss2 is {}".format(miss_1,miss_2)
  quit()

op_wrdCtm = open(language+'_wrd.ctm','w')
op_phnCtm = open(language+'_phn.ctm','w')

with open(wrd_ctm,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    uttID = ((line.rstrip()).split())[0]
    label = (uttID.split('_'))[0]
    number = (uttID.split('_'))[1]
    converted_number = int(number)
    converted_uttID = label+'_'+str(converted_number)
    if converted_uttID in phone_utt_list:
      op_wrdCtm.write(converted_uttID+ line[len(uttID):]+'\n')
    else:
      continue

with open(phone_ctm,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    uttID = ((line.rstrip()).split())[0]
    if uttID in wrd_utt_list:
      op_phnCtm.write(line+'\n')
    else:
      continue

op_wrdCtm.close()
op_phnCtm.close()


