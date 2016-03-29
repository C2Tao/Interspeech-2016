import sys

#From ctm to get valid uttID

#loc = '/home/allyoushawn/Documents/STD_globalPhone'
loc=sys.argv[1]
language=sys.argv[2]

valid_utt = []
op = open('{}/{}.utt'.format(loc,language,language),'w')
with open('src/{}/{}_phn.ctm'.format(language,language),'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    utt = (line.split())[0]
    if utt not in valid_utt:
      valid_utt.append(utt)
      op.write(utt+'\n')
op.close()


