import sys
loc=sys.argv[1]
language = sys.argv[2]

# -*- coding: utf-8 -*-
text_loc = loc+'/{}.text'.format(language)
op = open(loc+'/{}_plainForTf.text'.format(language),'w')
text = ''
with open(text_loc,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    blank_idx = line.index(' ')
    text = text + line[blank_idx+1:] + ' '
op.write(text)
op.close()

op = open(loc+'/{}_plain.text'.format(language),'w')
with open(text_loc,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    blank_idx = line.index(' ')
    op.write(line[blank_idx+1:]+ '\n')
op.close()
  
  
