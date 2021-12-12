import os,codecs,re

def main(dir_,out_path):
	emails = []
	for i in range(len(os.listdir(dir_))-1):
		content = codecs.open(os.path.join(dir_,str(i)+'.txt'), 'r',encoding='utf-8',errors='ignore').readlines()
		for line in content:
			match = re.findall(r'[\w\.-]+@[\w\.-]+', line)
			if len(match) > 0:
				emails.append(match[0].lower().strip())
			else:
				emails.append('')

	with codecs.open(out_path,'w',encoding='utf-8',errors='ignore') as f:
		for email in emails[:-1]:
			f.write(email+'\n')
		if emails[-1]=='':
			f.write('\n')
		else:
			f.write(emails[-1])

if __name__ == '__main__':
	main('../data/compiled_bios/','../data/emails')


	   