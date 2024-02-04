import itertools
import numpy as np

def onehot(sequence):
  onehot=[]
  for i in range(len(sequence)):
    if(sequence[i]=="A"):
      onehot.append([1,0,0,0])
    if(sequence[i]=="C"):
      onehot.append([0,1,0,0])
    if(sequence[i]=="G"):
      onehot.append([0,0,1,0])
    if(sequence[i]=="T"):
      onehot.append([0,0,0,1])
    if(sequence[i]=="N"):
      onehot.append([0,0,0,0])
  return onehot;





names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
name=names[0]
train_dir="data/GM12878/"

test_dir='data/GM12878/'
Data_dir='data/GM12878/'
print ('Experiment on %s dataset' % name)

print ('Loading seq data...')
enhancers_tra=open(train_dir+'%s_enhancer.fasta'%name,'r').read().splitlines()[1::2]
promoters_tra=open(train_dir+'%s_promoter.fasta'%name,'r').read().splitlines()[1::2]
y_tra=np.loadtxt(train_dir+'%s_label.txt'%name)

enhancers_tes=open(test_dir+'%s_enhancer_test.fasta'%name,'r').read().splitlines()[1::2]
promoters_tes=open(test_dir+'%s_promoter_test.fasta'%name,'r').read().splitlines()[1::2]
y_tes=np.loadtxt(test_dir+'%s_label_test.txt'%name)

print('?????')
print('pos_samples:'+str(int(sum(y_tra))))
print('neg_samples:'+str(len(y_tra)-int(sum(y_tra))))
print('??????')
#print('pos_samples:'+str(int(sum(y_imtra))))
#print('neg_samples:'+str(len(y_imtra)-int(sum(y_imtra))))
#print('???')
print('pos_samples:'+str(int(sum(y_tes))))
print('neg_samples:'+str(len(y_tes)-int(sum(y_tes))))


# In[ ]:
print(np.array(enhancers_tra).shape)
print(enhancers_tra[0])
print(np.array(promoters_tra).shape)
print(promoters_tra[0])

# X_en_tra=[]
# X_pr_tra=[]
X_en_tra = np.empty((len(enhancers_tra), 3000, 4))
X_pr_tra = np.empty((len(promoters_tra), 2000, 4))
for i in range(len(enhancers_tra)):
    onehot_en=onehot(enhancers_tra[i])
    onehot_pr=onehot(promoters_tra[i])
    print(i)

    # X_en_tra.append(onehot_en)
    # X_pr_tra.append(onehot_pr)
    X_en_tra[i] = onehot_en
    X_pr_tra[i] = onehot_pr


# X_en_tes=[]
# X_pr_tes=[]
X_en_tes = np.empty((len(enhancers_tes), 3000, 4))
X_pr_tes = np.empty((len(promoters_tes),2000, 4))
for i in range(len(promoters_tes)):
    onehot_en_test = onehot(enhancers_tes[i])
    onehot_pr_test = onehot(promoters_tes[i])
    # X_en_tes.append(onehot_en_test)
    # X_pr_tes.append(onehot_pr_test)
    X_en_tes[i] = onehot_en_test
    X_pr_tes[i] = onehot_pr_test


print(np.array(X_en_tra).shape)
print(np.array(X_en_tes).shape)
print(np.array(X_pr_tra).shape)
print(np.array(X_pr_tes).shape)
#
# X_en_tra,X_pr_tra=get_data(enhancers_tra,promoters_tra)
# X_en_imtra,X_pr_imtra=get_data(im_enhancers_tra,im_promoters_tra)
# X_en_tes,X_pr_tes=get_data(enhancers_tes,promoters_tes)
# print(X_en_tra.shape)
np.savez(Data_dir+'%s_train.npz'%name,X_en_tra=X_en_tra,X_pr_tra=X_pr_tra,y_tra=y_tra)
# np.savez(Data_dir+'im_%s_train.npz'%name,X_en_tra=X_en_imtra,X_pr_tra=X_pr_imtra,y_tra=y_imtra)
np.savez(Data_dir+'%s_test.npz'%name,X_en_tes=X_en_tes,X_pr_tes=X_pr_tes,y_tes=y_tes)

