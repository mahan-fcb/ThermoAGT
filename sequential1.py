
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParam import ProtParamData
from quantiprot.metrics.aaindex import get_aa2volume, get_aa2hydropathy
from quantiprot.metrics.basic import average
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.aaindex import get_aaindex_file
from quantiprot.metrics.basic import average
import warnings
warnings.filterwarnings('ignore')
feat0 = Feature(get_aaindex_file('ANDN920101'))
feat1 = Feature(get_aaindex_file('ARGP820101'))
feat2 = Feature(get_aaindex_file('BHAR880101'))
feat3 = Feature(get_aaindex_file('BIGC670101'))
feat4 = Feature(get_aaindex_file('CHAM820102'))
feat5 = Feature(get_aaindex_file('CHOC760102'))
feat6 = Feature(get_aaindex_file('EISD860101'))
feat7 = Feature(get_aaindex_file('FASG760105'))
feat8 = Feature(get_aaindex_file('GRAR740102'))
feat9 = Feature(get_aaindex_file('HUTJ700102'))
feat10 = Feature(get_aaindex_file('JOND750102'))
feat11 = Feature(get_aaindex_file('LEVM760106'))
feat12 = Feature(get_aaindex_file('PRAM900101'))
feat13 = Feature(get_aaindex_file('YUTK870101'))
feat14 = Feature(get_aaindex_file('YUTK870103'))
feat15 = Feature(get_aaindex_file('FASG890101'))
f0 = FeatureSet("my set")
f1 = FeatureSet("my set")
f2 = FeatureSet("my set")
f3 = FeatureSet("my set")
f4 = FeatureSet("my set")
f5 = FeatureSet("my set")
f6 = FeatureSet("my set")
f7 = FeatureSet("my set")
f8 = FeatureSet("my set")
f9 = FeatureSet("my set")
f10 = FeatureSet("my set")
f11 = FeatureSet("my set")
f12 = FeatureSet("my set")
f13 = FeatureSet("my set")
f14 = FeatureSet("my set")
f15 = FeatureSet("my set")
f0.add(feat0)
f1.add(feat1)
f2.add(feat2)
f3.add(feat3)
f4.add(feat4)
f5.add(feat5)
f6.add(feat6)
f7.add(feat7)
f8.add(feat8)
f9.add(feat9)
f10.add(feat10)
f11.add(feat11)
f12.add(feat12)
f13.add(feat13)
f14.add(feat14)
f15.add(feat15)


os.chdir('C:\\Users\\moham\\Downloads\\target\\pdbs\\train')
folder = glob('*')
r1 = []
#d=[]
for i in folder:
    os.chdir('C:\\Users\\moham\\Downloads\\target\\pdbs\\train\\'+str(i)) 

    j=[]
    S = 0
    pdb_files = glob('*.fasta')
    for seq in pdb_files:
        if seq == str(i).upper()+'_RELAXED.fasta':
            #print(seq)
            structure_id = seq[:-14]
            seq1 = load_fasta_file(seq)
            res_seqm0 = f0(seq1)
            resm0 = res_seqm0.columns()
            res_seqm1 = f1(seq1)
            resm1 = res_seqm1.columns()
            res_seqm2 = f2(seq1)
            resm2 = res_seqm2.columns()
            res_seqm3 = f3(seq1)
            resm3 = res_seqm3.columns()
            res_seqm4 = f4(seq1)
            resm4 = res_seqm4.columns()
            res_seqm5 = f5(seq1)
            resm5 = res_seqm5.columns()
            res_seqm6 = f6(seq1)
            resm6 = res_seqm6.columns()
            res_seqm7 = f7(seq1)
            resm7 = res_seqm7.columns()
            res_seqm8 = f8(seq1)
            resm8 = res_seqm8.columns()
            res_seqm9 = f9(seq1)
            resm9 = res_seqm9.columns()
            res_seqm10 = f10(seq1)
            resm10 = res_seqm10.columns()
            res_seqm11 = f11(seq1)
            resm11 = res_seqm11.columns()
            res_seqm12 = f12(seq1)
            resm12 = res_seqm12.columns()
            res_seqm13 = f13(seq1)
            resm13 = res_seqm13.columns()
            res_seqm14 = f14(seq1)
            resm14 = res_seqm14.columns()
            res_seqm15 = f15(seq1)
            resm15 = res_seqm15.columns()

            r1.append([structure_id.upper(), resm0,resm1,resm2,resm3,resm4,resm5,resm6,resm7,resm8,resm9,resm10,resm11,resm12,resm13,resm14,resm15])
S += 1

os.chdir('C:\\Users\\moham\\Downloads\\target\\pdbs\\train\\')
folder = glob('*')
for i in folder:
    os.chdir('C:\\Users\\moham\\Downloads\\target\\pdbs\\train\\'+str(i)) 
    #parser = PDBParser()
    pdb_files = glob('*.fasta')
    S = 0
    for fileName in pdb_files:
        if fileName== str(i).upper()+'_RELAXED.fasta':
            shutil.copy(fileName, 'C:\\Users\\moham\\Downloads\\target\\n')
        

