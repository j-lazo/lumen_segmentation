from scipy.io import loadmat
annots = loadmat('/home/nearlab/Jorge/data/ureteroscopy/Data_IEO/'
                 'labeled_data/ureteroscopy/p_005/useful_frames/'
                 'P_001_12h45m55s-converted/roi.mat')

annots.keys()
#dict_keys(['__header__', '__version__', '__globals__', 'annotations'])
print(annots.keys())

print((annots['__globals__'][0][:]))