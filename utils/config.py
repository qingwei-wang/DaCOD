import os 

backbone_path = './backbone/resnet50-19c8e357.pth'

datasets_root = './datasets'

depth_cod_training_root = os.path.join(datasets_root,'train/cod10k_depth_train')

camo_path = os.path.join(datasets_root,'test/CAMO_depth')

chameleon_path = os.path.join(datasets_root,'test/CHAMELEON_depth')

cod10k_path = os.path.join(datasets_root,'test/cod10k_depth_test')

NC4K_path = os.path.join(datasets_root,'test/NC4K')

plantcamo_path = './datasets1250/PlantCAMO1250'
