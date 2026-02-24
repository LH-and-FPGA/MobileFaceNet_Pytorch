BATCH_SIZE = 512
SAVE_FREQ = 1
TEST_FREQ = 1
TOTAL_EPOCH = 70

RESUME = ''
SAVE_DIR = './model'
MODEL_PRE = 'CASIA_B512_'


CASIA_DATA_DIR = '/home/han/data/CASIA'
LFW_DATA_DIR = '/home/han/data/lfw'

GPU = 0

# Model file: 'model' (original PReLU) or 'model_lh' (ReLU + GAP, FPGA-friendly)
MODEL_FILE = 'model_lh'

# Model size:
#   'tiny'     -- only for model_lh (default tiny setting, ~98k params)
#   'small'    -- both model files  (Mobilefacenet_small_setting, inplanes=32, mid_channels=256)
#   'original' -- both model files  (Mobilefacenet_bottleneck_setting, inplanes=64, mid_channels=512)
MODEL_SIZE = 'small'
